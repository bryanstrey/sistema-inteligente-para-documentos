"""Baixa de boletos: liga um comprovante ao boleto que ele pagou.

Quando chega um comprovante, procuramos boletos em aberto que ele pode ter pago:
  - a linha digitável do boleto aparece no texto do comprovante -> é ele, sem dúvida
  - mesmo valor + mesmo beneficiário + data perto do vencimento + operação de pagamento de boleto,
    e só um boleto nessa situação -> baixa automática
  - só o valor bate (ou há mais de um candidato) -> vira sugestão; a pessoa confirma no painel
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from pipeline import banco
from pipeline.validacao import normalizar, parse_data, parse_valor, so_digitos

# janela de datas aceita entre o pagamento e o vencimento (pagar adiantado é mais comum que atrasado)
DIAS_ANTES_VENCIMENTO = 45
DIAS_DEPOIS_VENCIMENTO = 30
PALAVRAS_GENERICAS = {"ltda", "eireli", "s/a", "s.a.", "sa", "me", "epp", "cia", "de", "da", "do", "das", "dos"}


@dataclass
class Candidato:
    boleto_id: int
    beneficiario: str
    valor: float | None
    vencimento: object
    linha_confere: bool
    nome_confere: bool
    motivo: str


def _palavras(nome) -> set[str]:
    return {p for p in normalizar(nome or "").replace(",", " ").split() if len(p) > 2 and p not in PALAVRAS_GENERICAS}


def nomes_conferem(a, b) -> bool:
    pa, pb = _palavras(a), _palavras(b)
    if not pa or not pb:
        return False
    return len(pa & pb) / min(len(pa), len(pb)) >= 0.5


def candidatos(campos: dict, texto_ocr: str) -> list[Candidato]:
    """Boletos em aberto que este comprovante pode ter pago, do mais provável para o menos."""
    valor = parse_valor(campos.get("valor"))
    data = parse_data(campos.get("data"))
    digitos = so_digitos(texto_ocr)
    out = []
    for _, b in banco.boletos_em_aberto().iterrows():
        c = b["campos"] or {}
        linha = so_digitos(c.get("linha_digitavel"))
        linha_confere = len(linha) == 47 and linha in digitos
        vb, venc = parse_valor(c.get("valor")), parse_data(c.get("vencimento"))
        valor_confere = valor is not None and vb is not None and abs(valor - vb) <= 0.01
        nome_confere = nomes_conferem(c.get("beneficiario_nome"), campos.get("recebedor_nome"))
        data_confere = not (data and venc) or -DIAS_ANTES_VENCIMENTO <= (data - venc).days <= DIAS_DEPOIS_VENCIMENTO
        if not linha_confere and not (valor_confere and data_confere):
            continue
        if linha_confere:
            motivo = "linha digitável do boleto aparece no comprovante"
        else:
            motivo = "mesmo valor" + (" e mesmo beneficiário" if nome_confere else "") + (" e data perto do vencimento" if data and venc else "")
        out.append(Candidato(int(b["id"]), c.get("beneficiario_nome") or "?", vb, venc, linha_confere, nome_confere, motivo))
    return sorted(out, key=lambda x: (not x.linha_confere, not x.nome_confere))


def conciliar(comprovante_id: int) -> tuple[Candidato | None, list[Candidato]]:
    """Dá baixa automática quando o casamento é seguro. Devolve (boleto baixado, sugestões para a pessoa confirmar)."""
    doc = banco.obter(comprovante_id)
    if not doc or doc["tipo_final"] != "comprovante" or banco.boleto_pago_por(comprovante_id):
        return None, []
    campos = doc["campos"] or {}
    cands = candidatos(campos, doc["texto_ocr"] or "")
    if not cands:
        return None, []
    melhor = cands[0]
    operacao_ok = campos.get("tipo_operacao") in (None, "", "pagamento_boleto")
    fortes = [c for c in cands if c.nome_confere]
    seguro = melhor.linha_confere or (operacao_ok and melhor.nome_confere and len(fortes) == 1)
    if seguro:
        banco.marcar_pago(melhor.boleto_id, parse_data(campos.get("data")), comprovante_id)
        return melhor, []
    return None, cands


def comprovantes_livres() -> pd.DataFrame:
    """Comprovantes que ainda não deram baixa em nenhum boleto."""
    df = banco.listar(["salvo", "revisado"])
    usados = set(df.pago_por.dropna().astype(int))
    return df[(df.tipo_final == "comprovante") & ~df.id.isin(usados)]


def conciliar_boleto_novo(boleto_id: int) -> int | None:
    """Boleto chegou depois do comprovante: tenta de novo a baixa com os comprovantes que sobraram."""
    for cid in comprovantes_livres().id:
        baixado, _ = conciliar(int(cid))
        if baixado and baixado.boleto_id == boleto_id:
            return int(cid)
    return None
