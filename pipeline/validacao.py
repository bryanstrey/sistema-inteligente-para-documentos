"""Validações determinísticas sobre os campos que o LLM extraiu.

A ideia é não confiar cegamente no LLM: tudo que dá pra checar com regra, a gente checa.
- dígitos verificadores (CNPJ, CPF, linha digitável do boleto, chave de acesso da NF-e)
- consistência cruzada (valor/vencimento do boleto batem com a linha digitável? CNPJ bate com a chave?)
- evidência: o valor extraído aparece mesmo no texto do documento? (detecta alucinação)
"""
from __future__ import annotations

import re
import unicodedata
from datetime import date, datetime, timedelta

# ----------------------------------------------------------------------------- utilitários


def so_digitos(s) -> str:
    return re.sub(r"\D", "", str(s or ""))


def parse_data(s) -> date | None:
    if not s:
        return None
    s = str(s).strip()
    for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y", "%d.%m.%Y", "%d/%m/%y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            pass
    return None


def parse_valor(v) -> float | None:
    """Aceita 187.4, '187,40', 'R$ 1.234,56', '1234.56'."""
    if v is None or v == "":
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = re.sub(r"[^\d,.\-]", "", str(v))
    if not s:
        return None
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".") if s.rfind(",") > s.rfind(".") else s.replace(",", "")
    elif "," in s:
        s = s.replace(".", "").replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def formatar_valor(v: float | None) -> str:
    if v is None:
        return "—"
    return "R$ " + f"{v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def normalizar(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"\s+", " ", s).strip()


# ----------------------------------------------------------------------------- CPF / CNPJ


def cnpj_valido(cnpj) -> bool:
    d = so_digitos(cnpj)
    if len(d) != 14 or d == d[0] * 14:
        return False

    def dv(base, pesos):
        r = sum(int(a) * b for a, b in zip(base, pesos)) % 11
        return "0" if r < 2 else str(11 - r)

    p1 = [5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2]
    p2 = [6] + p1
    return d[12] == dv(d[:12], p1) and d[13] == dv(d[:13], p2)


def cpf_valido(cpf) -> bool:
    d = so_digitos(cpf)
    if len(d) != 11 or d == d[0] * 11:
        return False
    for i in (9, 10):
        s = sum(int(d[j]) * (i + 1 - j) for j in range(i))
        if int(d[i]) != (s * 10 % 11) % 10:
            return False
    return True


# ----------------------------------------------------------------------------- boleto

DATA_BASE_FATOR_ANTIGA = date(1997, 10, 7)
DATA_RESET_FATOR = date(2025, 2, 22)  # a partir daqui o fator recomeça em 1000


def _mod10(num: str) -> int:
    soma, peso = 0, 2
    for c in reversed(num):
        p = int(c) * peso
        soma += p // 10 + p % 10
        peso = 1 if peso == 2 else 2
    return (10 - soma % 10) % 10


def _mod11_boleto(num: str) -> int:
    soma, peso = 0, 2
    for c in reversed(num):
        soma += int(c) * peso
        peso = 2 if peso == 9 else peso + 1
    dv = 11 - soma % 11
    return 1 if dv in (0, 10, 11) else dv


def fator_vencimento(venc: date) -> int:
    if venc >= DATA_RESET_FATOR:
        return 1000 + (venc - DATA_RESET_FATOR).days
    return (venc - DATA_BASE_FATOR_ANTIGA).days


def data_do_fator(fator: int, referencia: date | None = None) -> date:
    """O fator é cíclico; escolhe a data mais próxima da referência (hoje)."""
    referencia = referencia or date.today()
    candidatas = [DATA_BASE_FATOR_ANTIGA + timedelta(days=fator), DATA_RESET_FATOR + timedelta(days=fator - 1000)]
    return min(candidatas, key=lambda d: abs((d - referencia).days))


def gerar_linha_digitavel(banco: str, valor: float, venc: date, campo_livre: str) -> str:
    """Gera uma linha digitável válida (usada para criar boletos fictícios)."""
    fator = f"{fator_vencimento(venc):04d}"
    valor_s = f"{round(valor * 100):010d}"
    sem_dv = banco + "9" + fator + valor_s + campo_livre
    dv = _mod11_boleto(sem_dv)
    c1 = banco + "9" + campo_livre[:5]
    c2 = campo_livre[5:15]
    c3 = campo_livre[15:25]
    c1 += str(_mod10(c1))
    c2 += str(_mod10(c2))
    c3 += str(_mod10(c3))
    return f"{c1[:5]}.{c1[5:]} {c2[:5]}.{c2[5:]} {c3[:5]}.{c3[5:]} {dv} {fator}{valor_s}"


def analisar_linha_digitavel(linha) -> dict:
    """Valida a linha digitável e devolve valor e vencimento codificados nela."""
    d = so_digitos(linha)
    if len(d) != 47:
        return {"valida": False, "motivo": f"linha com {len(d)} dígitos (esperado 47)"}
    c1, c2, c3, dv, c5 = d[:10], d[10:21], d[21:32], d[32], d[33:]
    for nome, campo in (("campo 1", c1), ("campo 2", c2), ("campo 3", c3)):
        if int(campo[-1]) != _mod10(campo[:-1]):
            return {"valida": False, "motivo": f"dígito verificador do {nome} não confere"}
    campo_livre = c1[4:9] + c2[:10] + c3[:10]
    barras_sem_dv = c1[:4] + c5 + campo_livre
    if int(dv) != _mod11_boleto(barras_sem_dv):
        return {"valida": False, "motivo": "dígito verificador geral não confere"}
    fator = int(c5[:4])
    valor = int(c5[4:]) / 100
    return {
        "valida": True,
        "banco": c1[:3],
        "valor": valor if valor > 0 else None,
        "vencimento": data_do_fator(fator) if fator > 0 else None,
    }


# ----------------------------------------------------------------------------- NF-e


def _mod11_nfe(num: str) -> int:
    soma, peso = 0, 2
    for c in reversed(num):
        soma += int(c) * peso
        peso = 2 if peso == 9 else peso + 1
    r = soma % 11
    return 0 if r < 2 else 11 - r


def gerar_chave_nfe(uf: str, data_emissao: date, cnpj: str, modelo: str, serie: int, numero: int, codigo: int) -> str:
    base = f"{uf}{data_emissao:%y%m}{so_digitos(cnpj)}{modelo}{serie:03d}{numero:09d}1{codigo:08d}"
    chave = base + str(_mod11_nfe(base))
    return " ".join(chave[i:i + 4] for i in range(0, 44, 4))


def analisar_chave_nfe(chave) -> dict:
    d = so_digitos(chave)
    if len(d) != 44:
        return {"valida": False, "motivo": f"chave com {len(d)} dígitos (esperado 44)"}
    if int(d[-1]) != _mod11_nfe(d[:43]):
        return {"valida": False, "motivo": "dígito verificador da chave não confere"}
    return {"valida": True, "cnpj": d[6:20], "ano_mes": f"{d[4:6]}/{d[2:4]}", "numero": int(d[25:34])}


# ----------------------------------------------------------------------------- evidência


def aparece_no_texto(valor, texto: str) -> bool:
    """Verifica se um valor extraído tem respaldo no texto do documento."""
    if valor is None or valor == "":
        return True
    t = normalizar(texto)
    tdig = so_digitos(texto)
    if isinstance(valor, (int, float)):
        reais = f"{valor:,.2f}"
        variantes = {reais.replace(",", "X").replace(".", ",").replace("X", "."), f"{valor:.2f}".replace(".", ",")}
        return any(v in texto for v in variantes) or so_digitos(f"{valor:.2f}") in tdig
    s = str(valor)
    dt = parse_data(s)
    if dt:
        return any(x in texto for x in (dt.strftime("%d/%m/%Y"), dt.strftime("%d/%m/%y"), dt.strftime("%d.%m.%Y"), dt.strftime("%d-%m-%Y")))
    dig = so_digitos(s)
    if len(dig) >= 8 and len(dig) == len(re.sub(r"[\s./\-]", "", s)):
        return dig in tdig
    if s.startswith("[") and s.endswith("]"):  # placeholder de mascaramento
        return True
    palavras = [p for p in normalizar(s).split() if len(p) > 2]
    if not palavras:
        return True
    achadas = sum(1 for p in palavras if p in t)
    return achadas / len(palavras) >= 0.6


# ----------------------------------------------------------------------------- validação por tipo

CAMPOS_EVIDENCIA = {
    "nota_fiscal": ["emitente_nome", "emitente_cnpj", "data_emissao", "valor_total", "numero"],
    "boleto": ["beneficiario_nome", "beneficiario_cnpj", "valor", "vencimento"],
    "comprovante": ["recebedor_nome", "valor", "data"],
    "contrato": ["objeto", "data_inicio", "data_termino", "valor_mensal"],
}


def validar(tipo: str, campos: dict, texto: str) -> list[dict]:
    """Devolve uma lista de verificações: {campo, ok, nivel ('erro'|'aviso'|'ok'), mensagem}."""
    res: list[dict] = []

    def add(campo, ok, msg, nivel=None):
        res.append({"campo": campo, "ok": ok, "nivel": nivel or ("ok" if ok else "erro"), "mensagem": msg})

    campos = campos or {}

    for c in CAMPOS_EVIDENCIA.get(tipo, []):
        v = campos.get(c)
        if v not in (None, "") and not aparece_no_texto(v, texto):
            add(c, False, f"'{v}' não foi encontrado no texto do documento (possível invenção do LLM)")

    for c, v in campos.items():
        if v in (None, ""):
            continue
        if c.endswith("cnpj"):
            add(c, cnpj_valido(v), "CNPJ com dígito verificador válido" if cnpj_valido(v) else f"CNPJ {v} inválido")
        if c.startswith("data") or c in ("vencimento",):
            if parse_data(v) is None:
                add(c, False, f"data '{v}' em formato não reconhecido")
        if c.startswith("valor"):
            pv = parse_valor(v)
            if pv is None or pv <= 0:
                add(c, False, f"valor '{v}' inválido")

    if tipo == "boleto" and campos.get("linha_digitavel"):
        info = analisar_linha_digitavel(campos["linha_digitavel"])
        if not info["valida"]:
            add("linha_digitavel", False, "Linha digitável inválida: " + info["motivo"])
        else:
            add("linha_digitavel", True, "Linha digitável válida (dígitos verificadores conferem)")
            v = parse_valor(campos.get("valor"))
            if info["valor"] and v is not None and abs(info["valor"] - v) > 0.01:
                add("valor", False, f"valor extraído ({formatar_valor(v)}) difere do codificado no boleto ({formatar_valor(info['valor'])})")
            venc = parse_data(campos.get("vencimento"))
            if info["vencimento"] and venc and venc != info["vencimento"]:
                add("vencimento", False, f"vencimento extraído ({venc:%d/%m/%Y}) difere do codificado no boleto ({info['vencimento']:%d/%m/%Y})")

    if tipo == "nota_fiscal" and campos.get("chave_acesso"):
        info = analisar_chave_nfe(campos["chave_acesso"])
        if not info["valida"]:
            add("chave_acesso", False, "Chave de acesso inválida: " + info["motivo"])
        else:
            add("chave_acesso", True, "Chave de acesso válida")
            cnpj = so_digitos(campos.get("emitente_cnpj"))
            if cnpj and cnpj != info["cnpj"]:
                add("emitente_cnpj", False, "CNPJ do emitente não bate com o CNPJ da chave de acesso")

    if tipo == "contrato":
        ini, fim = parse_data(campos.get("data_inicio")), parse_data(campos.get("data_termino"))
        if ini and fim and fim < ini:
            add("data_termino", False, "data de término anterior à data de início")

    return res


def tem_erro(verificacoes: list[dict]) -> bool:
    return any(v["nivel"] == "erro" for v in verificacoes)
