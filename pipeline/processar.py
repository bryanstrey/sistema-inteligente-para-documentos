"""Pipeline de ponta a ponta: Entrada -> OCR -> ML -> LLM -> Validação -> Saída.

Regra de decisão (o que o ML descobre define o que o LLM procura):
  - confiança do classificador abaixo do limiar  -> NÃO extrai; vai para revisão humana escolher o tipo
  - extração com erro de validação, OCR ruim ou LLM discordando do tipo -> salva como 'revisao'
  - tudo certo -> 'salvo'
Comprovante salvo tenta dar baixa no boleto que pagou; boleto salvo procura um comprovante que já tinha chegado.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import config
from pipeline import banco
from pipeline.classificador import Classificacao, classificar
from pipeline.conciliacao import Candidato, conciliar, conciliar_boleto_novo
from pipeline.extrator_llm import ResultadoExtracao, extrair
from pipeline.mascaramento import Mascarador
from pipeline.ocr import ResultadoOCR, extrair_texto
from pipeline.validacao import tem_erro, validar


@dataclass
class Resultado:
    id: int | None
    ocr: ResultadoOCR
    classificacao: Classificacao
    extracao: ResultadoExtracao | None
    validacoes: list[dict]
    status: str
    motivos: list[str] = field(default_factory=list)
    texto_mascarado: str = ""
    itens_mascarados: dict = field(default_factory=dict)
    duplicado_de: int | None = None
    boleto_baixado: Candidato | None = None        # comprovante: boleto que ele quitou
    sugestoes_boleto: list[Candidato] = field(default_factory=list)  # comprovante: boletos que talvez tenha pago
    pago_por: int | None = None                    # boleto: comprovante que já tinha chegado e o quitou


def extrair_e_validar(tipo: str, texto: str, mascarador: Mascarador | None = None):
    mascarador = mascarador or Mascarador()
    texto_envio = mascarador.mascarar(texto) if config.MASCARAR_DADOS else texto
    ext = extrair(tipo, texto_envio)
    ext.campos = mascarador.restaurar(ext.campos)  # marcadores voltam a ser os dados reais, localmente
    val = validar(tipo, ext.campos, texto)
    return ext, val, texto_envio, mascarador


def processar(conteudo: bytes, nome_arquivo: str, salvar: bool = True) -> Resultado:
    h = hashlib.sha256(conteudo).hexdigest()
    duplicado = banco.existe_hash(h) if salvar else None

    ocr = extrair_texto(conteudo, nome_arquivo)
    cls = classificar(ocr.texto)
    motivos: list[str] = []
    if ocr.qualidade_baixa:
        motivos.append(f"Qualidade do OCR baixa ({ocr.qualidade:.0f}/100). Confira se a foto está nítida e bem iluminada.")
    pouco_texto = len(ocr.texto.strip()) < 60
    if pouco_texto:
        motivos.append("Quase nenhum texto foi lido do documento. Tire outra foto ou envie o PDF.")

    ext, val, texto_envio, masc = None, [], "", Mascarador()
    if cls.precisa_revisao:
        motivos.insert(0, f"Classificador sem certeza ({cls.confianca:.0%}). Escolha o tipo correto para continuar.")
    elif not pouco_texto:
        ext, val, texto_envio, masc = extrair_e_validar(cls.tipo, ocr.texto, masc)
        if tem_erro(val):
            motivos.append("Algum campo extraído não passou na validação.")
        if not ext.tipo_confere:
            motivos.append("O LLM achou que o documento não é do tipo previsto pelo classificador.")
        if ext.erro:
            motivos.append(ext.erro)

    status = "revisao" if motivos else "salvo"
    res = Resultado(None, ocr, cls, ext, val, status, motivos, texto_envio, masc.resumo, duplicado)

    if salvar:
        res.id = banco.inserir({
            "arquivo": nome_arquivo, "hash": h, "metodo_ocr": ocr.metodo, "qualidade_ocr": ocr.qualidade,
            "tipo_previsto": cls.tipo, "confianca": cls.confianca, "probabilidades": cls.probabilidades,
            "tipo_final": cls.tipo, "status": status, "motivo_revisao": " | ".join(motivos),
            "campos": ext.campos if ext else {}, "resumo": ext.resumo if ext else "", "validacoes": val,
            "motor_llm": ext.motor if ext else "", "texto_ocr": ocr.texto, "texto_mascarado": texto_envio or masc.mascarar(ocr.texto),
            "itens_mascarados": masc.resumo,
        })
        if status == "salvo" and not duplicado:
            depois_de_salvar(res)
    return res


def depois_de_salvar(res: Resultado):
    if res.classificacao.tipo == "comprovante":
        res.boleto_baixado, res.sugestoes_boleto = conciliar(res.id)
    elif res.classificacao.tipo == "boleto":
        res.pago_por = conciliar_boleto_novo(res.id)


def reextrair(doc_id: int, tipo: str) -> tuple[ResultadoExtracao, list[dict]]:
    """Usado na revisão humana: a pessoa escolhe o tipo e o LLM extrai com esse tipo."""
    doc = banco.obter(doc_id)
    ext, val, texto_envio, masc = extrair_e_validar(tipo, doc["texto_ocr"])
    banco.atualizar_extracao(doc_id, tipo, ext.campos, ext.resumo, val, ext.motor, texto_envio, masc.resumo)
    return ext, val
