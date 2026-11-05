"""Etapa 4: o LLM recebe o texto (mascarado) + o tipo que o ML decidiu e devolve um JSON
com os campos daquele tipo e um resumo curto.

Provedores:
  - anthropic: API comercial (precisa de ANTHROPIC_API_KEY)
  - ollama:    modelo aberto rodando local (nenhum dado sai da máquina)
  - sem_llm:   extrator por regex, só para testar o sistema sem chave (e como linha de base)
"""
from __future__ import annotations

import json
import os
import re
import urllib.request
from dataclasses import dataclass, field

import config
from pipeline.validacao import parse_valor

# ----------------------------------------------------------------------------- esquemas por tipo

ESQUEMAS: dict[str, dict[str, str]] = {
    "nota_fiscal": {
        "emitente_nome": "razão social de quem emitiu a nota (texto)",
        "emitente_cnpj": "CNPJ do emitente no formato 00.000.000/0000-00",
        "numero": "número da nota, só dígitos, sem zeros à esquerda",
        "data_emissao": "data de emissão no formato DD/MM/AAAA",
        "valor_total": "valor total a pagar, número com ponto decimal (ex.: 187.40)",
        "forma_pagamento": "forma de pagamento (ex.: Cartão de Crédito, PIX, Dinheiro)",
        "chave_acesso": "chave de acesso de 44 dígitos, em grupos de 4 separados por espaço",
        "categoria_gasto": "uma de: alimentação, saúde, transporte, casa, tecnologia, serviços, outros (pela natureza dos itens)",
    },
    "boleto": {
        "beneficiario_nome": "quem recebe o pagamento (beneficiário/cedente)",
        "beneficiario_cnpj": "CNPJ do beneficiário no formato 00.000.000/0000-00",
        "valor": "valor do documento, número com ponto decimal",
        "vencimento": "data de vencimento DD/MM/AAAA",
        "linha_digitavel": "linha digitável completa (47 dígitos) no formato 00000.00000 00000.000000 00000.000000 0 00000000000000",
        "banco": "nome do banco emissor",
        "numero_documento": "número do documento",
        "pagador": "nome do pagador/sacado (pode ser um marcador como [PESSOA_1])",
    },
    "comprovante": {
        "tipo_operacao": "uma de: pix, ted, pagamento_boleto, cartao, outro",
        "recebedor_nome": "quem recebeu o dinheiro (pode ser um marcador como [PESSOA_1])",
        "pagador_nome": "quem pagou (pode ser um marcador)",
        "valor": "valor da operação, número com ponto decimal",
        "data": "data da operação DD/MM/AAAA",
        "identificador": "ID/autenticação da transação",
    },
    "contrato": {
        "tipo_contrato": "ex.: prestação de serviços, locação",
        "objeto": "objeto do contrato em uma frase curta, como está no texto",
        "contratante": "nome do contratante/locatário (pode ser um marcador)",
        "contratada_nome": "nome da contratada/locador",
        "data_inicio": "início da vigência DD/MM/AAAA",
        "data_termino": "término da vigência DD/MM/AAAA",
        "valor_mensal": "valor mensal, número com ponto decimal",
        "multa_rescisao": "regra de multa por rescisão, resumida",
    },
    "outros": {
        "tipo_provavel": "o que esse documento parece ser (ex.: orçamento, recibo, declaração, e-mail)",
        "data_relevante": "a data mais importante do documento DD/MM/AAAA",
        "valor_relevante": "o valor mais importante, número com ponto decimal",
    },
}

INSTRUCOES = """Você é um extrator de dados de documentos brasileiros. O texto veio de OCR e pode ter erros \
(0 no lugar de O, 1 no lugar de l, palavras grudadas, linhas faltando).

Regras:
1. Extraia SOMENTE o que está escrito no documento. Se um campo não aparece, use null. Nunca invente nem calcule \
um valor que não está no texto.
2. Pode corrigir erros óbvios de OCR (ex.: "Dinhelro" -> "Dinheiro", "R$ 1.2O0,00" -> 1200.00), mas não troque dígitos \
de CNPJ, chave de acesso ou linha digitável por palpite.
3. Dados pessoais foram trocados por marcadores como [PESSOA_1], [CPF_1]. Copie o marcador exatamente quando ele for o valor do campo.
4. Valores em número com ponto decimal (1234.56). Datas em DD/MM/AAAA.
5. "resumo": uma ou duas frases em português dizendo o que é o documento, sem repetir CPF ou dados pessoais.
6. "tipo_confere": false se o texto claramente NÃO é do tipo informado (o classificador pode ter errado).
7. Responda APENAS com o JSON, sem texto antes ou depois e sem ```.

Formato da resposta:
{"campos": {<campos do esquema>}, "resumo": "...", "tipo_confere": true, "observacoes": "algo estranho no documento ou null"}"""

EXEMPLO_ENTRADA = """Tipo informado pelo classificador: boleto
Campos esperados:
- beneficiario_nome: quem recebe o pagamento
- valor: valor do documento
- vencimento: data de vencimento DD/MM/AAAA
- numero_documento: número do documento

Texto do documento:
Benefíciario: Escola Arco-Íris Ltda
Pagável em qualquer banco
VENClMENTO 1O/03/2026
(=) Valor d0 documento 850,O0
Pagador: [PESSOA_1]"""

EXEMPLO_SAIDA = ('{"campos": {"beneficiario_nome": "Escola Arco-Íris Ltda", "valor": 850.00, "vencimento": "10/03/2026", '
                 '"numero_documento": null}, "resumo": "Boleto da Escola Arco-Íris Ltda de R$ 850,00 com vencimento em 10/03/2026.", '
                 '"tipo_confere": true, "observacoes": "número do documento não aparece no texto"}')


def montar_prompt(tipo: str, texto: str) -> str:
    esquema = ESQUEMAS.get(tipo, ESQUEMAS["outros"])
    campos = "\n".join(f"- {k}: {v}" for k, v in esquema.items())
    return f"Tipo informado pelo classificador: {tipo}\nCampos esperados:\n{campos}\n\nTexto do documento:\n{texto[:12000]}"


# ----------------------------------------------------------------------------- chamada ao LLM


@dataclass
class ResultadoExtracao:
    campos: dict
    resumo: str
    tipo_confere: bool = True
    observacoes: str | None = None
    motor: str = ""
    erro: str | None = None
    bruto: str = field(default="", repr=False)


def _json_da_resposta(texto: str) -> dict:
    texto = re.sub(r"```(?:json)?", "", texto).strip()
    ini, fim = texto.find("{"), texto.rfind("}")
    return json.loads(texto[ini:fim + 1])


def _chamar_anthropic(prompt: str) -> str:
    import anthropic
    cliente = anthropic.Anthropic()  # lê ANTHROPIC_API_KEY do ambiente
    resp = cliente.messages.create(
        model=config.MODELO_ANTHROPIC,
        max_tokens=1200,
        system=INSTRUCOES,
        messages=[
            {"role": "user", "content": EXEMPLO_ENTRADA},
            {"role": "assistant", "content": EXEMPLO_SAIDA},
            {"role": "user", "content": prompt},
        ],
    )
    return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")


def _chamar_ollama(prompt: str) -> str:
    corpo = {
        "model": config.MODELO_OLLAMA,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0},
        "messages": [
            {"role": "system", "content": INSTRUCOES},
            {"role": "user", "content": EXEMPLO_ENTRADA},
            {"role": "assistant", "content": EXEMPLO_SAIDA},
            {"role": "user", "content": prompt},
        ],
    }
    req = urllib.request.Request(f"{config.URL_OLLAMA}/api/chat", data=json.dumps(corpo).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read())["message"]["content"]


def provedor_ativo() -> str:
    if config.PROVEDOR_LLM == "anthropic" and not os.getenv("ANTHROPIC_API_KEY"):
        return "sem_llm"
    return config.PROVEDOR_LLM


def extrair(tipo: str, texto_mascarado: str) -> ResultadoExtracao:
    prov = provedor_ativo()
    if prov == "sem_llm":
        r = extrair_regex(tipo, texto_mascarado)
        r.observacoes = "Modo sem LLM (regex). Configure ANTHROPIC_API_KEY ou o Ollama para a extração inteligente."
        return r
    prompt = montar_prompt(tipo, texto_mascarado)
    motor = f"{prov}:{config.MODELO_ANTHROPIC if prov == 'anthropic' else config.MODELO_OLLAMA}"
    bruto = ""
    try:
        bruto = _chamar_anthropic(prompt) if prov == "anthropic" else _chamar_ollama(prompt)
        dados = _json_da_resposta(bruto)
    except Exception as e:  # rede fora, JSON quebrado etc.: cai para regex e avisa
        r = extrair_regex(tipo, texto_mascarado)
        r.erro = f"Falha no LLM ({type(e).__name__}: {e}). Usado extrator por regex."
        r.bruto = bruto
        return r
    esquema = ESQUEMAS.get(tipo, ESQUEMAS["outros"])
    campos = {k: dados.get("campos", {}).get(k) for k in esquema}  # ignora campo que não estava no esquema
    for k in campos:
        if k.startswith("valor") and campos[k] is not None:
            campos[k] = parse_valor(campos[k])
    return ResultadoExtracao(campos, dados.get("resumo") or "", bool(dados.get("tipo_confere", True)),
                             dados.get("observacoes"), motor, None, bruto)


# ----------------------------------------------------------------------------- linha de base: regex


def _busca(rx: str, texto: str, grupo: int = 1):
    m = re.search(rx, texto, re.IGNORECASE)
    return m.group(grupo).strip() if m else None


RX_DATA = r"(\d{2}/\d{2}/\d{4})"
RX_VALOR = r"(\d{1,3}(?:\.\d{3})*,\d{2})"
RX_CNPJ = r"(\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2})"


def extrair_regex(tipo: str, texto: str) -> ResultadoExtracao:
    """O que dá pra fazer só com regex. Serve de comparação: quebra quando o layout muda ou o OCR erra."""
    c: dict = {k: None for k in ESQUEMAS.get(tipo, ESQUEMAS["outros"])}
    if tipo == "nota_fiscal":
        c["emitente_cnpj"] = _busca(r"CNPJ:?\s*" + RX_CNPJ, texto)
        c["data_emissao"] = _busca(r"emiss[ãa]o:?\s*" + RX_DATA, texto)
        c["valor_total"] = parse_valor(_busca(r"valor total[^\d]*" + RX_VALOR, texto))
        c["numero"] = _busca(r"N[ºo°]\s*(\d+)", texto)
        c["chave_acesso"] = _busca(r"((?:\d{4}\s){10}\d{4})", texto)
        linhas = [l for l in texto.split("\n") if l.strip()]
        c["emitente_nome"] = linhas[1].strip() if len(linhas) > 1 else None
    elif tipo == "boleto":
        c["beneficiario_cnpj"] = _busca(r"Benefici[áa]rio[^\n]*\n?[^\n]*?" + RX_CNPJ, texto)
        c["vencimento"] = _busca(r"vencimento:?\s*" + RX_DATA, texto)
        c["valor"] = parse_valor(_busca(r"valor do documento[^\d]*" + RX_VALOR, texto))
        c["linha_digitavel"] = _busca(r"(\d{5}\.\d{5}\s\d{5}\.\d{6}\s\d{5}\.\d{6}\s\d\s\d{14})", texto)
        c["beneficiario_nome"] = _busca(r"Benefici[áa]rio:\s*([^\n]+?)\s+CNPJ", texto)
        c["numero_documento"] = _busca(r"N[ºo°] do documento\s*(\d+)", texto)
    elif tipo == "comprovante":
        c["valor"] = parse_valor(_busca(r"R\$\s*" + RX_VALOR, texto))
        c["data"] = _busca(RX_DATA, texto)
    elif tipo == "contrato":
        c["data_inicio"] = _busca(r"in[íi]cio em\s*" + RX_DATA, texto)
        c["data_termino"] = _busca(r"t[ée]rmino em\s*" + RX_DATA, texto)
        c["valor_mensal"] = parse_valor(_busca(r"valor mensal de R\$\s*" + RX_VALOR, texto))
        c["objeto"] = _busca(r"tem por objeto (?:a |o )?([^.]+)", texto)
    else:
        c["data_relevante"] = _busca(RX_DATA, texto)
        c["valor_relevante"] = parse_valor(_busca(r"R\$\s*" + RX_VALOR, texto))
    return ResultadoExtracao(c, "", True, None, "regex")
