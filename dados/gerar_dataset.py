"""Gera a base própria de documentos fictícios em pt-BR.

Por que fictícios: documentos reais têm CPF, endereço e dados financeiros (LGPD). Aqui tudo é
gerado com Faker; CNPJ, linha digitável e chave da NF-e saem com dígitos verificadores válidos.

Para o classificador não ficar fácil demais (e o resultado não ser 100% "de mentira"), a gerada:
- varia muito o layout e o vocabulário de cada tipo;
- inclui na classe "outros" documentos parecidos com as outras classes (orçamento, recibo simples,
  aviso de cobrança, pedido de compra), que são os erros mais prováveis no mundo real;
- simula erros de OCR (troca de letras, acentos perdidos, linhas cortadas).

Uso:  python dados/gerar_dataset.py --por-classe 400
Saída: dados/dataset.csv (texto, rotulo, ruido) e dados/gabarito_extracao.jsonl (campos esperados)
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from faker import Faker

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.validacao import formatar_valor, gerar_chave_nfe, gerar_linha_digitavel  # noqa: E402

fk = Faker("pt_BR")
R = random.Random(42)
Faker.seed(42)

BANCOS = {"001": "Banco do Brasil", "237": "Bradesco", "341": "Itaú", "104": "Caixa Econômica Federal",
          "033": "Santander", "756": "Sicoob", "748": "Sicredi", "077": "Banco Inter", "260": "Nu Pagamentos"}
UFS = {"41": "PR", "35": "SP", "42": "SC", "43": "RS", "33": "RJ", "31": "MG"}
PRODUTOS = [
    ("Arroz tipo 1 5kg", 24.9), ("Feijão carioca 1kg", 8.49), ("Café torrado 500g", 18.9), ("Leite integral 1L", 5.29),
    ("Óleo de soja 900ml", 7.99), ("Detergente 500ml", 2.79), ("Papel higiênico 12un", 21.9), ("Açúcar cristal 2kg", 9.98),
    ("Pão de forma", 9.49), ("Banana prata kg", 6.99), ("Frango congelado kg", 13.9), ("Sabão em pó 1kg", 16.5),
    ("Cabo HDMI 2m", 39.9), ("Mouse sem fio", 79.0), ("Teclado USB", 89.9), ("Cartucho de tinta", 119.0),
    ("Parafuso 6mm cx", 14.0), ("Tinta acrílica 18L", 389.0), ("Lâmpada LED 9W", 12.9), ("Fita isolante", 6.5),
    ("Dipirona 500mg", 7.9), ("Protetor solar FPS50", 54.9), ("Shampoo 400ml", 22.9), ("Escova dental", 9.9),
    ("Gasolina comum L", 6.29), ("Óleo motor 1L", 42.0), ("Serviço de manutenção", 250.0), ("Consultoria técnica (h)", 180.0),
]
SERVICOS_CONTRATO = [
    "prestação de serviços de desenvolvimento de software", "locação do imóvel residencial situado à {end}",
    "prestação de serviços de limpeza e conservação", "manutenção preventiva de equipamentos de informática",
    "prestação de serviços de contabilidade", "fornecimento de internet banda larga", "locação de equipamento de impressão",
    "prestação de serviços de consultoria em marketing digital", "prestação de serviços educacionais",
]


def dt(base: date | None = None, ini=-120, fim=60) -> date:
    base = base or date(2026, 10, 15)
    return base + timedelta(days=R.randint(ini, fim))


def valor(a=10, b=3000) -> float:
    return round(R.uniform(a, b), 2)


def br(v: float) -> str:
    return formatar_valor(v).replace("R$ ", "")


def empresa() -> str:
    sufixo = R.choice(["Ltda", "LTDA", "S.A.", "ME", "EIRELI", "Comércio Ltda", "Serviços Ltda"])
    nome = R.choice([fk.company(), f"{fk.last_name()} {R.choice(['Supermercados', 'Distribuidora', 'Materiais', 'Tecnologia', 'Farmácia', 'Auto Posto', 'Comercial'])}"])
    return f"{nome.split(' - ')[0]} {sufixo}"


def endereco() -> str:
    return f"{fk.street_address()}, {fk.bairro()}, {fk.city()} - {R.choice(list(UFS.values()))}, CEP {fk.postcode()}"


def j(*linhas) -> str:
    return "\n".join(l for l in linhas if l is not None)


def talvez(p, s):
    return s if R.random() < p else None


# ============================================================================ NOTA FISCAL

def nota_fiscal() -> tuple[str, dict]:
    emit, cnpj, data_e = empresa(), fk.cnpj(), dt()
    uf = R.choice(list(UFS))
    modelo = R.choice(["55", "65"])
    numero, serie = R.randint(1000, 999999), R.randint(1, 9)
    chave = gerar_chave_nfe(uf, data_e, cnpj, modelo, serie, numero, R.randint(1, 99999999))
    itens = R.sample(PRODUTOS, R.randint(1, 7))
    linhas_itens, total = [], 0.0
    for i, (desc, preco) in enumerate(itens, 1):
        q = R.choice([1, 1, 1, 2, 3, 0.5, 1.25]) if "kg" in desc or " L" in desc else R.randint(1, 4)
        sub = round(q * preco, 2)
        total += sub
        linhas_itens.append(R.choice([
            f"{i:03d} {desc.upper()} {q} UN x {br(preco)} {br(sub)}",
            f"{desc} | Qtd {q} | Vl Unit {br(preco)} | Vl Total {br(sub)}",
            f"{i} {desc} {q} {br(preco)} {br(sub)}",
        ]))
    desconto = round(total * R.choice([0, 0, 0, 0.05, 0.1]), 2)
    total = round(total - desconto, 2)
    pag = R.choice(["Cartão de Crédito", "Cartão de Débito", "Dinheiro", "PIX", "Boleto"])
    dest = fk.name()
    cpf_dest = fk.cpf()
    titulo = R.choice([
        "DANFE - Documento Auxiliar da Nota Fiscal Eletrônica",
        "DANFE NFC-e - Documento Auxiliar da Nota Fiscal de Consumidor Eletrônica",
        "NOTA FISCAL ELETRÔNICA", "NF-e", "Cupom fiscal eletrônico - NFC-e",
    ])
    texto = j(
        titulo,
        emit.upper() if R.random() < .5 else emit,
        f"CNPJ: {cnpj}" + R.choice(["", f"  IE: {R.randint(10**8, 10**9)}"]),
        talvez(.7, endereco()),
        R.choice([f"NF-e Nº {numero:09d} Série {serie:03d}", f"Número: {numero} Série: {serie}", f"Nota nº {numero}"]),
        R.choice([f"Data de emissão: {data_e:%d/%m/%Y}", f"Emissão: {data_e:%d/%m/%Y} {R.randint(8, 21)}:{R.randint(10, 59)}", f"DATA DA EMISSÃO {data_e:%d/%m/%Y}"]),
        talvez(.6, R.choice([f"Destinatário: {dest} CPF: {cpf_dest}", f"CONSUMIDOR CPF: {cpf_dest}", "CONSUMIDOR NÃO IDENTIFICADO"])),
        R.choice(["Código Descrição Qtd Vl Unit Vl Total", "ITENS", "DADOS DOS PRODUTOS / SERVIÇOS", None]),
        *linhas_itens,
        talvez(.4, f"Qtd. total de itens {len(itens)}"),
        talvez(desconto > 0, f"Desconto R$ {br(desconto)}"),
        R.choice([f"Valor total R$ {br(total)}", f"VALOR TOTAL DA NOTA {br(total)}", f"TOTAL R$ {br(total)}", f"Valor a pagar R$ {br(total)}"]),
        f"Forma de pagamento: {pag}" if R.random() < .7 else pag,
        talvez(.5, f"Tributos aproximados R$ {br(total * .18)} (Fonte IBPT)"),
        R.choice(["Chave de acesso", "Consulte pela Chave de Acesso em www.fazenda.pr.gov.br/nfce", "CHAVE DE ACESSO"]),
        chave,
        talvez(.5, f"Protocolo de autorização: {R.randint(10**14, 10**15)} {data_e:%d/%m/%Y}"),
    )
    campos = {"emitente_nome": emit, "emitente_cnpj": cnpj, "numero": str(numero), "data_emissao": f"{data_e:%d/%m/%Y}",
              "valor_total": total, "forma_pagamento": pag, "chave_acesso": chave}
    return texto, campos


# ============================================================================ BOLETO

def boleto() -> tuple[str, dict]:
    cod = R.choice(list(BANCOS))
    benef, cnpj = empresa(), fk.cnpj()
    v, venc = valor(30, 5000), dt(ini=-20, fim=60)
    linha = gerar_linha_digitavel(cod, v, venc, "".join(R.choice("0123456789") for _ in range(25)))
    pagador, cpf = fk.name(), fk.cpf()
    doc = str(R.randint(10**5, 10**8))
    instr = R.sample([
        "Não receber após 30 dias do vencimento.", f"Após o vencimento cobrar multa de 2% e juros de 1% ao mês.",
        "Pagável em qualquer banco até o vencimento.", "Pagável preferencialmente na rede bancária ou casas lotéricas.",
        "Sr. Caixa, não aceitar pagamento em cheque.", "Em caso de dúvidas, entre em contato com o beneficiário.",
        "Desconto de 5% para pagamento até o vencimento.",
    ], R.randint(1, 3))
    blocos = [
        R.choice([f"{BANCOS[cod]} | {cod}-{R.randint(0, 9)} | {linha}", f"{cod}-{R.randint(0, 9)} {linha}", f"{BANCOS[cod]}\n{linha}"]),
        R.choice(["Local de pagamento", "LOCAL DE PAGAMENTO"]) + " " + R.choice(["Pagável em qualquer banco até o vencimento", "Em qualquer banco ou correspondente"]),
        R.choice([f"Vencimento {venc:%d/%m/%Y}", f"VENCIMENTO: {venc:%d/%m/%Y}", f"Data de vencimento {venc:%d/%m/%Y}"]),
        R.choice([f"Beneficiário: {benef} CNPJ {cnpj}", f"Beneficiário\n{benef} - CNPJ: {cnpj}", f"Cedente: {benef} ({cnpj})"]),
        f"Agência/Código do beneficiário {R.randint(1000, 9999)}/{R.randint(10000, 99999)}-{R.randint(0, 9)}",
        R.choice([f"Data do documento {dt(venc, -30, -5):%d/%m/%Y}", None]),
        f"Nº do documento {doc}",
        talvez(.7, f"Nosso número {R.randint(10**9, 10**11)}"),
        R.choice(["Espécie R$", "Espécie doc DM  Aceite N", None]),
        R.choice([f"(=) Valor do documento {br(v)}", f"Valor do Documento R$ {br(v)}", f"VALOR: R$ {br(v)}"]),
        "Instruções (texto de responsabilidade do beneficiário)" if R.random() < .6 else "Instruções",
        *instr,
        R.choice([f"Pagador: {pagador} CPF: {cpf}", f"Sacado: {pagador} - {cpf}", f"Pagador\n{pagador}\nCPF {cpf}"]),
        talvez(.6, endereco()),
        R.choice(["Autenticação mecânica - Ficha de Compensação", "Ficha de Compensação", "Recibo do pagador", None]),
    ]
    if R.random() < .3:
        blocos.insert(0, "RECIBO DO PAGADOR")
    campos = {"beneficiario_nome": benef, "beneficiario_cnpj": cnpj, "valor": v, "vencimento": f"{venc:%d/%m/%Y}",
              "linha_digitavel": linha, "banco": BANCOS[cod], "numero_documento": doc}
    return j(*blocos), campos


# ============================================================================ COMPROVANTE

def comprovante() -> tuple[str, dict]:
    tipo = R.choice(["pix", "pix", "ted", "pagamento_boleto", "cartao"])
    banco = R.choice(list(BANCOS.values()))
    v, d = valor(5, 4000), dt()
    hora = f"{R.randint(0, 23):02d}:{R.randint(0, 59):02d}:{R.randint(0, 59):02d}"
    pagador, receb = fk.name(), R.choice([fk.name(), empresa()])
    ident = "E" + "".join(R.choice("0123456789ABCDEF") for _ in range(31))
    if tipo == "pix":
        texto = j(
            R.choice(["Comprovante de transferência", "Comprovante Pix", "Pix enviado", "Transferência realizada"]),
            banco, f"{d:%d/%m/%Y} às {hora}" if R.random() < .5 else f"Data: {d:%d/%m/%Y} {hora}",
            R.choice([f"Valor R$ {br(v)}", f"R$ {br(v)}", f"Valor da transferência: R$ {br(v)}"]),
            "Tipo de transferência Pix",
            R.choice(["Destino", "Quem recebeu", "Dados do recebedor"]), f"Nome {receb}",
            R.choice([f"CPF •••.{R.randint(100, 999)}.{R.randint(100, 999)}-••", f"CNPJ {fk.cnpj()}", f"Chave Pix {fk.email()}"]),
            f"Instituição {R.choice(list(BANCOS.values()))}",
            R.choice(["Origem", "Quem pagou", "Dados do pagador"]), f"Nome {pagador}", f"CPF {fk.cpf()}",
            f"Instituição {banco}", talvez(.5, f"Mensagem: {R.choice(['aluguel', 'pizza', 'mensalidade', 'conserto', 'presente'])}"),
            R.choice([f"ID da transação: {ident}", f"Autenticação {ident}", f"Código de autenticação\n{ident}"]),
        )
    elif tipo == "ted":
        texto = j(
            banco, R.choice(["Comprovante de TED", "Comprovante de transferência entre bancos - TED", "TRANSFERÊNCIA ELETRÔNICA DISPONÍVEL"]),
            f"Data da transferência: {d:%d/%m/%Y}", f"Valor: R$ {br(v)}",
            f"Favorecido: {receb}", f"Banco favorecido: {R.choice(list(BANCOS))} Agência {R.randint(1000, 9999)} Conta {R.randint(10000, 999999)}-{R.randint(0, 9)}",
            f"Remetente: {pagador}", f"Finalidade: {R.choice(['Crédito em conta', 'Pagamento de fornecedor', 'Pagamento de aluguel'])}",
            f"Número de controle {R.randint(10**9, 10**10)}", "Transação efetuada com sucesso",
        )
    elif tipo == "pagamento_boleto":
        cod = R.choice(list(BANCOS))
        linha = gerar_linha_digitavel(cod, v, d, "".join(R.choice("0123456789") for _ in range(25)))
        texto = j(
            banco, R.choice(["Comprovante de pagamento de boleto", "Comprovante de pagamento", "PAGAMENTO DE TÍTULO - COMPROVANTE"]),
            f"Código de barras {linha}", f"Beneficiário: {receb}", f"Data do pagamento {d:%d/%m/%Y}",
            f"Data de vencimento {d:%d/%m/%Y}", f"Valor do documento R$ {br(v)}", f"Valor pago R$ {br(v)}",
            f"Pagador: {pagador}", f"Autenticação: {ident[:20]}", "Pagamento realizado com sucesso",
        )
    else:
        texto = j(
            R.choice(["COMPROVANTE DE VENDA", "VIA CLIENTE", "Comprovante - via do cliente"]),
            R.choice(["CIELO", "STONE", "REDE", "GETNET", "PagSeguro"]), receb.upper(),
            f"{R.choice(['VISA', 'MASTERCARD', 'ELO'])} {R.choice(['CREDITO', 'DEBITO'])} ************{R.randint(1000, 9999)}",
            f"{d:%d/%m/%y} {hora}", f"VALOR APROVADO: R$ {br(v)}", R.choice(["A VISTA", f"PARCELADO {R.randint(2, 10)}X"]),
            f"AUT: {R.randint(100000, 999999)} DOC: {R.randint(100000, 999999)}", "TRANSACAO AUTORIZADA",
        )
    campos = {"recebedor_nome": receb, "valor": v, "data": f"{d:%d/%m/%Y}", "tipo": tipo}
    return texto, campos


# ============================================================================ CONTRATO

def contrato() -> tuple[str, dict]:
    obj = R.choice(SERVICOS_CONTRATO).format(end=endereco())
    locacao = "locação" in obj
    contratante, contratada = fk.name(), empresa()
    ini = dt(ini=-400, fim=30)
    meses = R.choice([6, 12, 12, 24, 36])
    fim = ini + timedelta(days=round(meses * 30.44))
    v = valor(300, 9000)
    papel = ("LOCATÁRIO", "LOCADOR") if locacao else ("CONTRATANTE", "CONTRATADA")
    clausulas = [
        f"CLÁUSULA PRIMEIRA - DO OBJETO. O presente contrato tem por objeto a {obj}.",
        f"CLÁUSULA SEGUNDA - DO PRAZO. O prazo de vigência é de {meses} ({fk.word()}) meses, com início em {ini:%d/%m/%Y} e término em {fim:%d/%m/%Y}, podendo ser renovado mediante termo aditivo.",
        f"CLÁUSULA TERCEIRA - DO VALOR. Pelo objeto deste contrato será pago o valor mensal de R$ {br(v)}, com vencimento todo dia {R.randint(1, 28)} de cada mês.",
        "CLÁUSULA QUARTA - DO REAJUSTE. O valor será reajustado anualmente pelo IPCA/IBGE.",
        "CLÁUSULA QUINTA - DA RESCISÃO. Qualquer das partes poderá rescindir o presente instrumento mediante aviso prévio de 30 (trinta) dias, sob pena de multa equivalente a três parcelas.",
        "CLÁUSULA SEXTA - DAS OBRIGAÇÕES. Compete às partes cumprir fielmente as disposições deste contrato.",
        "CLÁUSULA SÉTIMA - DA CONFIDENCIALIDADE. As partes se comprometem a manter sigilo sobre as informações trocadas.",
        f"CLÁUSULA OITAVA - DO FORO. Fica eleito o foro da comarca de {fk.city()} para dirimir quaisquer dúvidas oriundas deste contrato.",
    ]
    R.shuffle(clausulas[3:7])
    texto = j(
        R.choice([f"CONTRATO DE {'LOCAÇÃO' if locacao else 'PRESTAÇÃO DE SERVIÇOS'}", "INSTRUMENTO PARTICULAR DE CONTRATO",
                  f"Contrato de {'Locação Residencial' if locacao else 'Prestação de Serviços'} nº {R.randint(1, 999)}/{ini.year}"]),
        "Pelo presente instrumento particular, de um lado:",
        f"{contratante}, brasileiro(a), {R.choice(['solteiro(a)', 'casado(a)'])}, inscrito(a) no CPF sob o nº {fk.cpf()}, residente à {endereco()}, doravante denominado(a) {papel[0]};",
        f"e de outro lado {contratada}, pessoa jurídica de direito privado, inscrita no CNPJ sob o nº {fk.cnpj()}, com sede à {endereco()}, doravante denominada {papel[1]};",
        "têm entre si justo e contratado o seguinte, que mutuamente aceitam e outorgam:",
        *clausulas[:R.randint(5, 8)],
        "E por estarem assim justas e contratadas, firmam o presente em 2 (duas) vias de igual teor e forma, na presença de duas testemunhas.",
        f"{fk.city()}, {ini:%d/%m/%Y}.",
        f"_____________________ {papel[0]}", f"_____________________ {papel[1]}",
        talvez(.6, "Testemunhas: 1. ______________ 2. ______________"),
    )
    campos = {"objeto": obj, "data_inicio": f"{ini:%d/%m/%Y}", "data_termino": f"{fim:%d/%m/%Y}", "valor_mensal": v,
              "contratada_nome": contratada}
    return texto, campos


# ============================================================================ OUTROS (inclui casos difíceis)

def outros() -> tuple[str, dict]:
    tipo = R.choice(["orcamento", "recibo", "aviso_cobranca", "pedido", "declaracao", "ata", "email", "receita", "comunicado", "curriculo_ficticio"])
    d, v = dt(), valor(50, 5000)
    if tipo == "orcamento":  # parece nota fiscal, mas não é documento fiscal
        itens = R.sample(PRODUTOS, R.randint(2, 5))
        texto = j(R.choice(["ORÇAMENTO", "Orçamento nº " + str(R.randint(1, 999)), "PROPOSTA COMERCIAL"]), empresa(), f"CNPJ {fk.cnpj()}",
                  f"Cliente: {fk.name()}", f"Data: {d:%d/%m/%Y}", *[f"{x} {br(p)}" for x, p in itens],
                  f"Total estimado R$ {br(sum(p for _, p in itens))}", "Validade da proposta: 15 dias.",
                  "Este documento não possui valor fiscal.", "Formas de pagamento: PIX, boleto ou cartão.")
    elif tipo == "recibo":  # parece comprovante
        texto = j("RECIBO", f"Recebi de {fk.name()} a importância de R$ {br(v)} ({fk.word()} reais)",
                  f"referente a {R.choice(['serviço de pintura', 'aula particular', 'conserto de bicicleta', 'diária de faxina'])}.",
                  f"Para clareza firmo o presente.", f"{fk.city()}, {d:%d/%m/%Y}", "______________________", fk.name())
    elif tipo == "aviso_cobranca":  # parece boleto
        texto = j(R.choice(["AVISO DE COBRANÇA", "Lembrete de pagamento", "Notificação de débito em aberto"]), empresa(),
                  f"Prezado(a) {fk.name()},", f"Identificamos que a fatura com vencimento em {d:%d/%m/%Y} no valor de R$ {br(v)} ainda não foi paga.",
                  "Caso já tenha efetuado o pagamento, desconsidere esta mensagem.",
                  "Para emitir a segunda via do boleto acesse nosso site ou responda este e-mail.", "Atenciosamente, Setor Financeiro")
    elif tipo == "pedido":
        itens = R.sample(PRODUTOS, R.randint(2, 5))
        texto = j(f"PEDIDO DE COMPRA Nº {R.randint(100, 9999)}", f"Solicitante: {fk.name()}", f"Fornecedor: {empresa()}",
                  f"Data do pedido: {d:%d/%m/%Y}", *[f"- {x} (qtd {R.randint(1, 10)})" for x, _ in itens],
                  f"Previsão de entrega: {dt(d, 3, 20):%d/%m/%Y}", "Aprovado por: Gerência de Compras")
    elif tipo == "declaracao":
        texto = j(R.choice(["DECLARAÇÃO", "DECLARAÇÃO DE MATRÍCULA", "ATESTADO", "DECLARAÇÃO DE COMPARECIMENTO"]),
                  f"Declaramos, para os devidos fins, que {fk.name()}, portador(a) do CPF {fk.cpf()},",
                  R.choice(["encontra-se regularmente matriculado(a) no curso de Sistemas de Informação.",
                            f"esteve presente nesta instituição no dia {d:%d/%m/%Y} das 14h às 16h.",
                            "presta serviços nesta empresa desde 2021."]),
                  f"{fk.city()}, {d:%d/%m/%Y}.", "____________________", "Secretaria")
    elif tipo == "ata":
        texto = j(f"ATA DA REUNIÃO {R.choice(['ORDINÁRIA', 'DO CONDOMÍNIO', 'DA EQUIPE'])}", f"Aos {d:%d/%m/%Y}, reuniram-se os presentes para deliberar sobre:",
                  "1. aprovação das contas do período;", f"2. orçamento de R$ {br(v)} para reforma;", "3. assuntos gerais.",
                  "Após discussão, os itens foram aprovados por maioria.", "Nada mais havendo a tratar, encerrou-se a reunião.")
    elif tipo == "email":
        texto = j(f"De: {fk.email()}", f"Para: {fk.email()}", f"Assunto: {R.choice(['Reunião amanhã', 'Relatório mensal', 'Dúvida sobre o pedido', 'Férias'])}",
                  f"Olá {fk.first_name()},", fk.paragraph(nb_sentences=3), "Abraço,", fk.first_name())
    elif tipo == "receita":
        texto = j("RECEITA DE BOLO DE CENOURA", "Ingredientes:", "3 cenouras médias", "4 ovos", "1 xícara de óleo", "2 xícaras de açúcar",
                  "2 xícaras de farinha de trigo", "1 colher de fermento", "Modo de preparo:", "Bata no liquidificador as cenouras, ovos e óleo.",
                  "Misture com o açúcar e a farinha, por último o fermento. Asse por 40 minutos a 180 graus.")
    elif tipo == "comunicado":
        texto = j(R.choice(["COMUNICADO", "AVISO AOS MORADORES", "INFORMATIVO"]), fk.paragraph(nb_sentences=4),
                  f"A partir de {d:%d/%m/%Y} o horário de atendimento será das 8h às 17h.", "A administração.")
    else:
        texto = j(fk.name(), f"{fk.email()} | {fk.phone_number()}", "OBJETIVO", "Atuar na área de tecnologia da informação.",
                  "FORMAÇÃO", "Bacharelado em Sistemas de Informação", "EXPERIÊNCIA", f"{empresa()} - Estagiário ({d.year - 1} - atual)",
                  "HABILIDADES", "Python, SQL, Power BI, inglês intermediário")
    return texto, {"subtipo": tipo}


GERADORES = {"nota_fiscal": nota_fiscal, "boleto": boleto, "comprovante": comprovante, "contrato": contrato, "outros": outros}

# ============================================================================ ruído de OCR

TROCAS = {"o": "0", "O": "0", "l": "1", "I": "1", "e": "c", "a": "o", "S": "5", "B": "8", "rn": "m", "ç": "c", "ã": "a",
          "é": "e", "í": "i", "ó": "o", "ê": "e", "õ": "o", "m": "rn", "i": "l", ",": ".", "u": "v"}


LIXO = ["|", "~", "\\", "_", "'", ";", "°", "ª", "»", "1l", "rn", "..."]


def ruido_ocr(texto: str, intensidade: float, recorte: float | None = None) -> str:
    """Simula um OCR ruim / foto ruim.

    - recorte: mantém só um trecho contínuo do documento (foto cortada, documento dobrado);
    - troca de caracteres parecidos (o/0, l/1, rn/m), acentos perdidos, palavras grudadas;
    - linhas perdidas e lixo inserido (sujeira, sombra, carimbo).
    """
    linhas = texto.split("\n")
    if recorte is None:
        recorte = R.choice([1, 1, 1, R.uniform(0.3, 0.7)])
    if recorte < 1 and len(linhas) > 4:
        n = max(3, int(len(linhas) * recorte))
        ini = R.randint(0, len(linhas) - n)
        linhas = linhas[ini:ini + n]
    if intensidade <= 0:
        return "\n".join(linhas)
    linhas = [l for l in linhas if R.random() > intensidade * 0.25]
    out = []
    for l in linhas:
        chars = list(l)
        for i, c in enumerate(chars):
            if c in TROCAS and R.random() < intensidade * 0.15:
                chars[i] = TROCAS[c]
            elif c == " " and R.random() < intensidade * 0.05:
                chars[i] = ""
            elif R.random() < intensidade * 0.01:
                chars[i] = c + R.choice(LIXO)
        out.append("".join(chars))
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--por-classe", type=int, default=400)
    args = ap.parse_args()
    linhas, gabarito = [], []
    for rotulo, gen in GERADORES.items():
        for i in range(args.por_classe):
            texto, campos = gen()
            intens = R.choice([0, 0, 0.2, 0.4, 0.6])  # parte limpa (PDF digital), parte com OCR ruim (foto)
            linhas.append({"texto": ruido_ocr(texto, intens), "rotulo": rotulo, "ruido": intens})
            if rotulo in ("nota_fiscal", "boleto") and intens == 0 and len([g for g in gabarito if g["rotulo"] == rotulo]) < 40:
                gabarito.append({"rotulo": rotulo, "texto": texto, "campos": campos})
    df = pd.DataFrame(linhas).sample(frac=1, random_state=42).reset_index(drop=True)
    destino = Path(__file__).parent
    df.to_csv(destino / "dataset.csv", index=False)
    with open(destino / "gabarito_extracao.jsonl", "w", encoding="utf-8") as f:
        for g in gabarito:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")
    print(f"{len(df)} documentos gerados -> dados/dataset.csv")
    print(df.rotulo.value_counts().to_string())
    print(f"{len(gabarito)} documentos com gabarito de extração -> dados/gabarito_extracao.jsonl")


if __name__ == "__main__":
    main()
