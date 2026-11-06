"""Gera documentos fictícios para a demo ao vivo (PDF digital e "fotos" de celular).

As datas são relativas a hoje, então os alertas aparecem no dia da apresentação.
Uso: python demo/gerar_documentos_demo.py
"""
from __future__ import annotations

import io
import random
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pymupdf
from PIL import Image, ImageEnhance, ImageFilter
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.validacao import gerar_chave_nfe, gerar_linha_digitavel  # noqa: E402

PASTA = Path(__file__).parent / "documentos"
HOJE = date.today()
R = random.Random(7)


def pdf(linhas: list[tuple[str, str, int]]) -> bytes:
    """linhas: (texto, estilo, tamanho). estilo: n (normal), b (negrito), m (mono), c (centralizado negrito), - (linha)."""
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    w, h = A4
    y = h - 60
    for texto, estilo, tam in linhas:
        if estilo == "-":
            c.setLineWidth(0.6)
            c.line(50, y + 4, w - 50, y + 4)
            y -= 10
            continue
        fonte = {"n": "Helvetica", "b": "Helvetica-Bold", "m": "Courier", "c": "Helvetica-Bold"}[estilo]
        c.setFont(fonte, tam)
        if estilo == "c":
            c.drawCentredString(w / 2, y, texto)
        else:
            # quebra simples de linha
            max_chars = int((w - 100) / (tam * 0.5))
            while len(texto) > max_chars:
                corte = texto.rfind(" ", 0, max_chars)
                c.drawString(50, y, texto[:corte])
                texto = texto[corte + 1:]
                y -= tam + 4
            c.drawString(50, y, texto)
        y -= tam + 6
    c.showPage()
    c.save()
    return buf.getvalue()


def foto(pdf_bytes: bytes, angulo=2.0, brilho=0.85, ruido=12, desfoque=0.8, escala_dpi=130) -> bytes:
    """Transforma um PDF em algo parecido com foto de celular: torto, escuro, com ruído e desfoque."""
    pag = pymupdf.open(stream=pdf_bytes, filetype="pdf")[0]
    img = Image.open(io.BytesIO(pag.get_pixmap(dpi=escala_dpi).tobytes("png"))).convert("RGB")
    fundo = Image.new("RGB", (int(img.width * 1.08), int(img.height * 1.06)), (120, 110, 95))  # mesa
    fundo.paste(img, (int(img.width * 0.04), int(img.height * 0.03)))
    img = fundo.rotate(angulo, expand=True, fillcolor=(120, 110, 95), resample=Image.BICUBIC)
    img = ImageEnhance.Brightness(img).enhance(brilho)
    arr = np.asarray(img).astype(np.int16)
    gradiente = np.linspace(0, -35, arr.shape[1])[None, :, None]  # sombra de um lado
    arr = np.clip(arr + np.random.default_rng(1).normal(0, ruido, arr.shape) + gradiente, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr).filter(ImageFilter.GaussianBlur(desfoque))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=70)
    return buf.getvalue()


def br(v):
    return f"{v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


# ----------------------------------------------------------------------------- documentos

def nota_fiscal(data_e: date, itens, emitente="Mercado Bom Preço Ltda", cnpj="11.222.333/0001-81", numero=48213):
    total = round(sum(q * p for _, q, p in itens), 2)
    chave = gerar_chave_nfe("41", data_e, cnpj, "65", 1, numero, 31415926)
    linhas = [
        (emitente.upper(), "c", 13), (f"CNPJ: {cnpj}   IE: 9081726354", "c", 9),
        ("Rua XV de Novembro, 1200, Centro, Curitiba - PR", "c", 9), ("", "-", 0),
        ("DANFE NFC-e - Documento Auxiliar da Nota Fiscal de Consumidor Eletrônica", "c", 9), ("", "-", 0),
        ("Código  Descrição                       Qtd    Vl Unit   Vl Total", "m", 9),
    ]
    for i, (d, q, p) in enumerate(itens, 1):
        linhas.append((f"{i:03d}     {d:<30} {q:>4}  {br(p):>8}  {br(q * p):>9}", "m", 9))
    linhas += [
        ("", "-", 0), (f"Qtd. total de itens                                    {len(itens)}", "m", 9),
        (f"Valor total R$                                    {br(total)}", "b", 10),
        ("Forma de pagamento: Cartão de Crédito", "n", 9), ("", "-", 0),
        (f"NFC-e nº {numero}  Série 001  Emissão: {data_e:%d/%m/%Y} 18:42:10", "n", 9),
        ("Consulte pela Chave de Acesso em www.fazenda.pr.gov.br/nfce", "n", 8), (chave, "m", 9),
        ("CONSUMIDOR - CPF: 123.456.789-09 - Fernanda Rocha Lima", "n", 8),
        (f"Tributos aproximados R$ {br(total * 0.17)} (Fonte IBPT)", "n", 8),
    ]
    return pdf(linhas), total


def linha_boleto(venc: date, valor: float) -> str:
    return gerar_linha_digitavel("341", valor, venc, "1091234567812345600123450")


def boleto(venc: date, valor: float, valor_impresso: float | None = None, benef="Colégio Horizonte Ltda", cnpj="58.301.927/0001-41"):
    linha = linha_boleto(venc, valor)
    vi = valor_impresso if valor_impresso is not None else valor
    linhas = [
        ("Itaú   |  341-7  |  " + linha, "b", 10), ("", "-", 0),
        ("Local de pagamento: Pagável em qualquer banco até o vencimento", "n", 9),
        (f"Vencimento: {venc:%d/%m/%Y}", "b", 11),
        (f"Beneficiário: {benef}   CNPJ {cnpj}", "n", 9),
        ("Agência/Código do beneficiário: 0912/34567-8", "n", 9),
        (f"Data do documento: {venc - timedelta(days=20):%d/%m/%Y}     Nº do documento: 2026110045", "n", 9),
        ("Nosso número: 109/12345678-1      Espécie: R$      Aceite: N", "n", 9),
        (f"(=) Valor do documento: R$ {br(vi)}", "b", 11), ("", "-", 0),
        ("Instruções (texto de responsabilidade do beneficiário)", "b", 9),
        ("Mensalidade escolar referente a novembro/2026.", "n", 9),
        ("Após o vencimento cobrar multa de 2% e juros de 1% ao mês.", "n", 9),
        ("Não receber após 30 dias do vencimento.", "n", 9), ("", "-", 0),
        ("Pagador: Ricardo Almeida Souza   CPF: 987.654.321-00", "n", 9),
        ("Av. Sete de Setembro, 4500, ap 302, Batel, Curitiba - PR, CEP 80240-000", "n", 9),
        ("Ficha de Compensação - Autenticação mecânica", "n", 8),
    ]
    return pdf(linhas)


def contrato(inicio: date, fim: date):
    linhas = [
        ("CONTRATO DE PRESTAÇÃO DE SERVIÇOS Nº 118/2025", "c", 12), ("", "-", 0),
        ("Pelo presente instrumento particular, de um lado:", "n", 10),
        ("Juliana Martins Prado, brasileira, casada, inscrita no CPF sob o nº 321.654.987-11, residente à Rua Mateus Leme, 300, "
         "São Francisco, Curitiba - PR, doravante denominada CONTRATANTE;", "n", 10),
        ("e de outro lado Nuvem Clara Tecnologia Ltda, pessoa jurídica de direito privado, inscrita no CNPJ sob o nº "
         "12.345.678/0001-95, com sede em Curitiba - PR, doravante denominada CONTRATADA;", "n", 10),
        ("têm entre si justo e contratado o seguinte:", "n", 10),
        ("CLÁUSULA PRIMEIRA - DO OBJETO. O presente contrato tem por objeto a manutenção preventiva de equipamentos de "
         "informática e suporte técnico remoto.", "n", 10),
        (f"CLÁUSULA SEGUNDA - DO PRAZO. O prazo de vigência é de 12 (doze) meses, com início em {inicio:%d/%m/%Y} e término em "
         f"{fim:%d/%m/%Y}, podendo ser renovado mediante termo aditivo.", "n", 10),
        ("CLÁUSULA TERCEIRA - DO VALOR. Será pago o valor mensal de R$ 1.850,00, com vencimento todo dia 10.", "n", 10),
        ("CLÁUSULA QUARTA - DA RESCISÃO. Qualquer das partes poderá rescindir mediante aviso prévio de 30 (trinta) dias, "
         "sob pena de multa equivalente a duas mensalidades.", "n", 10),
        ("CLÁUSULA QUINTA - DO FORO. Fica eleito o foro da comarca de Curitiba - PR.", "n", 10),
        (f"Curitiba, {inicio:%d/%m/%Y}.", "n", 10),
        ("_____________________________ CONTRATANTE        _____________________________ CONTRATADA", "n", 9),
    ]
    return pdf(linhas)


def comprovante_pix(d: date):
    return pdf([
        ("Comprovante Pix", "c", 14), ("Banco Inter", "c", 10), ("", "-", 0),
        (f"{d:%d/%m/%Y} às 14:27:05", "n", 11), ("Valor  R$ 350,00", "b", 14),
        ("Tipo de transferência: Pix", "n", 11), ("", "-", 0),
        ("Quem recebeu", "b", 11), ("Nome: Oficina Duas Rodas Ltda", "n", 11), ("CNPJ: 23.456.789/0001-95", "n", 11),
        ("Instituição: Sicredi", "n", 11), ("", "-", 0),
        ("Quem pagou", "b", 11), ("Nome: Ricardo Almeida Souza", "n", 11), ("CPF: 987.654.321-00", "n", 11),
        ("Instituição: Banco Inter", "n", 11), ("", "-", 0),
        ("ID da transação: E00416968202611021427ab12cd34ef5", "m", 9), ("Mensagem: conserto da bicicleta", "n", 10),
    ])


def comprovante_boleto(d: date, venc: date, valor: float):
    """Comprovante do pagamento do boleto da escola: serve para mostrar a baixa automática."""
    return pdf([
        ("Comprovante de pagamento", "c", 14), ("Banco Inter", "c", 10), ("", "-", 0),
        (f"{d:%d/%m/%Y} às 09:12:44", "n", 11), (f"Valor pago  R$ {br(valor)}", "b", 14),
        ("Tipo de pagamento: Boleto bancário", "n", 11), ("", "-", 0),
        ("Código de barras / linha digitável", "b", 11), (linha_boleto(venc, valor), "m", 9), ("", "-", 0),
        ("Beneficiário", "b", 11), ("Nome: Colégio Horizonte Ltda", "n", 11), ("CNPJ: 58.301.927/0001-41", "n", 11),
        (f"Data de vencimento: {venc:%d/%m/%Y}", "n", 11), ("", "-", 0),
        ("Pagador", "b", 11), ("Nome: Ricardo Almeida Souza", "n", 11), ("CPF: 987.654.321-00", "n", 11), ("", "-", 0),
        ("Autenticação: 7F3A.91C2.0B44.6E18.D5A0", "m", 9),
    ])


def orcamento(d: date):
    return pdf([
        ("ORÇAMENTO Nº 314", "c", 14), ("Marcenaria Pinho Forte Ltda - CNPJ 34.567.890/0001-30", "c", 9), ("", "-", 0),
        ("Cliente: Ricardo Almeida Souza", "n", 10), (f"Data: {d:%d/%m/%Y}", "n", 10), ("", "-", 0),
        ("Armário de cozinha sob medida (MDF branco)        4.200,00", "m", 10),
        ("Bancada de granito 2,10m                          1.350,00", "m", 10),
        ("Instalação                                          450,00", "m", 10), ("", "-", 0),
        ("Total estimado R$ 6.000,00", "b", 11), ("Validade da proposta: 15 dias.", "n", 10),
        ("Este documento não possui valor fiscal.", "n", 10), ("Formas de pagamento: PIX, boleto ou cartão em até 10x.", "n", 10),
    ])


def fatura_cartao(venc: date):
    return pdf([
        ("Fatura do cartão de crédito", "c", 14), ("Banco Inter", "c", 10), ("", "-", 0),
        ("Titular: Ricardo Almeida Souza", "n", 10), (f"Vencimento: {venc:%d/%m/%Y}", "b", 11),
        ("Total da fatura: R$ 2.314,55", "b", 12), ("Pagamento mínimo: R$ 347,18", "n", 10), ("", "-", 0),
        ("Lançamentos", "b", 10),
        ("Mercado Bom Preço              144,06", "m", 10), ("Farmácia Vida                  125,91", "m", 10),
        ("Auto Posto Avenida             210,00", "m", 10), ("Streaming Plus                  39,90", "m", 10),
        ("Loja Eletro Sul 3/10           179,90", "m", 10), ("", "-", 0),
        ("Pague pelo app, com Pix ou pelo código de barras.", "n", 10),
        ("Limite total R$ 8.000,00  Limite disponível R$ 5.685,45", "n", 9),
    ])


def main():
    PASTA.mkdir(parents=True, exist_ok=True)
    itens = [("Arroz tipo 1 5kg", 2, 24.90), ("Feijão carioca 1kg", 3, 8.49), ("Café torrado 500g", 1, 18.90),
             ("Leite integral 1L", 6, 5.29), ("Detergente 500ml", 4, 2.79), ("Banana prata kg", 1, 6.99)]
    nf, total = nota_fiscal(HOJE - timedelta(days=3), itens)
    arquivos = {
        "01_nota_fiscal_mercado.pdf": nf,
        "02_boleto_escola.pdf": boleto(HOJE + timedelta(days=4), 1280.00),
        "03_contrato_manutencao.pdf": contrato(HOJE - timedelta(days=345), HOJE + timedelta(days=20)),
        "04_comprovante_pix_foto.jpg": foto(comprovante_pix(HOJE - timedelta(days=1)), angulo=-2.5, brilho=0.8),
        "05_nota_fiscal_foto.jpg": foto(nota_fiscal(HOJE - timedelta(days=10), itens[:4], "Farmácia Vida Ltda", "91.827.364/0001-03", 9921)[0], angulo=3, escala_dpi=160),
        "06_orcamento_marcenaria.pdf": orcamento(HOJE - timedelta(days=2)),
        "07_boleto_adulterado.pdf": boleto(HOJE + timedelta(days=6), 1280.00, valor_impresso=1820.00),
        "08_boleto_foto_escura.jpg": foto(boleto(HOJE + timedelta(days=9), 640.00), angulo=5, brilho=0.5, ruido=35, desfoque=1.1, escala_dpi=105),
        "09_foto_muito_ruim.jpg": foto(boleto(HOJE + timedelta(days=9), 640.00), angulo=4, brilho=0.55, ruido=30, desfoque=1.3, escala_dpi=100),
        "10_fatura_cartao.pdf": fatura_cartao(HOJE + timedelta(days=13)),
        "11_comprovante_boleto_escola.pdf": comprovante_boleto(HOJE, HOJE + timedelta(days=4), 1280.00),
    }
    for nome, conteudo in arquivos.items():
        (PASTA / nome).write_bytes(conteudo)
        print("gerado:", PASTA / nome)


if __name__ == "__main__":
    main()
