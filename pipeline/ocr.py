"""Etapa 2: transforma PDF ou imagem em texto.

- PDF com texto digital: extrai direto com PyMuPDF (rápido e sem erro).
- PDF escaneado ou foto: trata a imagem e roda o Tesseract.
Devolve também uma estimativa de qualidade (confiança média do Tesseract).
"""
from __future__ import annotations

import io
from dataclasses import dataclass

import pymupdf
import pytesseract
from PIL import Image, ImageFilter, ImageOps

from config import IDIOMA_OCR, TESSERACT_CMD

if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD


@dataclass
class ResultadoOCR:
    texto: str
    metodo: str            # "pdf_texto" ou "ocr"
    qualidade: float       # 0 a 100 (100 para PDF digital)
    paginas: int
    imagem_tratada: Image.Image | None = None

    @property
    def qualidade_baixa(self) -> bool:
        return self.metodo == "ocr" and self.qualidade < 60


def tratar_imagem(img: Image.Image) -> Image.Image:
    """Pré-processamento simples que melhora muito o Tesseract em foto de celular."""
    img = ImageOps.exif_transpose(img)  # respeita a rotação da câmera
    img = img.convert("L")
    if max(img.size) < 1800:  # Tesseract gosta de texto com ~30px de altura
        fator = 1800 / max(img.size)
        img = img.resize((int(img.width * fator), int(img.height * fator)), Image.LANCZOS)
    img = ImageOps.autocontrast(img, cutoff=2)
    img = img.filter(ImageFilter.MedianFilter(3))
    img = img.filter(ImageFilter.SHARPEN)
    return img


def _corrigir_rotacao(img: Image.Image) -> Image.Image:
    try:
        osd = pytesseract.image_to_osd(img, output_type=pytesseract.Output.DICT)
        if osd.get("rotate"):
            return img.rotate(-osd["rotate"], expand=True)
    except pytesseract.TesseractError:
        pass
    return img


def ocr_imagem(img: Image.Image) -> tuple[str, float, Image.Image]:
    img = tratar_imagem(img)
    img = _corrigir_rotacao(img)
    dados = pytesseract.image_to_data(img, lang=IDIOMA_OCR, config="--psm 3", output_type=pytesseract.Output.DICT)
    confs = [float(c) for c, t in zip(dados["conf"], dados["text"]) if t.strip() and float(c) >= 0]
    qualidade = sum(confs) / len(confs) if confs else 0.0
    texto = pytesseract.image_to_string(img, lang=IDIOMA_OCR, config="--psm 3")
    return texto, qualidade, img


def extrair_texto(conteudo: bytes, nome_arquivo: str) -> ResultadoOCR:
    nome = nome_arquivo.lower()
    if nome.endswith(".pdf"):
        doc = pymupdf.open(stream=conteudo, filetype="pdf")
        texto = "\n".join(p.get_text() for p in doc)
        if len(texto.strip()) > 80:
            return ResultadoOCR(texto, "pdf_texto", 100.0, len(doc))
        # PDF escaneado: renderiza as páginas e faz OCR
        textos, quals, primeira = [], [], None
        for pagina in doc:
            pix = pagina.get_pixmap(dpi=300)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            t, q, tratada = ocr_imagem(img)
            textos.append(t)
            quals.append(q)
            primeira = primeira or tratada
        return ResultadoOCR("\n".join(textos), "ocr", sum(quals) / max(len(quals), 1), len(doc), primeira)
    img = Image.open(io.BytesIO(conteudo))
    texto, qualidade, tratada = ocr_imagem(img)
    return ResultadoOCR(texto, "ocr", qualidade, 1, tratada)
