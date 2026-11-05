"""Pré-processamento usado no treino e na inferência (precisa ser o mesmo nos dois)."""
import re
import unicodedata


def preprocessar(texto: str) -> str:
    """Minúsculas, sem acento, números viram tokens genéricos.

    Trocar números por tokens evita que o modelo decore CNPJ, valores e datas específicos
    e faz ele aprender o formato ("tem 47 dígitos agrupados = linha digitável").
    """
    t = unicodedata.normalize("NFKD", str(texto)).encode("ascii", "ignore").decode().lower()
    t = re.sub(r"\d{2}/\d{2}/\d{2,4}", " _data_ ", t)
    t = re.sub(r"\d{5}\.\d{5}\s+\d{5}\.\d{6}\s+\d{5}\.\d{6}\s+\d\s+\d{14}", " _linhadigitavel_ ", t)
    t = re.sub(r"(\d{4}\s){10}\d{4}", " _chave44_ ", t)
    t = re.sub(r"\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}", " _cnpj_ ", t)
    t = re.sub(r"\d{3}\.\d{3}\.\d{3}-\d{2}", " _cpf_ ", t)
    t = re.sub(r"\d+[.,]\d{2}\b", " _valor_ ", t)
    t = re.sub(r"\d+", " _n_ ", t)
    return re.sub(r"\s+", " ", t).strip()
