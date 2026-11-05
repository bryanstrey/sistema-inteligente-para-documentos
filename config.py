"""Configurações do projeto. Valores podem ser sobrescritos por variáveis de ambiente."""
import os
from pathlib import Path

RAIZ = Path(__file__).resolve().parent

# carrega o arquivo .env (se existir) sem precisar de biblioteca extra
_env = RAIZ / ".env"
if _env.exists():
    for _l in _env.read_text(encoding="utf-8").splitlines():
        _l = _l.strip()
        if _l and not _l.startswith("#") and "=" in _l:
            _k, _v = _l.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())
DADOS = RAIZ / "dados"
MODELOS = RAIZ / "modelos"
RELATORIOS = RAIZ / "relatorios"
BANCO = Path(os.getenv("TDE2_BANCO", RAIZ / "documentos.db"))

CATEGORIAS = ["nota_fiscal", "boleto", "comprovante", "contrato", "outros"]
NOMES_CATEGORIAS = {
    "nota_fiscal": "Nota fiscal",
    "boleto": "Boleto",
    "comprovante": "Comprovante de pagamento",
    "contrato": "Contrato",
    "outros": "Outros",
}

# Abaixo deste valor o documento vai para revisão humana em vez de ser salvo direto.
LIMIAR_CONFIANCA = float(os.getenv("TDE2_LIMIAR", "0.60"))

# Alertas
DIAS_ALERTA_BOLETO = 7
DIAS_ALERTA_CONTRATO = 30

# LLM: "anthropic" (API comercial), "ollama" (modelo aberto local) ou "sem_llm" (só regex, para testes)
PROVEDOR_LLM = os.getenv("TDE2_LLM", "anthropic")
MODELO_ANTHROPIC = os.getenv("TDE2_MODELO_ANTHROPIC", "claude-haiku-5-5")
MODELO_OLLAMA = os.getenv("TDE2_MODELO_OLLAMA", "qwen2.5:7b")
URL_OLLAMA = os.getenv("TDE2_URL_OLLAMA", "http://localhost:11434")

# Mascarar dados pessoais antes de mandar para a API externa (recomendado manter True)
MASCARAR_DADOS = os.getenv("TDE2_MASCARAR", "1") == "1"

# Tesseract: no Windows, aponte para o executável, ex.: C:\Program Files\Tesseract-OCR\tesseract.exe
TESSERACT_CMD = os.getenv("TESSERACT_CMD", "")
IDIOMA_OCR = os.getenv("TDE2_IDIOMA_OCR", "por")
