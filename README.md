# TDE2 · Classificação e extração inteligente de documentos

Sistema que recebe um documento (PDF ou foto), descobre o tipo com um classificador treinado pelo grupo,
extrai os campos com um LLM, valida tudo que dá pra validar e organiza num banco com painel e alertas.

```
Entrada (PDF/foto) → OCR → ML (tipo + confiança) → mascaramento → LLM (campos + resumo) → validação → SQLite + painel + alertas
                                  └── confiança baixa ──→ revisão humana
```

## Instalação (Windows)

1. **Python 3.10+** e, na pasta do projeto:
   ```
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. **Tesseract com português.** Baixe o instalador do UB Mannheim (github.com/UB-Mannheim/tesseract/wiki).
   Na instalação, em *Additional language data*, marque **Portuguese**. Depois, no `.env`, aponte o executável:
   `TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe`
   (Linux: `sudo apt install tesseract-ocr tesseract-ocr-por`. Mac: `brew install tesseract tesseract-lang`.)
3. **Chave do LLM.** Copie `.env.exemplo` para `.env` e coloque a `ANTHROPIC_API_KEY`.
   Sem chave, o sistema roda em modo *sem LLM* (só regex), útil para testar o resto.
   Para usar modelo aberto local: instale o Ollama, rode `ollama pull qwen2.5:7b` e coloque `TDE2_LLM=ollama` no `.env`.

## Rodando

O zip já vem com a base, o modelo treinado e os documentos de demo. Para regerar do zero:

```
python dados/gerar_dataset.py          # base própria pt-BR (2.000 documentos fictícios)
python ml/treinar.py                   # treina e compara 4 abordagens, salva métricas e gráficos
python demo/gerar_documentos_demo.py   # documentos de demo com datas relativas a hoje
python ml/avaliar_extracao.py --n 20   # LLM x regex na extração (precisa da chave)
streamlit run app.py
```

> Se aparecer erro ao carregar `modelos/classificador.joblib` (versão diferente do scikit-learn), é só rodar `python ml/treinar.py` de novo.
> **Rode `gerar_documentos_demo.py` no dia da apresentação** para os alertas (boleto vencendo, contrato terminando) aparecerem.

## Estrutura

| Arquivo | O que faz |
|---|---|
| `app.py` | Interface Streamlit: envio, revisão humana, painel, métricas do modelo, ética |
| `config.py` | Configurações (limiar de confiança, provedor do LLM, alertas) |
| `dados/gerar_dataset.py` | Gera a base própria com documentos fictícios e ruído de OCR |
| `ml/treinar.py` | Treina e compara: regras, TF-IDF+Regressão Logística, TF-IDF+SVM, TF-IDF de caracteres+SVM |
| `ml/avaliar_extracao.py` | Mede extração LLM x regex com o gabarito |
| `pipeline/ocr.py` | PyMuPDF para PDF digital, Tesseract com tratamento de imagem para foto |
| `pipeline/classificador.py` | Carrega o modelo e devolve tipo + confiança |
| `pipeline/mascaramento.py` | Troca CPF, nomes, e-mail, telefone, CEP e endereço por marcadores antes da API |
| `pipeline/extrator_llm.py` | Prompt por tipo, saída JSON, Anthropic / Ollama / regex |
| `pipeline/validacao.py` | CNPJ, linha digitável, chave NF-e, consistência e evidência no texto |
| `pipeline/banco.py` | SQLite, alertas e exclusão |
| `pipeline/processar.py` | Orquestra o pipeline e decide entre salvar e mandar para revisão |

## Divisão sugerida (3 pessoas)

- **ML:** `dados/`, `ml/treinar.py`, slides de métricas e comparação.
- **OCR e LLM:** `pipeline/ocr.py`, `mascaramento.py`, `extrator_llm.py`, `validacao.py`, `ml/avaliar_extracao.py`.
- **Sistema:** `app.py`, `banco.py`, `processar.py`, roteiro e ensaio da demo.
