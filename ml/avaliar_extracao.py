"""Compara a extração por LLM com a extração por regex usando o gabarito da base sintética.

Para cada documento do gabarito, testa o texto limpo e uma versão com OCR ruim.
Mede a % de campos corretos por abordagem. É o número que mostra na apresentação
"onde só regra/regex falharia".

Uso:  python ml/avaliar_extracao.py --n 20        (precisa de ANTHROPIC_API_KEY ou Ollama)
Saída: relatorios/avaliacao_extracao.json e relatorios/avaliacao_extracao.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dados"))
import gerar_dataset as g  # noqa: E402
from config import DADOS, RELATORIOS  # noqa: E402
from pipeline.extrator_llm import extrair, extrair_regex, provedor_ativo  # noqa: E402
from pipeline.mascaramento import Mascarador  # noqa: E402
from pipeline.validacao import normalizar, parse_data, parse_valor, so_digitos  # noqa: E402

CAMPOS_AVALIADOS = {
    "nota_fiscal": ["emitente_nome", "emitente_cnpj", "data_emissao", "valor_total", "chave_acesso"],
    "boleto": ["beneficiario_nome", "beneficiario_cnpj", "valor", "vencimento", "linha_digitavel"],
}


def igual(campo, esperado, obtido) -> bool:
    if obtido in (None, ""):
        return False
    if campo.startswith("valor"):
        v = parse_valor(obtido)
        return v is not None and abs(v - float(esperado)) < 0.01
    if campo.startswith("data") or campo == "vencimento":
        return parse_data(obtido) == parse_data(esperado)
    if campo.endswith(("cnpj", "chave_acesso", "linha_digitavel")):
        return so_digitos(obtido) == so_digitos(esperado)
    return normalizar(obtido) == normalizar(esperado)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="documentos por tipo")
    args = ap.parse_args()
    if provedor_ativo() == "sem_llm":
        print("Sem LLM configurado. Defina ANTHROPIC_API_KEY (ou TDE2_LLM=ollama) e rode de novo.")
        return
    gab = [json.loads(l) for l in open(DADOS / "gabarito_extracao.jsonl", encoding="utf-8")]
    g.R.seed(99)
    placar = {}  # (abordagem, condicao) -> [acertos, total]
    for tipo in CAMPOS_AVALIADOS:
        docs = [d for d in gab if d["rotulo"] == tipo][: args.n]
        for i, d in enumerate(docs):
            for condicao, texto in (("texto limpo", d["texto"]), ("OCR ruim", g.ruido_ocr(d["texto"], 0.5, recorte=1))):
                masc = Mascarador()
                saida_llm = masc.restaurar(extrair(tipo, masc.mascarar(texto)).campos)
                saida_rx = extrair_regex(tipo, texto).campos
                for abord, saida in (("LLM", saida_llm), ("Regex", saida_rx)):
                    for c in CAMPOS_AVALIADOS[tipo]:
                        k = (abord, condicao)
                        placar.setdefault(k, [0, 0])
                        placar[k][0] += igual(c, d["campos"][c], saida.get(c))
                        placar[k][1] += 1
            print(f"{tipo} {i + 1}/{len(docs)}", end="\r")
    res = {f"{a} - {c}": round(ac / tot, 3) for (a, c), (ac, tot) in placar.items()}
    print("\n", json.dumps(res, ensure_ascii=False, indent=2))
    RELATORIOS.mkdir(exist_ok=True)
    json.dump(res, open(RELATORIOS / "avaliacao_extracao.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    fig, ax = plt.subplots(figsize=(7, 4))
    conds = ["texto limpo", "OCR ruim"]
    for i, abord in enumerate(["Regex", "LLM"]):
        vals = [res.get(f"{abord} - {c}", 0) for c in conds]
        b = ax.bar([x + i * 0.38 for x in range(2)], vals, 0.38, label=abord)
        ax.bar_label(b, fmt="%.0f%%", labels=[f"{v * 100:.0f}%" for v in vals])
    ax.set_xticks([0.19, 1.19], conds)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("campos extraídos corretamente")
    ax.set_title("Extração de campos: regex x LLM")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(RELATORIOS / "avaliacao_extracao.png", dpi=150)
    print("Salvo em relatorios/avaliacao_extracao.json e .png")


if __name__ == "__main__":
    main()
