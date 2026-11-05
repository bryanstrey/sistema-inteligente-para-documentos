"""Treina e compara classificadores de tipo de documento.

Abordagens comparadas:
  0. Regras por palavra-chave (linha de base, sem ML)
  1. TF-IDF de palavras + Regressão Logística
  2. TF-IDF de palavras + SVM linear (calibrado para dar probabilidade)
  3. TF-IDF de n-gramas de caracteres + SVM linear (calibrado) — mais robusto a erro de OCR

Avaliação:
  - divisão estratificada 80/20 + validação cruzada de 5 dobras no treino
  - teste de robustez: documentos novos com OCR muito ruim (simula foto escura/torta)

Uso:  python ml/treinar.py
Saídas: modelos/classificador.joblib, relatorios/metricas.json, relatorios/*.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.base import BaseEstimator, ClassifierMixin  # noqa: E402
from sklearn.calibration import CalibratedClassifierCV  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,  # noqa: E402
                             confusion_matrix, f1_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.svm import LinearSVC  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import CATEGORIAS, DADOS, MODELOS, NOMES_CATEGORIAS, RELATORIOS  # noqa: E402
from ml.texto import preprocessar  # noqa: E402


class RegrasPalavraChave(BaseEstimator, ClassifierMixin):
    """Linha de base sem aprendizado: conta palavras-chave de cada tipo."""

    PALAVRAS = {
        "nota_fiscal": ["danfe", "nf-e", "nfc-e", "nota fiscal", "chave de acesso", "emissao", "tributos"],
        "boleto": ["vencimento", "beneficiario", "pagador", "local de pagamento", "nosso numero", "sacado", "cedente"],
        "comprovante": ["comprovante", "transferencia", "pix", "autenticacao", "pagamento realizado", "aprovado", "favorecido"],
        "contrato": ["contrato", "clausula", "contratante", "contratada", "foro", "vigencia", "locatario"],
    }

    def fit(self, X, y):
        self.classes_ = np.array(CATEGORIAS)
        return self

    def predict(self, X):
        out = []
        for t in X:
            pont = {k: sum(p in t for p in ps) for k, ps in self.PALAVRAS.items()}
            melhor = max(pont, key=pont.get)
            out.append(melhor if pont[melhor] >= 2 else "outros")
        return np.array(out)


def tfidf_palavras():
    return TfidfVectorizer(preprocessor=preprocessar, ngram_range=(1, 2), min_df=2, sublinear_tf=True)


def tfidf_caracteres():
    return TfidfVectorizer(preprocessor=preprocessar, analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True, max_features=60000)


MODELOS_CANDIDATOS = {
    "Regras (sem ML)": lambda: Pipeline([("pre", _Pre()), ("clf", RegrasPalavraChave())]),
    "TF-IDF palavras + Regressão Logística": lambda: Pipeline([("tfidf", tfidf_palavras()), ("clf", LogisticRegression(max_iter=2000, C=5))]),
    "TF-IDF palavras + SVM": lambda: Pipeline([("tfidf", tfidf_palavras()), ("clf", CalibratedClassifierCV(LinearSVC(C=1), cv=3))]),
    "TF-IDF caracteres + SVM": lambda: Pipeline([("tfidf", tfidf_caracteres()), ("clf", CalibratedClassifierCV(LinearSVC(C=1), cv=3))]),
}


class _Pre(BaseEstimator):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [preprocessar(t) for t in X]


def gerar_teste_robustez(n_por_classe=60):
    sys.path.insert(0, str(DADOS))
    import gerar_dataset as g  # noqa: E402
    g.R.seed(2026)
    g.Faker.seed(2026)
    textos, rotulos = [], []
    for rot, gen in g.GERADORES.items():
        for _ in range(n_por_classe):
            t, _ = gen()
            textos.append(g.ruido_ocr(t, 0.9, recorte=g.R.uniform(0.25, 0.5)))
            rotulos.append(rot)
    return textos, rotulos


def main():
    RELATORIOS.mkdir(exist_ok=True)
    MODELOS.mkdir(exist_ok=True)
    df = pd.read_csv(DADOS / "dataset.csv")
    X_tr, X_te, y_tr, y_te = train_test_split(df.texto, df.rotulo, test_size=0.2, stratify=df.rotulo, random_state=42)
    X_rob, y_rob = gerar_teste_robustez()
    cv = StratifiedKFold(5, shuffle=True, random_state=42)

    resultados, treinados = {}, {}
    for nome, fab in MODELOS_CANDIDATOS.items():
        print(f"\n== {nome}")
        modelo = fab()
        cv_f1 = cross_val_score(fab(), X_tr, y_tr, cv=cv, scoring="f1_macro") if "Regras" not in nome else np.array([np.nan])
        modelo.fit(X_tr, y_tr)
        pred = modelo.predict(X_te)
        pred_rob = modelo.predict(X_rob)
        resultados[nome] = {
            "acuracia": accuracy_score(y_te, pred),
            "f1_macro": f1_score(y_te, pred, average="macro"),
            "f1_cv_media": float(np.nanmean(cv_f1)) if not np.isnan(cv_f1).all() else None,
            "f1_cv_desvio": float(np.nanstd(cv_f1)) if not np.isnan(cv_f1).all() else None,
            "f1_ocr_ruim": f1_score(y_rob, pred_rob, average="macro"),
            "relatorio": classification_report(y_te, pred, output_dict=True, zero_division=0),
        }
        treinados[nome] = modelo
        r = resultados[nome]
        print(f"acurácia {r['acuracia']:.3f} | F1 {r['f1_macro']:.3f} | F1 CV {r['f1_cv_media']} | F1 OCR ruim {r['f1_ocr_ruim']:.3f}")

    # escolha: melhor média entre documento limpo e OCR ruim (fotos são o caso real mais difícil)
    candidatos = {k: v for k, v in resultados.items() if "Regras" not in k}
    escolhido = max(candidatos, key=lambda k: (candidatos[k]["f1_macro"] + candidatos[k]["f1_ocr_ruim"]) / 2)
    print(f"\nModelo escolhido: {escolhido}")

    # retreina o escolhido com todos os dados para o sistema
    final = MODELOS_CANDIDATOS[escolhido]()
    final.fit(df.texto, df.rotulo)
    joblib.dump({"modelo": final, "nome": escolhido, "classes": list(final.classes_)}, MODELOS / "classificador.joblib")

    # matriz de confusão do escolhido (conjunto de teste)
    m = treinados[escolhido]
    rotulos = [NOMES_CATEGORIAS[c] for c in CATEGORIAS]
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, (X, y, tit) in zip(axs, [(X_te, y_te, "Teste (20%)"), (X_rob, y_rob, "Teste com OCR muito ruim")]):
        cm = confusion_matrix(y, m.predict(X), labels=CATEGORIAS)
        ConfusionMatrixDisplay(cm, display_labels=rotulos).plot(ax=ax, cmap="Blues", colorbar=False, xticks_rotation=30)
        ax.set_title(f"{tit}")
    fig.suptitle(f"Matriz de confusão — {escolhido}")
    fig.tight_layout()
    fig.savefig(RELATORIOS / "matriz_confusao.png", dpi=150)

    # comparação das abordagens
    nomes = list(resultados)
    x = np.arange(len(nomes))
    fig, ax = plt.subplots(figsize=(11, 4.8))
    for i, (chave, rot) in enumerate([("acuracia", "Acurácia"), ("f1_macro", "F1 macro"), ("f1_ocr_ruim", "F1 com OCR ruim")]):
        vals = [resultados[n][chave] for n in nomes]
        barras = ax.bar(x + (i - 1) * 0.26, vals, 0.26, label=rot)
        ax.bar_label(barras, fmt="%.2f", fontsize=8)
    ax.set_xticks(x, [n.replace(" + ", "\n+ ") for n in nomes], fontsize=9)
    ax.set_ylim(0, 1.22)
    ax.legend(loc="upper left", ncol=3, frameon=False)
    ax.set_title("Comparação das abordagens")
    fig.tight_layout()
    fig.savefig(RELATORIOS / "comparacao_modelos.png", dpi=150)

    json.dump({"escolhido": escolhido, "n_treino": len(X_tr), "n_teste": len(X_te), "n_robustez": len(X_rob), "resultados": resultados},
              open(RELATORIOS / "metricas.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\nSalvo: modelos/classificador.joblib, relatorios/metricas.json, relatorios/matriz_confusao.png, relatorios/comparacao_modelos.png")


if __name__ == "__main__":
    main()
