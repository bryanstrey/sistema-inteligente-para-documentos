"""Etapa 3: classifica o tipo do documento com o modelo treinado em ml/treinar.py."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import joblib

from config import LIMIAR_CONFIANCA, MODELOS


@dataclass
class Classificacao:
    tipo: str
    confianca: float
    probabilidades: dict[str, float]
    modelo: str

    @property
    def precisa_revisao(self) -> bool:
        return self.confianca < LIMIAR_CONFIANCA


@lru_cache(maxsize=1)
def _carregar():
    caminho = MODELOS / "classificador.joblib"
    if not caminho.exists():
        raise FileNotFoundError("Modelo não encontrado. Rode antes: python dados/gerar_dataset.py && python ml/treinar.py")
    return joblib.load(caminho)


def classificar(texto: str) -> Classificacao:
    pacote = _carregar()
    modelo = pacote["modelo"]
    probs = modelo.predict_proba([texto])[0]
    dist = {c: float(p) for c, p in zip(modelo.classes_, probs)}
    tipo = max(dist, key=dist.get)
    return Classificacao(tipo, dist[tipo], dict(sorted(dist.items(), key=lambda x: -x[1])), pacote["nome"])
