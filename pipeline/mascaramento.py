"""Mascaramento de dados pessoais antes de mandar o texto para a API do LLM (LGPD).

É uma pseudonimização reversível: CPF, nome de pessoa, e-mail, telefone e CEP viram
marcadores como [CPF_1] e [PESSOA_2]. O LLM trabalha só com os marcadores e, quando a resposta
volta, a gente troca de novo pelo valor real AQUI, localmente. O dado pessoal nunca sai da máquina.

CNPJ e nome de empresa não são mascarados: são dados públicos de pessoa jurídica e são
justamente o que a gente quer extrair (emitente, beneficiário).
"""
from __future__ import annotations

import re

PADROES = [
    ("CPF", re.compile(r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b")),
    ("CPF", re.compile(r"(?<=CPF)[:\s]*\d{11}\b")),
    ("CPF", re.compile(r"[•*]{3}\.\d{3}\.\d{3}-[•*]{2}")),
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")),
    ("TELEFONE", re.compile(r"(?:\+55\s?)?\(?\b\d{2}\)?\s?9?\d{4}-\d{4}\b")),
    ("CEP", re.compile(r"\b\d{5}-\d{3}\b")),
]

ROTULOS_PESSOA = (r"Pagador|Sacado|Destinat[áa]rio|Cliente|Consumidor|Nome|Favorecido|Remetente|Solicitante|"
                  r"Recebi de|Prezad[oa](?:\(a\))?|Aluno|Paciente|Titular")
RE_ROTULO = re.compile(rf"(?P<rot>\b(?:{ROTULOS_PESSOA})\b\s*:?\s*)(?P<nome>[A-ZÁÉÍÓÚÂÊÔÃÕÇ][A-Za-zÀ-ú'. ]{{2,60}})")
RE_PARTE_CONTRATO = re.compile(r"(?m)^(?P<nome>[A-ZÁÉÍÓÚÂÊÔÃÕÇ][a-zà-ú']+(?: (?:d[aeo]s?|[A-ZÁÉÍÓÚÂÊÔÃÕÇ][a-zà-ú'.]+))+)(?=, brasileir)")
RE_ENDERECO = re.compile(r"(?<=residente à )[^\n]+?(?=, doravante)")
PALAVRAS_PARADA = {"CPF", "CNPJ", "Instituição", "Instituicao", "Chave", "Banco", "Agência"}


class Mascarador:
    def __init__(self):
        self.mapa: dict[str, str] = {}      # marcador -> valor real
        self._reverso: dict[str, str] = {}  # valor real -> marcador
        self._cont: dict[str, int] = {}

    def _marcador(self, tipo: str, valor: str) -> str:
        valor = valor.strip()
        if valor in self._reverso:
            return self._reverso[valor]
        self._cont[tipo] = self._cont.get(tipo, 0) + 1
        m = f"[{tipo}_{self._cont[tipo]}]"
        self.mapa[m] = valor
        self._reverso[valor] = m
        return m

    def mascarar(self, texto: str) -> str:
        texto = RE_ENDERECO.sub(lambda m: self._marcador("ENDERECO", m.group(0)), texto)
        texto = RE_PARTE_CONTRATO.sub(lambda m: self._marcador("PESSOA", m.group("nome")), texto)

        def troca_rotulo(m):
            nome = m.group("nome")
            palavras = nome.split()
            corte = next((i for i, p in enumerate(palavras) if p in PALAVRAS_PARADA), len(palavras))
            nome_limpo = " ".join(palavras[:corte]).strip(" .-")
            # nome de pessoa: 2 a 6 palavras, sem cara de empresa
            if not (2 <= len(nome_limpo.split()) <= 6) or re.search(r"\b(Ltda|LTDA|S\.A\.|ME|EIRELI|S/A)\b", nome_limpo):
                return m.group(0)
            resto = nome[len(" ".join(palavras[:corte])):]
            return m.group("rot") + self._marcador("PESSOA", nome_limpo) + resto

        texto = RE_ROTULO.sub(troca_rotulo, texto)
        # aplica nomes já conhecidos em outras partes do texto (ex.: assinatura no fim)
        for valor, marc in sorted(self._reverso.items(), key=lambda x: -len(x[0])):
            if marc.startswith("[PESSOA"):
                texto = texto.replace(valor, marc)
        for tipo, rx in PADROES:
            texto = rx.sub(lambda m: (" " if m.group(0).startswith((":", " ")) else "") + self._marcador(tipo, m.group(0).strip(": ")), texto)
        return texto

    def restaurar(self, obj):
        """Troca os marcadores de volta pelos valores reais em qualquer estrutura (dict, list, str)."""
        if isinstance(obj, str):
            for m, v in self.mapa.items():
                obj = obj.replace(m, v)
            return obj
        if isinstance(obj, dict):
            return {k: self.restaurar(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self.restaurar(v) for v in obj]
        return obj

    @property
    def resumo(self) -> dict[str, int]:
        return dict(self._cont)
