"""Etapa 5: armazenamento (SQLite) e alertas."""
from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime

import pandas as pd

from config import BANCO, DIAS_ALERTA_BOLETO, DIAS_ALERTA_CONTRATO
from pipeline.validacao import parse_data, parse_valor

SQL_CRIAR = """
CREATE TABLE IF NOT EXISTS documentos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    criado_em TEXT NOT NULL,
    arquivo TEXT,
    hash TEXT,
    metodo_ocr TEXT,
    qualidade_ocr REAL,
    tipo_previsto TEXT,
    confianca REAL,
    probabilidades TEXT,
    tipo_final TEXT,
    status TEXT,              -- 'salvo' | 'revisao' | 'revisado' | 'descartado'
    motivo_revisao TEXT,
    campos TEXT,
    resumo TEXT,
    validacoes TEXT,
    motor_llm TEXT,
    texto_ocr TEXT,           -- fica só no banco local; é o que permite reprocessar na revisão
    texto_mascarado TEXT,
    itens_mascarados TEXT,
    revisado_em TEXT,
    pago_em TEXT,             -- boletos: data do pagamento (vazio = em aberto)
    pago_por INTEGER          -- boletos: id do comprovante que deu a baixa (vazio = marcado à mão)
);
"""
COLUNAS_NOVAS = {"pago_em": "TEXT", "pago_por": "INTEGER"}  # bancos criados antes da baixa de boletos


def conectar() -> sqlite3.Connection:
    con = sqlite3.connect(BANCO, check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.execute(SQL_CRIAR)
    existentes = {r["name"] for r in con.execute("PRAGMA table_info(documentos)")}
    for col, tipo in COLUNAS_NOVAS.items():
        if col not in existentes:
            con.execute(f"ALTER TABLE documentos ADD COLUMN {col} {tipo}")
    return con


def inserir(reg: dict) -> int:
    with conectar() as con:
        reg = {**reg, "criado_em": datetime.now().isoformat(timespec="seconds")}
        for k in ("probabilidades", "campos", "validacoes", "itens_mascarados"):
            if k in reg and not isinstance(reg[k], str):
                reg[k] = json.dumps(reg[k], ensure_ascii=False, default=str)
        cols = ", ".join(reg)
        cur = con.execute(f"INSERT INTO documentos ({cols}) VALUES ({', '.join('?' * len(reg))})", list(reg.values()))
        return cur.lastrowid


def atualizar_revisao(doc_id: int, tipo_final: str, campos: dict, status: str = "revisado"):
    with conectar() as con:
        con.execute("UPDATE documentos SET tipo_final=?, campos=?, status=?, revisado_em=? WHERE id=?",
                    (tipo_final, json.dumps(campos, ensure_ascii=False, default=str), status, datetime.now().isoformat(timespec="seconds"), doc_id))


def excluir(doc_id: int):
    """Apaga de verdade o documento e o texto (direito de exclusão, LGPD art. 18)."""
    with conectar() as con:
        con.execute("DELETE FROM documentos WHERE id=?", (doc_id,))
        # sem o comprovante, a baixa que ele deu perde a prova: o boleto volta a ficar em aberto
        con.execute("UPDATE documentos SET pago_em=NULL, pago_por=NULL WHERE pago_por=?", (doc_id,))


def marcar_pago(boleto_id: int, pago_em: date | None = None, comprovante_id: int | None = None):
    with conectar() as con:
        con.execute("UPDATE documentos SET pago_em=?, pago_por=? WHERE id=?",
                    ((pago_em or date.today()).isoformat(), comprovante_id, boleto_id))


def desmarcar_pago(boleto_id: int):
    with conectar() as con:
        con.execute("UPDATE documentos SET pago_em=NULL, pago_por=NULL WHERE id=?", (boleto_id,))


def boleto_pago_por(comprovante_id: int) -> int | None:
    with conectar() as con:
        r = con.execute("SELECT id FROM documentos WHERE pago_por=?", (comprovante_id,)).fetchone()
        return r["id"] if r else None


def boletos_em_aberto() -> pd.DataFrame:
    df = listar(["salvo", "revisado"])
    return df[(df.tipo_final == "boleto") & df.pago_em.isna()]


def atualizar_extracao(doc_id: int, tipo: str, campos: dict, resumo: str, validacoes: list, motor: str, texto_mascarado: str, itens: dict):
    with conectar() as con:
        con.execute("UPDATE documentos SET tipo_final=?, campos=?, resumo=?, validacoes=?, motor_llm=?, texto_mascarado=?, itens_mascarados=? WHERE id=?",
                    (tipo, json.dumps(campos, ensure_ascii=False, default=str), resumo, json.dumps(validacoes, ensure_ascii=False, default=str),
                     motor, texto_mascarado, json.dumps(itens), doc_id))


def obter(doc_id: int) -> dict | None:
    df = listar()
    df = df[df.id == doc_id]
    return df.iloc[0].to_dict() if len(df) else None


def existe_hash(h: str) -> int | None:
    with conectar() as con:
        r = con.execute("SELECT id FROM documentos WHERE hash=? AND status!='descartado'", (h,)).fetchone()
        return r["id"] if r else None


def listar(status: list[str] | None = None) -> pd.DataFrame:
    with conectar() as con:
        q = "SELECT * FROM documentos"
        if status:
            q += f" WHERE status IN ({', '.join('?' * len(status))})"
        df = pd.read_sql_query(q + " ORDER BY id DESC", con, params=status or [])
    for k in ("campos", "validacoes", "probabilidades", "itens_mascarados"):
        if k in df:
            df[k] = df[k].apply(lambda s: json.loads(s) if s else {})
    return df


def limpar():
    with conectar() as con:
        con.execute("DELETE FROM documentos")


def alertas(hoje: date | None = None) -> list[dict]:
    """Boletos vencendo/vencidos e contratos perto do fim."""
    hoje = hoje or date.today()
    df = listar(["salvo", "revisado"])
    out = []
    for _, r in df.iterrows():
        c = r["campos"] or {}
        if r["tipo_final"] == "boleto" and pd.isna(r["pago_em"]):
            venc = parse_data(c.get("vencimento"))
            if venc:
                dias = (venc - hoje).days
                if dias < 0:
                    out.append({"nivel": "erro", "id": r["id"], "texto": f"Boleto de {c.get('beneficiario_nome') or '?'} venceu há {-dias} dia(s)", "valor": parse_valor(c.get("valor")), "data": venc})
                elif dias <= DIAS_ALERTA_BOLETO:
                    out.append({"nivel": "aviso", "id": r["id"], "texto": f"Boleto de {c.get('beneficiario_nome') or '?'} vence em {dias} dia(s)", "valor": parse_valor(c.get("valor")), "data": venc})
        if r["tipo_final"] == "contrato":
            fim = parse_data(c.get("data_termino"))
            if fim:
                dias = (fim - hoje).days
                if 0 <= dias <= DIAS_ALERTA_CONTRATO:
                    out.append({"nivel": "aviso", "id": r["id"], "texto": f"Contrato com {c.get('contratada_nome') or '?'} termina em {dias} dia(s)", "valor": parse_valor(c.get("valor_mensal")), "data": fim})
                elif dias < 0:
                    out.append({"nivel": "info", "id": r["id"], "texto": f"Contrato com {c.get('contratada_nome') or '?'} encerrado em {fim:%d/%m/%Y}", "valor": None, "data": fim})
    return sorted(out, key=lambda a: a["data"])
