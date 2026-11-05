"""Interface do sistema (Streamlit).  Rodar com:  streamlit run app.py"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

import config
from config import NOMES_CATEGORIAS, RELATORIOS
from pipeline import banco
from pipeline.extrator_llm import ESQUEMAS, provedor_ativo
from pipeline.processar import processar, reextrair
from pipeline.validacao import formatar_valor, parse_data, parse_valor

st.set_page_config(page_title="Leitor de documentos", page_icon="📄", layout="wide")

st.markdown("""
<style>
  .block-container {max-width: 1150px; padding-top: 2rem;}
  .etapa {border-left: 3px solid #2B3A8C; padding: .15rem 0 .15rem .8rem; margin: 1.2rem 0 .4rem 0;
          font-weight: 600; color: #2B3A8C;}
  .aviso-ia {font-size: .85rem; color: #5b6170; border: 1px dashed #c8ccd6; border-radius: 6px; padding: .5rem .75rem;}
  .tipo {font-size: 1.6rem; font-weight: 700; line-height: 1.1;}
</style>
""", unsafe_allow_html=True)

PASTA_DEMO = Path(__file__).parent / "demo" / "documentos"
ICONE = {"ok": "✅", "erro": "❌", "aviso": "⚠️"}


def nome_tipo(t):
    return NOMES_CATEGORIAS.get(t, t)


def formatar_campo(k, v):
    if v is None or v == "":
        return "não encontrado"
    if k.startswith("valor"):
        return formatar_valor(parse_valor(v))
    return str(v)


def tabela_campos(campos: dict, validacoes: list[dict]):
    por_campo = {}
    for v in validacoes or []:
        por_campo.setdefault(v["campo"], []).append(v)
    linhas = []
    for k, v in (campos or {}).items():
        vs = por_campo.get(k, [])
        status = "❌" if any(x["nivel"] == "erro" for x in vs) else ("✅" if vs else ("—" if v is None else ""))
        linhas.append({"Campo": k.replace("_", " "), "Valor": formatar_campo(k, v), "Checagem": status,
                       "Detalhe": "; ".join(x["mensagem"] for x in vs)})
    st.dataframe(pd.DataFrame(linhas), hide_index=True, width="stretch")


# ============================================================================ páginas

def pagina_enviar():
    st.title("Enviar documento")
    st.caption("PDF ou foto. O sistema descobre o tipo, extrai os dados e organiza tudo.")

    col1, col2 = st.columns([3, 2])
    with col1:
        arquivo = st.file_uploader("Arquivo", type=["pdf", "png", "jpg", "jpeg", "webp"], label_visibility="collapsed")
    with col2:
        exemplos = sorted(p.name for p in PASTA_DEMO.glob("*")) if PASTA_DEMO.exists() else []
        exemplo = st.selectbox("ou use um documento de exemplo", ["—"] + exemplos)

    conteudo, nome = None, None
    if arquivo is not None:
        conteudo, nome = arquivo.getvalue(), arquivo.name
    elif exemplo != "—":
        conteudo, nome = (PASTA_DEMO / exemplo).read_bytes(), exemplo

    if conteudo and st.button("Processar documento", type="primary"):
        with st.spinner("Lendo, classificando e extraindo…"):
            st.session_state["ultimo"] = (processar(conteudo, nome), conteudo, nome)

    if "ultimo" not in st.session_state:
        return
    r, conteudo, nome = st.session_state["ultimo"]

    if r.duplicado_de:
        st.info(f"Este mesmo arquivo já tinha sido enviado (documento #{r.duplicado_de}).")

    # ---- resultado em destaque
    st.divider()
    a, b = st.columns([2, 3])
    with a:
        st.markdown(f"<div class='tipo'>{nome_tipo(r.classificacao.tipo)}</div>", unsafe_allow_html=True)
        st.progress(min(r.classificacao.confianca, 1.0), text=f"Confiança do classificador: {r.classificacao.confianca:.0%}")
        if r.status == "salvo":
            st.success(f"Salvo no banco (documento #{r.id}).")
        else:
            if r.classificacao.precisa_revisao:
                st.warning("Não tenho certeza do tipo, dá uma revisada. Mandei para a fila de revisão.")
            else:
                st.warning("Encontrei algo estranho, dá uma revisada. Mandei para a fila de revisão.")
            for m in r.motivos:
                st.markdown(f"- {m}")
    with b:
        if r.extracao and r.extracao.resumo:
            st.markdown(f"**Resumo:** {r.extracao.resumo}")
        if r.extracao:
            tabela_campos(r.extracao.campos, r.validacoes)
        st.markdown("<div class='aviso-ia'>Resultado gerado por IA (classificador treinado pelo grupo + LLM). "
                    "Pode conter erros: confira valores e datas antes de pagar ou assinar qualquer coisa.</div>",
                    unsafe_allow_html=True)

    # ---- etapas do pipeline (transparência)
    st.markdown("<div class='etapa'>Como o sistema chegou nisso</div>", unsafe_allow_html=True)
    e1, e2, e3, e4 = st.tabs(["1 · Leitura (OCR)", "2 · Classificação (ML)", "3 · O que foi para o LLM", "4 · Resposta do LLM"])
    with e1:
        c1, c2 = st.columns(2)
        with c1:
            st.write(f"Método: **{'texto do PDF' if r.ocr.metodo == 'pdf_texto' else 'OCR (Tesseract)'}**")
            if r.ocr.metodo == "ocr":
                st.write(f"Qualidade estimada: **{r.ocr.qualidade:.0f}/100**")
            if nome.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                st.image(conteudo, caption="original", width="stretch")
        with c2:
            st.text_area("Texto lido", r.ocr.texto, height=360, disabled=True)
    with e2:
        st.write(f"Modelo: **{r.classificacao.modelo}**  ·  limiar para revisão: {config.LIMIAR_CONFIANCA:.0%}")
        probs = pd.DataFrame({"tipo": [nome_tipo(k) for k in r.classificacao.probabilidades],
                              "probabilidade": list(r.classificacao.probabilidades.values())}).set_index("tipo")
        st.bar_chart(probs, horizontal=True)
    with e3:
        if r.texto_mascarado:
            st.write("Dados pessoais trocados por marcadores antes do envio (o valor real nunca sai da máquina): "
                     + (", ".join(f"{v} {k.lower()}" for k, v in r.itens_mascarados.items()) or "nenhum encontrado"))
            st.text_area("Texto enviado", r.texto_mascarado, height=320, disabled=True)
        else:
            st.write("Nada foi enviado ao LLM: o documento foi direto para a revisão humana.")
    with e4:
        if r.extracao:
            st.write(f"Motor: **{r.extracao.motor}**")
            if r.extracao.observacoes:
                st.write(f"Observações: {r.extracao.observacoes}")
            st.write(f"O LLM concorda com o tipo? **{'sim' if r.extracao.tipo_confere else 'não'}**")
            if r.extracao.bruto:
                st.code(r.extracao.bruto, language="json")
        else:
            st.write("Sem extração.")


def pagina_revisao():
    st.title("Revisão humana")
    df = banco.listar(["revisao"])
    if df.empty:
        st.success("Nada esperando revisão.")
        return
    st.caption(f"{len(df)} documento(s) aguardando. A pessoa decide; o sistema só sugere.")
    for _, d in df.iterrows():
        with st.expander(f"#{d['id']} · {d['arquivo']} · sugerido: {nome_tipo(d['tipo_previsto'])} ({d['confianca']:.0%})", expanded=len(df) == 1):
            for m in (d["motivo_revisao"] or "").split(" | "):
                if m:
                    st.markdown(f"- {m}")
            tipos = list(NOMES_CATEGORIAS)
            tipo = st.selectbox("Tipo correto", tipos, index=tipos.index(d["tipo_final"]), format_func=nome_tipo, key=f"t{d['id']}")
            if tipo != d["tipo_final"] or not d["campos"]:
                if st.button("Extrair com este tipo", key=f"x{d['id']}"):
                    with st.spinner("Extraindo…"):
                        reextrair(int(d["id"]), tipo)
                    st.rerun()
            campos = dict(d["campos"] or {}) or {k: None for k in ESQUEMAS[tipo]}
            if d["validacoes"]:
                for v in d["validacoes"]:
                    if v["nivel"] == "erro":
                        st.markdown(f"{ICONE['erro']} **{v['campo']}**: {v['mensagem']}")
            novos = {}
            cols = st.columns(2)
            for i, (k, v) in enumerate(campos.items()):
                novos[k] = cols[i % 2].text_input(k.replace("_", " "), "" if v is None else str(v), key=f"c{d['id']}{k}") or None
            with st.expander("Texto do documento"):
                st.text(d["texto_ocr"] or "")
            a, b, _ = st.columns([1, 1, 3])
            if a.button("Aprovar e salvar", type="primary", key=f"a{d['id']}"):
                for k in novos:
                    if k.startswith("valor") and novos[k]:
                        novos[k] = parse_valor(novos[k])
                banco.atualizar_revisao(int(d["id"]), tipo, novos, "revisado")
                st.rerun()
            if b.button("Descartar", key=f"d{d['id']}"):
                banco.atualizar_revisao(int(d["id"]), tipo, novos, "descartado")
                st.rerun()


def pagina_painel():
    st.title("Painel")
    df = banco.listar(["salvo", "revisado"])
    pend = len(banco.listar(["revisao"]))
    hoje = date.today()

    notas = df[df.tipo_final == "nota_fiscal"]
    gasto_mes = sum(parse_valor(c.get("valor_total")) or 0 for c in notas.campos
                    if (dt := parse_data(c.get("data_emissao"))) and dt.month == hoje.month and dt.year == hoje.year)
    boletos = df[df.tipo_final == "boleto"]
    a_pagar = sum(parse_valor(c.get("valor")) or 0 for c in boletos.campos if (v := parse_data(c.get("vencimento"))) and v >= hoje)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Documentos organizados", len(df))
    m2.metric("Esperando revisão", pend)
    m3.metric("Gasto em notas neste mês", formatar_valor(gasto_mes))
    m4.metric("Boletos a pagar", formatar_valor(a_pagar))

    st.subheader("Alertas")
    al = banco.alertas()
    if not al:
        st.write("Nenhum alerta. Boletos que vencem em até "
                 f"{config.DIAS_ALERTA_BOLETO} dias e contratos que terminam em até {config.DIAS_ALERTA_CONTRATO} dias aparecem aqui.")
    for x in al:
        texto = f"{x['texto']} ({x['data']:%d/%m/%Y})" + (f" · {formatar_valor(x['valor'])}" if x["valor"] else "")
        {"erro": st.error, "aviso": st.warning, "info": st.info}[x["nivel"]](texto)

    if df.empty:
        st.info("Envie documentos na página Enviar documento para preencher o painel.")
        return

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Por tipo")
        st.bar_chart(df.tipo_final.map(nome_tipo).value_counts())
    with c2:
        st.subheader("Notas por categoria de gasto")
        if len(notas):
            cat = pd.Series([c.get("categoria_gasto") or "sem categoria" for c in notas.campos])
            val = pd.Series([parse_valor(c.get("valor_total")) or 0 for c in notas.campos])
            st.bar_chart(val.groupby(cat).sum())
        else:
            st.write("Ainda sem notas fiscais.")

    st.subheader("Documentos")
    tabela = pd.DataFrame({
        "#": df.id, "arquivo": df.arquivo, "tipo": df.tipo_final.map(nome_tipo),
        "confiança": (df.confianca * 100).round(0).astype(int).astype(str) + "%", "status": df.status, "resumo": df.resumo,
        "dados": [json.dumps({k: v for k, v in c.items() if v is not None}, ensure_ascii=False) for c in df.campos],
    })
    st.dataframe(tabela, hide_index=True, width="stretch")
    a, b = st.columns([1, 2])
    a.download_button("Exportar CSV", tabela.to_csv(index=False).encode("utf-8-sig"), "documentos.csv", "text/csv")
    with b:
        alvo = st.selectbox("Excluir documento (apaga arquivo de texto e dados)", ["—"] + list(df.id))
        if alvo != "—" and st.button("Excluir definitivamente"):
            banco.excluir(int(alvo))
            st.rerun()


def pagina_modelo():
    st.title("Modelo de classificação")
    caminho = RELATORIOS / "metricas.json"
    if not caminho.exists():
        st.warning("Rode python ml/treinar.py para gerar as métricas.")
        return
    m = json.load(open(caminho, encoding="utf-8"))
    st.write(f"Modelo em uso: **{m['escolhido']}**. Treino com {m['n_treino']} documentos, teste com {m['n_teste']}, "
             f"e um teste extra de robustez com {m['n_robustez']} documentos com OCR muito ruim e cortados.")
    linhas = [{"Abordagem": k, "Acurácia": f"{v['acuracia']:.3f}", "F1 macro": f"{v['f1_macro']:.3f}",
               "F1 validação cruzada": "—" if v["f1_cv_media"] is None else f"{v['f1_cv_media']:.3f} ± {v['f1_cv_desvio']:.3f}",
               "F1 com OCR ruim": f"{v['f1_ocr_ruim']:.3f}"} for k, v in m["resultados"].items()]
    st.dataframe(pd.DataFrame(linhas), hide_index=True, width="stretch")
    for img in ("comparacao_modelos.png", "matriz_confusao.png", "avaliacao_extracao.png"):
        if (RELATORIOS / img).exists():
            st.image(str(RELATORIOS / img), width="stretch")
    st.caption("A base é sintética (documentos fictícios em pt-BR). O número que importa é o do teste com OCR ruim; "
               "o 100% no texto limpo mostra que a base sintética é mais fácil que a vida real.")


def pagina_etica():
    st.title("Ética, privacidade e limitações")
    st.markdown(f"""
**Privacidade e LGPD.** Antes de qualquer texto ir para a API do LLM, CPF, nomes de pessoas, e-mail, telefone,
CEP e endereço viram marcadores (ex.: `[PESSOA_1]`). A resposta volta com os marcadores e o dado real é recolocado
aqui, na máquina. Dá para ver exatamente o que foi enviado na aba *O que foi para o LLM*. Mascaramento ligado:
**{'sim' if config.MASCARAR_DADOS else 'não'}**. Para não enviar nada a terceiros, troque para um modelo aberto
local (`TDE2_LLM=ollama`).

**Segurança dos dados.** Tudo fica num banco SQLite local. Qualquer documento pode ser excluído de forma
definitiva no painel (direito de exclusão). O sistema não guarda o arquivo original, só o texto lido.

**Erro que custa caro.** Valor ou data errados podem gerar prejuízo. Por isso: (1) se o classificador tem menos
de {config.LIMIAR_CONFIANCA:.0%} de confiança, nada é extraído e uma pessoa escolhe o tipo; (2) tudo que dá para
checar com regra é checado (CNPJ, linha digitável, chave da NF-e, se o valor aparece mesmo no texto); (3) o LLM
dá uma segunda opinião sobre o tipo. Na dúvida, vai para revisão humana.

**Transparência.** Todo resultado mostra a confiança, o texto lido e a resposta do LLM, com aviso de que foi
gerado por IA.

**Vieses e limitações.** O classificador foi treinado com documentos fictícios gerados por nós; documentos
reais têm layouts que ele nunca viu. Ele vai pior com manuscrito, foto escura ou torta e formatos fora das 5
categorias, e pode errar *com confiança alta* (ex.: fatura de cartão). O OCR é o elo mais fraco: se ele lê mal,
tudo depois sofre. O LLM pode inventar campos; a checagem de evidência reduz, mas não elimina, esse risco.
""")
    st.write(f"LLM em uso agora: **{provedor_ativo()}**")


# ============================================================================ navegação

PAGINAS = {"Enviar documento": pagina_enviar, "Revisão": pagina_revisao, "Painel": pagina_painel,
           "Modelo": pagina_modelo, "Ética e limitações": pagina_etica}
with st.sidebar:
    st.markdown("### Leitor de documentos")
    pend = len(banco.listar(["revisao"]))
    escolha = st.radio("Navegação", list(PAGINAS), label_visibility="collapsed",
                       format_func=lambda p: f"{p} ({pend})" if p == "Revisão" and pend else p)
    st.divider()
    prov = provedor_ativo()
    if prov == "sem_llm":
        st.warning("LLM não configurado: usando só regex. Defina ANTHROPIC_API_KEY.")
    else:
        st.caption(f"LLM: {prov}")
    if st.button("Limpar banco (demo)"):
        banco.limpar()
        st.session_state.pop("ultimo", None)
        st.rerun()
PAGINAS[escolha]()
