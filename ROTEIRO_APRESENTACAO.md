# Roteiro da apresentação (≈15 min + perguntas)

## Slides

**1. O problema (1 min).** Notas, boletos, comprovantes e contratos chegam em PDF e foto, misturados. Organizar à mão é lento e
erro em valor ou data custa dinheiro. Tema 6 da lista: análise e classificação automática de documentos.

**2. A solução em uma frase + o fluxo (1 min).** Mostrar o diagrama:
Entrada → OCR → ML classifica → LLM extrai e resume → validação → banco, painel e alertas; confiança baixa → revisão humana.
Frase-chave: *o que o ML descobre define o que o LLM procura.*

**3. Por que ML e LLM com papéis diferentes (1 min).** O classificador é barato, rápido, local e mede a própria confiança.
O LLM entende layouts variados e texto com erro de OCR, onde regex quebra. Um não substitui o outro.

**4. Dados (1,5 min).** Não existe dataset público de documento brasileiro com licença clara e sem dado pessoal, então montamos
uma base própria: 2.000 documentos fictícios, 5 classes, CNPJ/linha digitável/chave da NF-e com dígito verificador válido.
A classe "outros" tem armadilhas de propósito: orçamento (parece nota), recibo (parece comprovante), aviso de cobrança (parece boleto).
Simulamos foto ruim: troca de letras, acento perdido, linha cortada, documento pela metade.

**5. Modelos e métricas (2 min).** Mostrar `relatorios/comparacao_modelos.png` e a tabela:

| Abordagem | Acurácia | F1 macro | F1 CV (5 dobras) | F1 com OCR ruim |
|---|---|---|---|---|
| Regras por palavra-chave | 0,870 | 0,870 | — | 0,390 |
| TF-IDF palavras + Regressão Logística | 1,000 | 1,000 | 0,999 | 0,904 |
| TF-IDF palavras + SVM | 1,000 | 1,000 | 1,000 | 0,944 |
| **TF-IDF caracteres + SVM (escolhido)** | 0,998 | 0,997 | 0,998 | **0,954** |

Dizer com todas as letras: *no texto limpo todo mundo acerta, porque a base sintética é mais fácil que a vida real. A diferença
aparece no teste de robustez, e é por isso que escolhemos n-gramas de caracteres: "N0ta F1scal" ainda compartilha pedaços com "nota fiscal".*
Mostrar a matriz de confusão (`matriz_confusao.png`).

**6. LLM (1,5 min).** Prompt por tipo com esquema de campos, saída só em JSON, `null` quando não acha, exemplo few-shot com erro de OCR,
segunda opinião sobre o tipo. Mostrar `avaliacao_extracao.png` (regex × LLM, texto limpo e OCR ruim). **Rodar antes:** `python ml/avaliar_extracao.py --n 20`.

**7. Validação: não confiar cegamente na IA (1 min).** Dígito verificador de CNPJ, linha digitável e chave da NF-e; valor e vencimento
do boleto conferidos contra o código de barras; checagem se o valor extraído aparece mesmo no texto (pega alucinação).

**8. DEMO AO VIVO (5 min).** Ver abaixo.

**9. Ética e responsabilidade (1,5 min).** Um slide:
- Privacidade/LGPD: mascaramento reversível antes da API; mostrar a aba "O que foi para o LLM". Alternativa 100% local com Ollama.
- Segurança: dados só no SQLite local; exclusão definitiva pelo painel; não guarda o arquivo original.
- Erro que custa caro: limiar de confiança, validações, revisão humana. O sistema sugere, a pessoa decide.
- Transparência: confiança, texto lido e resposta do LLM visíveis; aviso "gerado por IA".
- Vieses e limitações: treino sintético; pior com manuscrito, foto ruim e tipos fora das 5 classes; **pode errar com confiança alta** (exemplo da fatura).

**10. Limitações e próximos passos (0,5 min).** Testar com documentos reais anonimizados, calibrar a confiança com dados reais,
detecção de "fora da distribuição", bot de Telegram/WhatsApp como segunda entrada.

## Roteiro da demo (ensaiar pelo menos 2 vezes)

Antes: `python demo/gerar_documentos_demo.py` (datas relativas ao dia), botão **Limpar banco (demo)**, conferir que a barra lateral mostra o LLM ativo.

1. **01_nota_fiscal_mercado.pdf** → nota fiscal, ~100%, campos validados, categoria de gasto. Abrir a aba *O que foi para o LLM* e mostrar CPF e nome mascarados.
2. **05_nota_fiscal_foto.jpg** → mesmo fluxo por foto (OCR). Mostrar qualidade do OCR e o texto lido.
3. **02_boleto_escola.pdf** e **03_contrato_manutencao.pdf** → salvos sem conversa.
4. **07_boleto_adulterado.pdf** → o "golpe do boleto": o valor impresso (R$ 1.820) não bate com o código de barras (R$ 1.280). Vai para revisão.
5. **10_fatura_cartao.pdf** → o classificador erra com confiança alta, o LLM discorda e manda para revisão. Ótimo para falar de limitação.
6. **09_foto_muito_ruim.jpg** → "não tenho certeza" (≈48%): nada é extraído, vai para revisão.
7. Página **Revisão** → na fatura, escolher "Outros", *Extrair com este tipo*, *Aprovar*.
8. Página **Painel** → alertas (boleto vencendo em poucos dias, contrato terminando em 20 dias), gráficos, exportar CSV, excluir um documento.

Plano B: se a internet cair, mudar `TDE2_LLM=sem_llm` e explicar que a extração inteligente depende da API (ou ter o Ollama instalado). Grave um vídeo da demo funcionando como reserva.

## Perguntas prováveis da banca

- **"100% de acurácia não é suspeito?"** É, e por isso criamos o teste de robustez. Base sintética é fácil; o número honesto é 0,95 com OCR ruim, e o próximo passo é medir com documentos reais.
- **"Por que não deixar o LLM classificar tudo?"** Custo, latência e privacidade: o classificador roda local, de graça, e devolve uma confiança mensurável que usamos para decidir quando chamar uma pessoa. O LLM ainda dá a segunda opinião.
- **"E se o LLM inventar um valor?"** Pedimos `null` quando não acha, e conferimos se o valor aparece no texto e se bate com códigos com dígito verificador.
- **"Como garante a LGPD?"** Minimização: só o necessário vai para a API, com dados pessoais mascarados; armazenamento local; exclusão definitiva; opção 100% local.
- **"Por que SVM com caracteres?"** Melhor F1 no cenário difícil (0,954 contra 0,904 da regressão logística) e mais tolerante a erro de OCR.
