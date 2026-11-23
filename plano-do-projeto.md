# Predição personalizada da dose de insulina na refeição (diabetes tipo 1)

- **Disciplina:** Machine Learning for Healthcare (Technion)
- **Entrega:** quarta-feira, 25/11/2026
- **Registro iniciado em:** 22/11/2026

## Objetivo

- **Objetivo final:** prever quantas unidades de insulina o paciente deve tomar a partir de três coisas: o perfil dele, como o corpo dele reage à insulina e o que ele comeu e fez nas últimas horas.
- **Formulação decidida:** o modelo é **personalizado**.
  - Um paciente só entra depois de termos **um histórico dele**; não queremos prever para alguém "do nada".
  - O "como o corpo reage à insulina" é aprendido com as refeições anteriores do próprio paciente.
- **Entrega de quarta:** uma regressão linear da dose de bolus por refeição. Depois vem uma regressão logística do risco de hipoglicemia.

---

## 1. Bases avaliadas (22/11/2026)

Cada base foi baixada e testada; nenhuma foi avaliada só pela descrição. Os testes foram:

- **Coerência fisiológica:** a glicose sobe depois de comer e cai depois da insulina?
- **Refeições válidas:** refeições com bolus, janela "limpa" de 3 h e desfecho "bom".
- **Equilíbrio entre pacientes:** quanto do total vem do maior paciente.

| Base | Onde está | Refeições válidas | Pacientes | Maior paciente | Veredito |
|---|---|---|---|---|---|
| GlucoBench | Kaggle | — | 10 | — | ❌ Dados sintéticos incoerentes: a glicose não responde a refeições nem a insulina |
| UCI Diabetes (Kahn), cópia | Kaggle | — | 70 | — | ❌ Coluna `value` corrompida |
| UCI Diabetes (Kahn), original | UCI | — | 70 | — | ⚠️ Real, mas de 1991, sem carboidratos em gramas e sem perfil |
| DiaData | Kaggle | — | — | — | ❌ Quase só CGM |
| BrisT1D, competição | Kaggle | — | — | — | 🔒 Exige login e aceite das regras |
| HUPA-UCM | Mendeley | 607 | 19 | 44% | ⚠️ Perfil rico (inclui peso), mas desequilibrada e com unidades de carboidrato misturadas |
| T1D-UOM | Zenodo | 201 | 8 | 48% | ❌ Refeições digitadas à mão em hora cheia; muitas lacunas de CGM |
| BrisT1D-Open | data.bris | 3.516 | 19 | 12% | ⚠️ Volume grande e tem FC e passos, mas o bolus precisa ser **inferido** (a base soma basal e bolus) |
| **AZT1D** | **Mendeley** | **907** | **24** | **8%** | ✅ **Escolhida:** bolus explícito na bomba, CGM com 99% de cobertura, dados equilibrados |

O número do AZT1D é depois da limpeza do notebook 01; o das outras bases vem dos testes rápidos. Os dados das outras bases continuam em `data/raw/`.

- **HUPA-UCM** pode servir de **validação externa** (treinar no AZT1D e testar nos pacientes de Madri, com as features em comum).
- **BrisT1D** combina melhor com a fase 2 e com a regressão da glicose.

---

## 2. Base escolhida: AZT1D

- **Fonte:** https://data.mendeley.com/datasets/gk9m674wcx/1 (DOI 10.17632/gk9m674wcx.1, licença CC BY 4.0)
- **Artigo:** Khamesian et al., *AZT1D: A Real-World Dataset for Type 1 Diabetes*, arXiv:2506.14789 (2025)
- **População:** 25 adultos com diabetes tipo 1 da Mayo Clinic Arizona.
  - Todos usam sistema automático de insulina: bomba Tandem t:slim X2 com Control-IQ e sensor Dexcom G6.
  - Acompanhamento: mediana de 44 dias (de 28 a 49). Cobertura do sensor: mediana de 99%.
- **Perfil** (Tabela I do artigo): HbA1c 6,6 ± 0,7%, idade 59 ± 15 anos (de 27 a 80), 13 mulheres e 12 homens.
  - **Não tem** peso, razão insulina/carboidrato nem fator de sensibilidade.
- **Fuso:** horário local do Arizona, que não tem horário de verão.

### Colunas

| Coluna | Unidade | Significado |
|---|---|---|
| `EventDateTime` | horário local | uma linha de CGM a cada ~5 min e uma linha por bolus |
| `CGM` | mg/dL | glicose do sensor; o Dexcom limita a faixa entre 40 e 400 |
| `Basal` | U/h | basal programada, extraída de PDFs por OCR (tem erros) |
| `BolusType` | — | `Standard`, `BLE Standard Bolus` (pelo app), `Automatic Bolus/Correction` (Control-IQ), `Extended …`, `Quick` |
| `TotalBolusInsulinDelivered` | U | bolus aplicado |
| `FoodDelivered` / `CorrectionDelivered` | U | parte do bolus para a comida / parte de correção |
| `CarbSize` | g | carboidratos digitados na calculadora de bolus |
| `DeviceMode` | — | `sleep` ou `exercise`; vazio = normal |

### Problemas tratados no notebook 01

1. **5.189 linhas duplicadas exatas (1,7%).** O mesmo bolus aparecia até 6 vezes no mesmo minuto, e isso inflava as contagens preliminares. Foram removidas.
2. O paciente 14 usa o cabeçalho `Readings (CGM / BGM)` no lugar de `CGM`.
3. O paciente 8 usa `'0'` como marcador de vazio em `BolusType` e `DeviceMode`, e um paciente tem o modo escrito `sleepsleep`. Ambos foram recodificados.
4. **Basal com erros de OCR:** 6.176 linhas acima de 10 U/h, sobretudo nos pacientes 10 e 23 (por exemplo, `825` no lugar de `0,825`). Viraram faltantes.
   - Ao todo, a basal fica faltante em ~35% das refeições. No notebook 02, será imputada com a mediana **do treino**.
5. O sensor censura a glicose em 40 e em 400 mg/dL (151 e 50 leituras). Os valores foram mantidos.
6. Em 16% dos bolus, total ≠ comida + correção. Quase todos são bolus **sem** carboidrato, digitados direto em unidades, sem a calculadora.
7. **Refeições divididas:** 4.254 entradas de carboidrato viraram 3.743 episódios. Entradas a menos de 30 min uma da outra são a mesma refeição (396 junções).
8. O modo exercício/sono só foi exportado para 20 dos 25 pacientes (faltam 1, 5, 7, 11 e 22), e só 43 refeições têm exercício nas 2 h anteriores. É uma feature fraca.
9. O paciente 23 não tem nenhum carboidrato registrado, então ficam **24 pacientes**.
10. Episódios com bolus estendido (63) foram excluídos, porque a insulina deles é distribuída por horas.
11. **Fator de confusão:** o Control-IQ ajusta a basal e dá bolus automáticos sozinho. Os bolus automáticos **nunca** entram no rótulo.

---

## 3. O que a base tem e o que falta

| Necessidade do objetivo final | Cobertura no AZT1D |
|---|---|
| Perfil do paciente | ⚠️ Idade, sexo e HbA1c. Sem peso. |
| Como o corpo reage à insulina | ✅ Estimado pelo **histórico** do paciente (razão insulina/carboidrato das refeições anteriores) |
| Alimentação | ✅ Carboidratos em gramas, estimados pelo paciente. ❌ Sem gordura nem proteína. |
| Insulina ativa | ✅ Bolus manuais e automáticos com horário exato |
| Atividade física | ⚠️ Só o modo exercício da bomba. ❌ Sem passos nem FC. |
| Sono | ⚠️ Só o modo sono da bomba |
| Dose "correta" | ❌ Não existe. Usamos as **refeições que deram certo** como aproximação. |

---

## 4. Formulação (decidida em 22/11/2026)

### Unidade de análise, target e features

- **Unidade de análise:** o episódio de refeição de um paciente, do primeiro registro de carboidrato ($t_0$) ao último ($t_1$).
- **Target `dose_u`:** soma dos bolus **manuais** entre 30 min antes e 30 min depois do episódio. Cada bolus vai para o episódio mais próximo.
  - Também guardamos as partes comida (`dose_food_u`) e correção (`dose_corr_u`).
- **Features:** só informação disponível até $t_0$:
  - carboidratos da refeição; glicose atual, tendência e média das 3 h anteriores;
  - insulina ativa (IOB, decaimento linear em 4 h) e carboidratos ativos (COB, decaimento linear em 3 h);
  - tempo desde a refeição anterior; insulina automática nas 2 h anteriores; basal;
  - hora do dia (seno e cosseno, mais dummies de período); modo exercício/sono;
  - perfil (idade, sexo, HbA1c);
  - **histórico pessoal** (`hist_ratio`, `hist_ratio_good`, `hist_n_meals`), calculado só com refeições anteriores.

### Linhas de treino

- **Principal:** refeições **limpas e boas**.
  - **Limpa:** sem comida e sem **nenhum** bolus entre 30 min e 3 h depois.
  - **Boa:** sem glicose < 70 e com glicose entre 70 e 180 em 3 h.
  - São **907 refeições**, de 24 pacientes; o maior tem 8%.
- **Sensibilidade:** `clean_manual`, que permite bolus automáticos depois da refeição. São 1.371 refeições.
- **Regra anti-vazamento:** o que acontece depois de $t_0$ só serve para escolher linhas ou como rótulo. Nunca vira feature.

### Por que personalizado: experimento preliminar

O experimento rodou antes da limpeza de duplicatas. O notebook 02 confirmou o padrão (ver seção 5).

| Cenário | MAE | Conclusão |
|---|---|---|
| **Paciente novo**, só com idade, sexo e HbA1c | ~3 U | Praticamente igual a prever sempre a média, com dose mediana de ~4 U. Não funciona. |
| **Paciente conhecido**, com a razão pessoal do histórico | 0,8 U com 5 refeições de histórico; ~0,7 U a partir de 10 a 20 | Funciona e estabiliza rápido. |

### Ressalva para a apresentação

Na bomba, a dose sai da calculadora (`carbs/razão + correção`). O modelo personalizado, em parte, **reaprende a configuração da bomba** de cada paciente a partir dos dados.

Ainda assim, isso tem valor:

- é útil para quem usa caneta e não tem calculadora;
- o filtro de refeições boas mostra quando essa configuração funcionou;
- dá para medir o quanto o contexto (tendência da glicose, insulina ativa, exercício) acrescenta à fórmula.

---

## 5. Notebooks

O padrão segue os tutoriais do curso (`class-2/tutorials/t2/t2_linreg.ipynb` e `class-3/t3_logreg_hf.ipynb`): textos em inglês, seções numeradas, LaTeX, regras anti-vazamento explícitas e a mesma paleta de gráficos.

### `01_preprocessing.ipynb` ✅ pronto

O notebook percorre as seguintes seções:

1. Formulação do problema.
2. Dataset e download automático.
3. Variáveis brutas.
4. Integridade dos dados: duplicatas, marcadores de vazio, OCR, sensor e cobertura.
5. Perfil do paciente.
6. Fluxos de eventos e episódios de refeição, com um dia de exemplo.
7. Desfechos e janela limpa, com as curvas médias por desfecho.
8. Features.
9. Funil até os candidatos a treino.
10. Salvamento da tabela.

- **Saída:** `data/processed/azt1d_meal_episodes.csv`, com 3.743 episódios × 50 colunas: features, rótulos, desfechos e flags.
- **Funil:** 4.254 entradas de carboidrato → 3.743 episódios → 3.679 com dose utilizável → 3.547 com CGM completo → 1.099 limpos → **907 bons (candidatos a treino)**.

### `02_linear_regression.ipynb` ✅ pronto

#### Configuração

- **Linhas:** 840 refeições (candidatos a treino com pelo menos 10 refeições anteriores), de 24 pacientes.
- **Divisão:** temporal dentro de cada paciente, com os primeiros 70% no treino (582) e os últimos 30% no teste (258).
- **Escolhas feitas só no treino:** janela de histórico, α do Ridge, imputação, padronização e peso igual por paciente.

#### Ideia central

A razão pessoal `Σdose/Σcarbs` **é uma regressão linear**: sem intercepto, por paciente, ajustada por mínimos quadrados ponderados com peso $1/c$. O notebook mostra a derivação em uma linha.

#### Resultados no teste (MAE por paciente)

| Modelo | MAE | Comentário |
|---|---|---|
| B0: dose média do paciente | 2,26 U | |
| M1: linear populacional (contexto + perfil, sem histórico) | 2,46 U | |
| **M2: razão pessoal (regressão pela origem)** | **0,80 U** (IC 0,44–1,26) | R² 0,87; **81% das refeições com erro de até ±1 U**; melhor que o M1 para 23 de 24 pacientes (p < 0,001) |
| M3: linear personalizado (contexto + razão pessoal) | OLS 0,84 / Ridge 0,80 U | Empata com o M2 (melhor para 10 de 24 pacientes, p = 0,99) |

#### Interpretação

- **Coeficientes do M3-OLS:** só `x_food` é significativo, com β = 0,91 (0,87–0,96); **o modelo redescobre a calculadora de bolus**. Glicose, tendência, insulina ativa, horário, exercício e perfil não são significativos.
- **Paciente novo:** o modelo populacional fica em 3–4 U mesmo com 16 a 22 pacientes no treino. Mais pacientes não substituem o histórico.
- **Deriva da razão:** o paciente 14 reduziu a razão pela metade no meio do estudo (0,5 → 0,2 U/g).
  - Pós-hoc, uma janela das últimas 20 a 40 refeições daria ~0,65 U no teste.
  - Mas a escolha feita no treino foi "todo o histórico": no começo do acompanhamento os históricos são curtos e todas as janelas parecem iguais.
  - Isso fica registrado como hipótese, não como resultado.
- **Sensibilidade:** o M2 fica entre 0,77 e 0,80 U nas variações de limpeza e de histórico mínimo.

### `03_logistic_regression.ipynb` (depois da entrega)

Prever `hypo_3h` (hipoglicemia em até 3 h) a partir do contexto e da dose escolhida. A ideia é que o modelo A sugere a dose e o modelo B checa se ela é segura.

---

## 6. Fase 2 (depois de quarta)

- **Paciente novo:** juntar bases (AZT1D + HUPA-UCM + BrisT1D, ~60 pacientes) com as features em comum.
- **Dados sintéticos fisiologicamente coerentes:**
  - **Simulador:** `simglucose`, uma implementação do simulador UVA/Padova de 2008, já instalado no `.venv`. Ele precisa de `setuptools<70`.
  - **Novos pacientes virtuais:** criados perturbando ou interpolando os parâmetros dos 30 pacientes que o simulador já traz, o que preserva as correlações fisiológicas.
  - **Rotinas:** calibradas nas distribuições reais de refeições.
  - **Rótulo de "dose ideal":** a dose que minimiza o índice de risco glicêmico (LBGI/HBGI) nas 4 a 5 h seguintes.
  - **Avaliação:** treinar com dados sintéticos e **testar só em pacientes reais**.
  - **O que evitar:** inventar perfis e calcular a dose com uma fórmula nossa. É circular, e a regressão só redescobre a fórmula.
  - **Limitações:** o simulador não modela exercício nem tem FC ou passos.
- **Validação externa:** treinar no AZT1D e testar no HUPA-UCM.
- **Target alternativo:** a dose corrigida pelo desfecho, $\text{dose} + (G_{3h} - 120)/\text{ISF}$, com o ISF estimado por paciente.

---

## 7. Ambiente e pastas

```
diabetes-project/
├── .venv/                          # Python 3.12: pandas 3, scikit-learn, statsmodels, jupyter, simglucose
├── 01_preprocessing.ipynb          # notebook 1 (executado)
├── 02_linear_regression.ipynb      # notebook 2 (executado)
├── plano-do-projeto.md
└── data/
    ├── processed/azt1d_meal_episodes.csv
    └── raw/
        ├── azt1d/                  # base principal; AZT1D_2025.zip (776 MB) pode ser apagado, os CSVs já estão extraídos
        ├── hupa_ucm/               # validação externa / fase 2
        ├── brist1d_open/           # fase 2
        ├── t1d_uom/  uci_kahn_original/
        └── kaggle_*                # mantidos só para documentar os problemas
```

- Ativar o ambiente: `source .venv/bin/activate`
- Reexecutar o notebook: `jupyter nbconvert --to notebook --execute --inplace 01_preprocessing.ipynb` (cerca de 25 s)

---

## 8. Referências

- Khamesian S. et al. *AZT1D: A Real-World Dataset for Type 1 Diabetes*. arXiv:2506.14789 (2025). Dados: https://doi.org/10.17632/gk9m674wcx.1
- Battelino T. et al. *Clinical Targets for Continuous Glucose Monitoring Data Interpretation: Recommendations From the International Consensus on Time in Range*. Diabetes Care 42(8):1593–1603 (2019).
- Brown S. A. et al. *Six-Month Randomized, Multicenter Trial of Closed-Loop Control in Type 1 Diabetes*. N Engl J Med 381:1707–1717 (2019).
- Hidalgo J. I. et al. *HUPA-UCM diabetes dataset*. Data in Brief 55:110559 (2024).
- Gordon James S. et al. *BrisT1D Dataset: Young Adults with Type 1 Diabetes in the UK using Smartwatches*. arXiv:2507.17757 (2025).
- simglucose: https://github.com/jxx123/simglucose
