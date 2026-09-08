# Manutenção Preditiva — RUL de Motores Turbofan (CMAPSS)

Previsão do tempo de vida útil restante (RUL — *Remaining Useful Life*) de
motores de avião a partir de leituras de sensores, usando XGBoost e uma
avaliação orientada a custo de negócio.

## Contexto

Motores de avião degradam-se ao longo da sua vida útil até falharem. Prever
quantos ciclos de operação faltam até essa falha (RUL) permite planear
manutenção preventiva em vez de reativa — evitando tanto falhas inesperadas
como manutenção desnecessária a motores ainda saudáveis.

## Dataset

[NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation)](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps),
subset FD001: 100 motores simulados, cada um correndo até falhar, com 21
sensores e 3 condições operacionais registados por ciclo.

## Abordagem

1. **Limpeza** — remoção de sensores com variância ~0 (sem informação útil)
2. **Target** — cálculo do RUL por motor, com *clipping* a 125 ciclos (RUL muito
   alto não é distinguível a partir dos sensores nos primeiros ciclos de vida)
3. **Baseline** — Random Forest com sensores brutos
4. **Feature engineering** — rolling mean/std (janela de 5 ciclos) por sensor,
   para capturar tendência e instabilidade de degradação
5. **Modelo final** — XGBoost, validado com `GroupKFold` (split por motor, não
   por linha — ver nota sobre data leakage abaixo)
6. **Avaliação de negócio** — métrica assimétrica que penaliza mais previsões
   otimistas (perigosas) do que conservadoras
7. **Dashboard** — Streamlit, upload de CSV → previsão de RUL

## Resultados

| Modelo | Validação | RMSE | MAE |
|---|---|---|---|
| Random Forest (baseline) | split simples | 18.77 | — |
| Random Forest + rolling features | split simples | 15.59 | — |
| **XGBoost + rolling features** | **GroupKFold (5 folds)** | **18.31** | — |

**Nota metodológica importante:** o RMSE de 15.59 foi obtido com
`train_test_split` simples, que mistura linhas do mesmo motor entre treino e
validação — isto é *data leakage* (o modelo "já viu" o motor, mesmo que em
ciclos diferentes), inflacionando artificialmente a performance. O resultado
correto, validado com `GroupKFold` (motores inteiros nunca vistos), é 18.31 —
um pouco pior, mas honesto.

### Avaliação 

Das previsões no fold de teste (4127 amostras, ~20 motores nunca vistos):

- **1766 (43%) previsões otimistas** (RUL a mais do que o real) — erro médio de 11.98 ciclos
- **2361 (57%) previsões conservadoras** (RUL a menos do que o real) — erro médio de 10.82 ciclos

**Implicação prática:** quase metade das previsões, se seguidas literalmente,
arriscariam manter um motor em operação além do seu tempo real de vida útil.
Recomenda-se aplicar uma margem de segurança de ~12 ciclos à previsão bruta do
modelo antes de a usar para decisões de manutenção.

## Dashboard

Interface Streamlit para carregar dados de sensores de um motor e visualizar
o RUL previsto ao longo do tempo, com aviso automático quando o RUL previsto
é baixo.

```bash
streamlit run app.py
```

## Como correr o projeto

```bash
git clone https://github.com/Vasco-Grilo/predictive_maintainance_cmapss.git
cd predictive_maintainance_cmapss
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Nota (Mac):** o XGBoost requer OpenMP instalado via Homebrew:
```bash
brew install libomp
```

Descarrega o dataset FD001 do [Kaggle](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps)
e coloca `train_FD001.txt`, `test_FD001.txt` e `RUL_FD001.txt` em `data/raw/`.

Corre o notebook `notebooks/exploracao_001.ipynb` para reproduzir a análise
completa, ou `streamlit run app.py` para o dashboard (requer `model.pkl` e
`feature_cols.pkl`, gerados no fim do notebook).

## Estrutura do repositório

```
├── data/
│   ├── raw/              # dados originais (não versionados)
│   └── processed/        # dados processados
├── notebooks/
│   └── exploracao_001.ipynb
├── app.py                 # dashboard Streamlit
├── model.pkl               # modelo XGBoost treinado
├── feature_cols.pkl        # lista de features usadas pelo modelo
├── requirements.txt
└── README.md
```

## Lessons learned

- Rolling features (tendência e instabilidade dos sensores) melhoraram
  significativamente o RMSE do baseline, mesmo sendo o CMAPSS um dataset
  simulado com transições relativamente suaves.
- Validar com split aleatório simples em dados de séries temporais agrupadas
  (múltiplas linhas por motor) inflaciona artificialmente a performance —
  `GroupKFold` é essencial para uma avaliação honesta.
- RMSE sozinho esconde informação de negócio relevante: decompor os erros por
  direção (otimista vs conservador) revelou que quase metade das previsões
  seriam perigosas se aplicadas sem margem de segurança.

## Licença

MIT
