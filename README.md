# Análise da Qualidade de Vinhos (MLOps)
Portfólio focado em boas práticas básicas de MLOps, com **classificação binária**: **Alta qualidade** (≥6) e **Baixa qualidade** (<6).

## Propósito do Projeto

Este repositório tem o objetivo de demonstrar um fluxo completo de MLOps aplicado à predição da qualidade de vinhos a partir de medidas físico-químicas. O foco é entregar um processo reprodutível que inclui:

- Ingestão e tratamento dos dados brutos;
- Engenharia de atributos e seleção de features;
- Balanceamento de classes e comparação de modelos;
- Automação de treino, teste e empacotamento (API + app Streamlit);
- Boas práticas de reprodutibilidade (Docker, testes, CI).

O artefato principal é um pipeline treinado que pode ser servido via API (`/predict`) ou usado no app Streamlit para auxiliar decisões operacionais.

## Insights principais (para apresentação)

- **Importância de features**: Em análises comuns do dataset de vinho tinto, atributos como `alcohol`, `volatile_acidity`, `sulphates` e interações envolvendo `density` frequentemente aparecem como fortes preditores de qualidade. Nosso pipeline inclui seleção automática (Top 20) para concentrar sinal.
- **Balanceamento é crucial**: O conjunto original tende a ter distribuição desigual entre classes; técnicas como SMOTEENN/ADASYN melhoram desempenho em métricas ponderadas (F1 weighted) comparado a treinar sem balanceamento.
- **Comparação de algoritmos**: Testamos RandomForest, GradientBoosting e, quando disponíveis, XGBoost/LightGBM. Escolhemos o melhor pipeline por F1-weighted e retreinamos para produção.
- **Trade-offs operacionais**: Modelos com maior F1 tendem a ser mais complexos; para deploy em ambientes com restrição de recursos, RandomForest com menos estimators pode ser um bom compromisso.
- **Recomendações rápidas**: monitorar distribuição das features críticas (`alcohol`, `pH`, `volatile_acidity`), validar contagens por rótulo e automatizar alertas de drift.


## Visão Geral
- **Stacks:** pandas, scikit-learn, FastAPI e testes com pytest.
- **Pipeline reproduzível:** leitura do dado bruto, engenharia de atributos, balanceamento com SMOTE e modelo RandomForest.
- **Entrega pronta:** Dockerfile, API `/predict`, CI/CD via GitHub Actions e relatórios claros para stakeholders.

## Arquitetura
- `src/analise_qualidade_vinhos/config/settings.py` – caminhos, seeds e colunas alvo.
- `src/analise_qualidade_vinhos/features/engineering.py` – renomeia colunas, cria atributos e o alvo binário (2 classes).
- `src/analise_qualidade_vinhos/pipeline/model_builder.py` – pré-processador, balanceamento e modelo.
- `src/analise_qualidade_vinhos/pipeline/train.py` – treino + salvamento de métricas e artefatos.
- `src/analise_qualidade_vinhos/pipeline/predict.py` – preparação e inferência.
- `src/analise_qualidade_vinhos/api.py` – API FastAPI (health e predict).
- `tests/` – unidade e integração do pipeline.

## Como rodar localmente
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt

# Treino
python -m analise_qualidade_vinhos.pipeline.train

# Testes
pytest
```

### Subir com Docker Compose
```bash
# Subir API (web) e Streamlit juntos (builda as imagens se necessário)
docker compose up --build

# Ou subir em background (detached)
docker compose up --build -d
```

Observação: o arquivo `docker-compose.yml` deve estar na raiz do projeto. Os serviços padrão iniciam a API (`web`) em `:8000` e o app Streamlit em `:8501`.

Endpoints:
- `GET /health` → status
- `POST /predict` → envia lista de amostras com as 11 features originais (snake_case).

## Dados e Engenharia de Atributos
Fonte: `data/raw/winequality-red.csv` (UCI).
- Normalização de nomes para snake_case.
- Criação de interações simples (ex.: `density_alcohol_ratio`, `total_free_sulfur_ratio`, `acidity_index`).
- **Classificação binária**: ≥6 = Alta qualidade, <6 = Baixa qualidade (`quality_label`).
- Balanceamento com SMOTEENN/ADASYN/SMOTE antes do treino.
- Seleção de features (Top 20) para melhor performance.

## Parâmetros de produção — Limites e recomendações

Para auxiliar na interpretação dos atributos químicos do vinho e orientar controles de qualidade, abaixo estão os limites de segurança, faixas recomendadas para melhor qualidade e riscos quando fora dos limites:

| Parâmetro                 | Intervalo Aceitável (Segurança) | Faixa Recomendada (Qualidade) | Riscos se fora do limite                                        |
| ------------------------- | ------------------------------- | ----------------------------- | --------------------------------------------------------------- |
| **Acidez fixa (g/L)**     | 3.5 – 14.0                      | 4.0 – 10.0                    | Acidez baixa = vinho “mole”; acidez alta = agressivo ao paladar |
| **Acidez volátil (g/L)**  | **≤ 1.20 g/L** (legal)          | 0.30 – 0.90                   | Acima disso → cheiro de vinagre (ácido acético)                 |
| **Ácido cítrico (g/L)**   | 0.0 – 1.0                       | 0.2 – 0.5                     | Muito alto causa sabor artificial, muito baixo reduz frescor    |
| **Açúcar residual (g/L)** | 0.2 – 20.0                      | Secos: < 4.0                  | Açúcar excessivo favorece contaminações microbianas             |
| **Cloretos (g/L)**        | 0.01 – 0.60                     | 0.05 – 0.15                   | Acima gera gosto salgado e instabilidade microbiológica         |
| **SO₂ Livre (mg/L)**      | **0 – 50 mg/L**                 | 10 – 30                       | Baixo → oxidação; alto → alergias, irritações                   |
| **SO₂ Total (mg/L)**      | **0 – 150 mg/L** (legal)        | 30 – 100                      | Acima → risco toxicológico e sabor picante                      |
| **Densidade (g/cm³)**     | 0.990 – 1.010                   | 0.992 – 0.998                 | Fora → erros de fermentação ou adulteração                      |
| **pH**                    | **2.8 – 4.2**                   | 3.0 – 3.5                     | pH alto = risco microbiológico; pH baixo = acidez agressiva     |
| **Sulfatos (g/L)**        | 0.3 – 1.5                       | 0.5 – 1.0                     | Excesso → sensação metálica; pouco → baixa estabilidade         |
| **Álcool (%)**            | **8% – 16%** (mercado)          | 10% – 13%                     | Baixo → instável microbiologicamente; alto → caráter quente     |


## Métricas
- O sistema testa automaticamente múltiplos algoritmos (RandomForest, GradientBoosting, XGBoost, LightGBM).
- Seleciona o melhor modelo baseado em F1-score.
- Métricas salvas em `reports/metrics.json`.

## App Streamlit para Produção
Execute o app interativo para uso pelos funcionários:
```bash
streamlit run app.py
```
Interface completa com formulário, predições em tempo real e recomendações.
- Classes previstas: `Baixa qualidade` (<6) e `Alta qualidade` (≥6).

## CI/CD (GitHub Actions)
- Workflow em `.github/workflows/ci.yml`:
  - Instala dependências
  - Roda `pytest`
  - (opcional) build do Docker

## Entregáveis para stakeholders
- Relatório executivo: `reports/stakeholders.md`
- Métricas do modelo: `reports/metrics.json` (gerado no treino)
- API pronta para consumo ou uso via CLI.

## Próximos passos ao adquirir mais experiência.
- Adicionar monitoramento de drift e logging estruturado.
- Comparar modelos adicionais (XGBoost/LightGBM) se houver necessidade.
- Publicar imagem no GHCR e ativar CD para ambiente cloud.