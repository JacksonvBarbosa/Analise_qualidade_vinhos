# Análise da Qualidade de Vinhos
**Portfólio de Data Analytics | Jackson dos Santos Ventura**

## 📊 Projeto: Análise Exploratória de Vinhos Importados para JACKWine

### Contexto do Projeto
A distribuidora JACKWine está expandindo seu catálogo através da importação de vinhos portugueses. Como analista de dados da empresa, realizei uma análise exploratória dos dados, identificando os fatores químicos que influenciam a qualidade dos produtos. Além disso, desenvolvi modelos de machine learning para prever a pontuação de qualidade dos vinhos, fornecendo insights estratégicos para apoiar o processo de seleção e importação.

### Objetivo
Identificar relações entre os componentes químicos e a qualidade dos vinhos portugueses, visando compreender os principais fatores que influenciam sua avaliação e utilizar esses insights para apoiar decisões estratégicas e o desenvolvimento de modelos preditivos com machine learning.

## 🔬 Metodologia

### 1. Aquisição e Preparação dos Dados
**Dataset:** winequality-red.csv (Fonte: UCI Machine Learning Repository)

**Ferramentas:** Python, Pandas, NumPy, Matplotlib, Seaborn

**Procedimentos:**
- Importação e inspeção inicial do dataset
- Tratamento de valores duplicados e ausentes
- Ajuste de tipos de dados
- Criação de funções modulares para extração, transformação e armazenamento de dados no pacote `etl/`
- Implementação de tratamento de outliers e balanceamento de classes no pacote `features/`

### 2. Análise Exploratória de Dados (EDA)
- Visualização e análise de distribuições de variáveis químicas
- Identificação de correlações entre variáveis e qualidade do vinho
- Uso de gráficos de dispersão, boxplots, histogramas e mapas de calor
- Criação do módulo `visualization/` para centralizar funções gráficas reutilizáveis

### 3. Desenvolvimento de Modelos de Machine Learning
- Estrutura de código organizada em pacotes reutilizáveis (`models/`) para classificação, regressão e clustering
- Implementação de pipelines (`pipeline_classification.py`, `pipeline_regression.py`, `pipeline_clustering.py`) para padronizar o fluxo de treino e avaliação
- Utilização do `model_factory.py` com lazy loading, permitindo carregar modelos sob demanda e melhorar a escalabilidade do projeto
- Aplicação de técnicas de otimização de hiperparâmetros com RandomizedSearchCV
- Avaliação de modelos utilizando métricas como Acurácia, Precisão, Recall, F1-score e ROC AUC

### 4. Modularização e Escalabilidade
- Estrutura do projeto planejada para reuso e manutenção em diferentes datasets
- Separação de responsabilidades por pacotes:
  - `etl/` → Funções de extração, transformação e armazenamento
  - `features/` → Engenharia de variáveis e tratamento de dados
  - `models/` → Treinamento, avaliação e pipelines de ML
  - `visualization/` → Geração de gráficos e plots
- Suporte para inclusão de novos modelos no `model_factory.py` sem alteração no restante do código

### 5. Armazenamento e Versionamento de Modelos
- Modelos treinados salvos em `models_storage/` para reutilização futura
- Uso de joblib para serialização
- Versionamento do código via GitHub

## 📊 Análise Exploratória e Pré-Processamento — Qualidade de Vinhos

Este estudo tem como objetivo analisar o Wine Quality Dataset, obtido através do Kaggle, e aplicar técnicas de pré-processamento para preparar os dados para modelos de machine learning voltados à previsão da qualidade de vinhos.

### 1. Entendimento Inicial dos Dados
O dataset foi analisado em sua forma bruta (raw data), contendo atributos físico-químicos e a nota de qualidade do vinho.

A partir das estatísticas descritivas, identificamos:
- Baixa dispersão na maioria das variáveis, devido ao baixo desvio padrão
- Maior variabilidade no dióxido de enxofre (livre e total), com desvio padrão elevado — podendo apresentar valores muito acima ou abaixo da média

Esse comportamento pode comprometer a capacidade de generalização dos modelos, tornando necessária a padronização dos dados para equilibrar as escalas durante o treinamento e teste.

### 2. Dados Duplicados
Registros duplicados são comuns neste tipo de análise, pois amostras com propriedades físico-químicas semelhantes tendem a gerar notas de qualidade próximas ou iguais. Por isso, neste caso, a duplicidade não foi tratada como erro.

### 3. Distribuição e Outliers
Os gráficos de distribuição e boxplots mostraram:
- Ausência de distribuição normal em várias variáveis
- Presença de outliers detectados pelo IQR e z-score

Optou-se por manter os outliers detectados pelo IQR, aplicando capping para limitar valores extremos. Essa decisão reduz a dispersão e melhora o desempenho de algoritmos sensíveis à escala, como Logistic Regression e SVM.

### 4. Relações Entre Variáveis

#### 4.1 Acidez Volátil × Qualidade
Vinhos de alta qualidade tendem a ter menor acidez volátil, idealmente abaixo de 1,0 g/L. No entanto, a acidez não atua isoladamente e deve ser analisada junto a outros fatores.

#### 4.2 Teor Alcoólico × Qualidade
Os dados indicam relação diretamente proporcional (r ≈ 0,48). Apesar disso:
- Maior teor alcoólico não garante melhor qualidade
- Para equilíbrio no paladar, recomenda-se manter abaixo de 13%
- Pela legislação brasileira, o mínimo para ser considerado vinho é 7%

#### 4.3 Sulfitos e Ácido Cítrico × Qualidade
- **Sulfito:** manter abaixo de 1,0 g/L para priorizar processos mais naturais
- **Ácido cítrico:** manter abaixo de 0,5 g/L
- pH baixo contribui para maior longevidade do vinho

#### 4.4 Acidez Volátil × Teor Alcoólico
Não há padrão claro que permita prever o teor alcoólico a partir da acidez volátil.

### 5. Balanceamento de Classes
O dataset apresenta alto desbalanceamento nas classes de qualidade, o que poderia enviesar os modelos.

Foi aplicada a técnica **SMOTEENN**, que combina:
- **SMOTE:** gera amostras sintéticas para a classe minoritária
- **Undersampling inteligente:** remove ruídos e pontos conflitantes entre classes

Com isso, as fronteiras entre as classes ficaram mais limpas, aumentando a robustez do treinamento.

### 6. Conclusões e Próximos Passos
- A correlação entre teor alcoólico e qualidade é moderada, mas não implica causalidade
- O pré-processamento incluiu capping para outliers e padronização para variáveis de alta variabilidade
- O desbalanceamento foi corrigido com SMOTEENN, reduzindo viés nos modelos
- A próxima etapa será o treinamento de algoritmos de classificação usando as variáveis originais
- Caso o desempenho não atinja o esperado, serão incorporadas variáveis externas (tipo de uva, região climática, técnicas de vinificação) para aprimorar a previsão da qualidade

## 🤖 Machine Learning

Nesta etapa, aplicamos diferentes algoritmos de machine learning para prever a qualidade do vinho com base nas variáveis do nosso dataset.

O objetivo foi comparar modelos, avaliar métricas de desempenho e realizar validações para garantir que as previsões sejam consistentes e livres de overfitting, assegurando a melhor escolha para uso em produção.

### 🔹 Modelos utilizados

#### Classificação
- `logistic_regression` → log_reg
- `random_forest` → rf_clf
- `xgboost` → xgb_clf
- `lightgbm` → lgbm_clf
- `catboost` → catb_clf
- `tree_classifier` → treec_clf
- `svm_classifier` → svm_clf

#### Regressão
- `linear_regression` → lin_reg
- `random_forest` → rf_reg
- `xgboost` → xgb_reg
- `lightgbm` → lgbm_reg

#### Clustering
- `kmeans` → kmeans_cluster
- `dbscan` → dbscan_cluster

Nosso modelo base, a **Árvore de Classificação**, já apresentou um resultado muito satisfatório, como mostrado anteriormente. A partir dele, rodamos outros modelos para comparação e aplicamos validações para garantir que nossos dados não estivessem sofrendo de overfitting, o que poderia prejudicar as previsões.

### 📊 Comparação de Modelos
- **Regressão Logística** → Teve desempenho inferior ao modelo base
- **Todos os modelos** → Obtiveram F1-score acima de 80%, um ótimo resultado para nossas predições
- **Melhor desempenho** → Random Forest, com 97% de F1-score — modelo escolhido para as validações finais

### 🔍 Validações Realizadas

#### 1. Validação Cruzada + Random Search
Foi aplicada validação cruzada combinada com Random Search, que testa diferentes blocos de dados separadamente, preservando a generalização.

**Resultados:**
- Média dos scores por fold: 0.96
- Desvio padrão: 0.0053 (baixo, indicando consistência)
- Resultado idêntico de acurácia (0.955) foi obtido mesmo sem cruzar os dados, reforçando a estabilidade do modelo

#### 2. Análise de Overfitting
- Apesar de o treino apresentar score 1.0, o teste manteve 0.955, mostrando que o modelo está generalizando bem para dados novos
- A curva de aprendizado confirmou que a validação está próxima ao treino, sem evidências de overfitting

### ✅ Conclusão
Os testes e validações confirmaram que o **Random Forest** é o modelo mais adequado para este problema, entregando alta performance e mantendo a capacidade de generalização. O próximo passo será aplicar este modelo em dados novos para validar seu comportamento em produção.

## 🚀 Como utilizar no projeto

A arquitetura do projeto foi pensada para ser prática. Para treinar, avaliar e fazer previsões com qualquer modelo disponível, siga o guia abaixo.

### 1. Instalação e Configuração
Para replicar o ambiente de desenvolvimento, siga estes passos:

**Crie e ative o ambiente virtual:**
```bash
# Criar um novo ambiente
# Conda
conda create --name <nome_do_seu_projeto> python=3.10

# Ativar o ambiente
# Conda
conda activate <nome_do_seu_projeto>
# Git Bash (ou WSL/Linux)
source .venv/Scripts/activate
# PowerShell 
.venv\Scripts\Activate.ps1
# Prompt de Comando (cmd) 
.venv\Scripts\activate.bat
```

**Instale as dependências caso não esteja utilizando o poetry:**
```bash
pip install -r requirements.txt
```
```bash
conda env create -f environment.yml
```

**Com o poetry**
```bash
poetry install
```

**Configure no VS Code:**
1. Pressione `Ctrl + Shift + P`
2. Digite "Python: Select Interpreter"
3. Escolha o ambiente que você criou

### 2. Exemplo de uso com o pipeline_classification
A forma mais prática de testar um modelo é utilizando a função de pipeline. Basta fornecer o caminho do seu arquivo de dados e o nome do modelo desejado.

```python
from src.models.pipeline_classification import pipeline_classification

# Exemplo de uso para o modelo RandomForest
results = pipeline_classification(
    data_path=DATA_PROCESSED / 'seu_dataset_processado.csv',
    target_column='qualidade',
    model_name='random_forest',
    scale_type='standard',
    test_size=0.2
)

# Para inspecionar os resultados
print("Métricas de Avaliação:", results['metrics'])
print("Modelo Treinado:", results['model'])
```

### 3. Testes individuais
Para testar um modelo específico sem usar o pipeline completo, você pode criar e treinar diretamente:

```python
from src.models.model_factory import ModelFactory

# Carregar o modelo desejado
modelo = ModelFactory.create_classification_model("random_forest")

# Treinar e usar o modelo
modelo.fit(X_train, y_train)
predicoes = modelo.predict(X_test)
```

## 📚 Referências

- **UCI Machine Learning Repository** - Wine Quality Dataset

- **Winefun** - "Acidez volátil: conheça um dos defeitos mais controvertidos do mundo dos vinhos"  
  Fonte: Winefun  
  https://winefun.com.br/acidez-volatil-conheca-um-dos-defeitos-mais-controvertidos-do-mundo-dos-vinhos/

- **Wine.com.br** - Winepedia: "Álcool pra quê?"  
  https://www.wine.com.br/winepedia/alcool-pra-que/

---

*Este projeto faz parte do meu portfólio em desenvolvimento durante a pós-graduação em Data Analytics. À medida que avanço no curso, novas técnicas e análises serão incorporadas para enriquecer este e outros estudos.*