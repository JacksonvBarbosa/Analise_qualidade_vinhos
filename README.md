# Análise da Qualidade de Vinhos
## Portfólio de Data Analytics | Jackson dos Santos Ventura

---

## 📊 Projeto: Análise Exploratória de Vinhos Importados para JACKWine

### Contexto do Projeto

A distribuidora **JACKWine** está expandindo seu catálogo através da importação de vinhos portugueses. Como analista de dados da empresa, realizei uma análise exploratória dos dados, identificando os fatores químicos que influenciam a qualidade dos produtos. Além disso, desenvolvi modelos de machine learning para prever a pontuação de qualidade dos vinhos, fornecendo insights estratégicos para apoiar o processo de seleção e importação.

### Objetivo

Identificar relações entre os componentes químicos e a qualidade dos vinhos portugueses, visando compreender os principais fatores que influenciam sua avaliação e utilizar esses insights para apoiar decisões estratégicas e o desenvolvimento de modelos preditivos com **machine learning**.

---

## 🔬 Metodologia

### 1. Aquisição e Preparação dos Dados
- Dataset: winequality-red.csv (Fonte: UCI Machine Learning Repository)
- Ferramentas: Python, Pandas, NumPy, Matplotlib, Seaborn
- Procedimentos:
  - Importação e inspeção inicial do dataset
  - Tratamento de valores duplicados e ausentes
  - Ajuste de tipos de dados
  - Criação de funções modulares para extração, transformação e armazenamento de dados no pacote ***etl/***
  - Implementação de tratamento de outliers e balanceamento de classes no pacote ***features/***

### 2. Análise Exploratória de Dados (EDA)
- Visualização e análise de distribuições de variáveis químicas
- Identificação de correlações entre variáveis e qualidade do vinho
- Uso de gráficos de dispersão, boxplots, histogramas e mapas de calor
- Criação do módulo **visualization/** para centralizar funções gráficas reutilizáveis

### 3. Desenvolvimento de Modelos de Machine Learning
- Estrutura de código organizada em pacotes reutilizáveis (**models/**) para classificação, regressão e clustering
- Implementação de pipelines (**pipeline_classification.py**, **pipeline_regression.py**, **pipeline_clustering.py**) para padronizar o fluxo de treino e avaliação
- Utilização do **model_factory.py** com lazy loading, permitindo carregar modelos sob demanda e melhorar a escalabilidade do projeto
- Aplicação de técnicas de otimização de hiperparâmetros com **RandomizedSearchCV**
- Avaliação de modelos utilizando métricas como Acurácia, Precisão, Recall, F1-score e ROC AUC

### 4. Modularização e Escalabilidade
- Estrutura do projeto planejada para reuso e manutenção em diferentes datasets
- Separação de responsabilidades por pacotes:
  - **etl/** → Funções de extração, transformação e armazenamento
  - features/ → Engenharia de variáveis e tratamento de dados
  - models/ → Treinamento, avaliação e pipelines de ML
  - visualization/ → Geração de gráficos e plots
- Suporte para inclusão de novos modelos no **model_factory.py** sem alteração no restante do código

### 5. Armazenamento e Versionamento de Modelos
- Modelos treinados salvos em **models_storage/** para reutilização futura
- Uso de joblib para serialização
- Versionamento do código via GitHub

---

## 📈 Resultados e Análises

### Acidez Volátil vs. Qualidade do Vinho

![Acidez Volátil vs. Qualidade](Graficos\acidez_qualidade_vinho.png)

**Análise:**
- Vinhos de qualidade superior apresentam **menor acidez volátil**
- Vinhos de alta qualidade (nível 8) apresentam acidez volátil média de **0,40 g/L**
- **Significado prático:** Conforme pesquisa no site Winefun, a maioria dos vinhos premium mantém níveis de ácido acético entre 0,3 a 0,5 g/L
- O limiar de percepção sensorial fica entre 0,6 a 0,9 g/L, a partir do qual a acidez volátil torna-se perceptível ao paladar
- **Implicação para importação:** Priorizar vinhos com acidez volátil na faixa de 0,3 a 0,5 g/L para garantir maior aceitação de mercado

### Teor Alcoólico vs. Qualidade do Vinho

![Teor Alcoólico vs. Qualidade](Graficos\teor_alcoolico_qualidade_vinho.png)

**Análise:**
- Vinhos com maior teor alcoólico tendem a receber avaliações mais altas
- Vinhos de qualidade superior (nível 8) apresentam teor alcoólico médio de **12,10%**
- **Consideração importante:** Segundo a Winepedia, o teor alcoólico isoladamente não determina qualidade, sendo ideal manter-se abaixo de 13% para garantir equilíbrio no paladar
- **Implicação para importação:** Selecionar vinhos com teor alcoólico entre 10,5% e 12,5% para melhor harmonia sensorial

### Relação entre Acidez Volátil e Teor Alcoólico

![Acidez Volátil vs. Teor Alcoólico](Graficos\relacao_acidez_teor_alcoolicopng.png)

**Análise:**
- **Correlação de Pearson: -0,20** (correlação negativa fraca)
- Quando a acidez volátil aumenta, o teor alcoólico tende a diminuir levemente
- O impacto é limitado, sugerindo que outros fatores influenciam mais significativamente o teor alcoólico
- **Implicação para importação:** A acidez volátil não deve ser utilizada como preditor do teor alcoólico

### Análise Detalhada da Relação entre Acidez Volátil e Qualidade

![Acidez Volátil vs. Qualidade Detalhada](Graficos\relacao_acidez_qualidade_vinho.png)

**Análise:**
- Relação inversa: maiores valores de acidez volátil associam-se a avaliações inferiores
- A intensidade moderada da correlação (-0,39) indica que, embora relevante, a acidez volátil não é o único determinante da qualidade
- Outros fatores como taninos, açúcar residual e perfil aromático também influenciam a qualidade percebida
- **Recomendação:** Realizar análises complementares (testes sensoriais, avaliação de outras variáveis químicas) para compreensão mais abrangente

### Análise Detalhada da Relação entre Teor Alcoólico e Qualidade

![Teor Alcoólico vs. Qualidade Detalhada](Graficos\relacao_teor_alcoolico_qualidade_vinho.png)

**Análise:**
- A correlação moderada positiva (0,48) não implica causalidade direta entre maior teor alcoólico e melhor qualidade
- O contexto é essencial: o teor alcoólico precisa estar em harmonia com outros componentes do vinho
- **Recomendação:** Considerar o teor alcoólico como um dos múltiplos fatores na seleção de vinhos de qualidade

---

## 🎯 Conclusões e Recomendações

### Principais Insights:
1. **Acidez Volátil:**
   - Manter na faixa de 0,40-0,42 g/L para vinhos de qualidade superior
   - Evitar produtos com acidez volátil acima de 0,6 g/L, onde se torna perceptível sensorialmente

2. **Teor Alcoólico:**
   - Focar em vinhos com teor alcoólico entre 10,5% e 12,5%
   - Lembrar que o equilíbrio é mais importante que valores elevados

3. **Seleção para Importação:**
   - Utilizar estes parâmetros como guia inicial, mas não como critérios absolutos
   - Complementar análises químicas com avaliação sensorial

### Próximos Passos:
- Expandir análise para incluir outros parâmetros químicos (pH, açúcares residuais, sulfitos)
- Desenvolver modelo preditivo de qualidade baseado em múltiplas variáveis
- Realizar análises comparativas entre vinhos portugueses e outras regiões produtoras

---

## 📚 Referências
- UCI Machine Learning Repository - Wine Quality Dataset
- Winefun - "Acidez volátil: conheça um dos defeitos mais controvertidos do mundo dos vinhos"
  Fonte: Winefun  
  https://winefun.com.br/acidez-volatil-conheca-um-dos-defeitos-mais-controvertidos-do-mundo-dos-vinhos/
- Wine.com.br - Winepedia: "Álcool pra quê?"
  link:  
  https://www.wine.com.br/winepedia/alcool-pra-que/

---

*Este projeto faz parte do meu portfólio em desenvolvimento durante a pós-graduação em Data Analytics. À medida que avanço no curso, novas técnicas e análises serão incorporadas para enriquecer este e outros estudos.*
