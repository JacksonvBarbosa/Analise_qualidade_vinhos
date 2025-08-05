# Análise da Qualidade de Vinhos Tintos Portugueses
## Portfólio de Data Analytics | Jackson dos Santos Ventura

---

## 📊 Projeto: Análise Exploratória de Vinhos Importados para JACKWine

### Contexto do Projeto

A distribuidora **JACKWine** está expandindo seu catálogo através da importação de vinhos portugueses, com destaque para a marca "Vinho Verde". Como analista de dados da empresa, fui designado para realizar uma análise exploratória dos vinhos tintos, a fim de entender os fatores químicos que influenciam sua qualidade e fornecer insights para o processo de seleção e importação.

### Objetivo

Identificar relações entre os componentes químicos e a qualidade dos vinhos tintos portugueses, com foco específico na acidez volátil e teor alcoólico, a fim de determinar parâmetros ideais para novos produtos a serem importados.

---

## 🔬 Metodologia

### 1. Aquisição e Preparação dos Dados
- Dataset: "winequality-red.csv" (Fonte: UCI Machine Learning Repository)
- Ferramenta: Python (Pandas, Matplotlib, Seaborn)
- Procedimentos:
  - Importação e inspeção inicial da estrutura do dataset
  - Identificação e tratamento de valores duplicados
  - Tratamento de valores ausentes
  - Verificação de tipos de dados e conversões necessárias

### 2. Análise Exploratória de Dados (EDA)
Foco nas seguintes variáveis e relacionamentos:
- Distribuição da acidez volátil por níveis de qualidade
- Distribuição do teor alcoólico por níveis de qualidade
- Correlação entre acidez volátil e teor alcoólico
- Correlação entre acidez volátil e qualidade do vinho
- Correlação entre teor alcoólico e qualidade do vinho

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
