# 🍷 App Streamlit - Predição de Qualidade de Vinhos

## Como executar o app

### 1. Instalar dependências
```bash
pip install -r requirements.txt
```

### 2. Treinar o modelo (se ainda não foi treinado)
```bash
python src/analise_qualidade_vinhos/pipeline/train.py
```

### 3. Executar o app Streamlit
```bash
streamlit run app.py
```

O app será aberto automaticamente no navegador em `http://localhost:8501`

### Executar com Docker Compose

Se você estiver usando Docker Compose (recomendado para consistência):

```bash
docker compose up --build streamlit
```

Ou subir API + Streamlit juntos:

```bash
docker compose up --build web streamlit
```

## Funcionalidades

- ✅ Interface intuitiva e bonita
- ✅ Formulário completo com todas as características do vinho
- ✅ Predição em tempo real
- ✅ Visualização de métricas
- ✅ Recomendações de melhoria
- ✅ Valores de referência na sidebar

## Uso em Produção

O app está pronto para uso pelos funcionários da empresa. Basta:
1. Preencher os campos com as características do vinho
2. Clicar em "Prever Qualidade"
3. Ver o resultado e seguir as recomendações



