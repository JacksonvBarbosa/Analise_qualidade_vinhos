FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

# Instala dependências do sistema necessárias para bibliotecas como lightgbm/xgboost
# `libgomp1` fornece libgomp.so.1 (OpenMP) utilizada por esses pacotes.
RUN apt-get update \
	&& apt-get install -y --no-install-recommends \
	   ca-certificates \
	   libgomp1 \
	&& rm -rf /var/lib/apt/lists/*

# Instala dependências Python
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app/src

# Observação: treinar durante o build pode causar falhas (tempo, memória, dependências).
# Recomenda-se executar o treino em etapa separada (CI/job) ou no container em runtime.
# Se quiser treinar manualmente dentro do container, use:
#   docker run --rm -v $(pwd)/models:/app/models <image> python -m analise_qualidade_vinhos.pipeline.train

EXPOSE 8000

CMD ["uvicorn", "analise_qualidade_vinhos.api:app", "--host", "0.0.0.0", "--port", "8000"]

