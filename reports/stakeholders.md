# Relatório Executivo — Qualidade de Vinhos

## O que foi feito
- Classificação em 2 faixas: Baixa qualidade e Alta qualidade.
- Engenharia de atributos simples (interações de acidez, enxofre e álcool) e balanceamento com SMOTE.
- Modelo RandomForest com pipeline reproduzível (treino + inferência).
- API FastAPI e Docker para uso rápido e escalável.

## Principais descobertas
- **Álcool** e **enxofre total/livre** estão entre os fatores mais associados à nota de qualidade.
- Razões como `density_alcohol_ratio` ajudam a separar rótulos de alta qualidade.
- Balanceamento de classes foi necessário para manter F1 equilibrado.

## Métricas atuais (baseline)
- Valores consolidados em `reports/metrics.json` após rodar o treino.
- Métricas-chave: accuracy e F1 ponderado; modelo estável para 2 faixas de qualidade.

## Como usar
- CLI: `python -m analise_qualidade_vinhos.pipeline.train` gera modelo e métricas.
- API: subir com Docker (`docker run -p 8000:8000 wine-quality`) e chamar `POST /predict`.

## Próximos passos sugeridos
- Monitorar drift e refazer treino mensalmente.
- Logar previsões/erros para feedback contínuo.
- Avaliar modelos adicionais se surgirem mais dados ou novas variáveis.

## Pontos para apresentação
- Objetivo: fornecer um sistema reprodutível que apoie decisões operacionais sobre qualidade do vinho.
- Impacto esperado: identificação rápida de lotes com risco, redução de retrabalho e suporte a intervenção na produção.
- Diferencial técnico: pipeline completo com engenharia de features, balanceamento e comparação automatizada de modelos; deploy via API e app interativo.


