from src.etl.extract import load_csv
from src.features.engineering import create_features
from src.models.train_model import train_model

# Insira as funções do seu projeto para que o Pipeline possa agira corretamente
def main():
    df = load_csv("insira o arquivo")
    df = create_features(df)

    X = df[["o que for treinar"]]
    y = df['target']

    model = train_model(X, y)
    print('Modelo treinado com sucesso!')


if __name__ == "__main__":
    main()