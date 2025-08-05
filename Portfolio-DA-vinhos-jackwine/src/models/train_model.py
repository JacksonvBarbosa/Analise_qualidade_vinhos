from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Modelo simples crie modelos conforme seu projeto necessite
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model

