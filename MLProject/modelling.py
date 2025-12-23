import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

def main():
    # 1. Load dataset (clean)
    df = pd.read_csv("heart_disease_uci_clean.csv")

    # 2. Pisahkan fitur dan target
    X = df.drop(columns=["target"])
    y = df["target"]

    # 3. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 4. Aktifkan MLflow autolog
    mlflow.sklearn.autolog()

    # 5. Train model
    with mlflow.start_run():
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        # 6. Evaluasi
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        print(f"Accuracy: {acc}")

if __name__ == "__main__":
    main()
