import os
import joblib

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Initial model configuration
N_ESTIMATORS = 100
MODEL_FILE = "model.joblib"


def train_model():
    print("STEP 1: Loading Iris dataset")

    iris = load_iris()
    X = iris.data
    y = iris.target

    print("Total records:", len(X))
    print("Number of features:", X.shape[1])

    print("\nSTEP 2: Splitting dataset")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("Training records:", len(X_train))
    print("Testing records:", len(X_test))

    print("\nSTEP 3: Training Random Forest model")

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print("Number of trees:", N_ESTIMATORS)
    print("Model accuracy:", round(accuracy * 100, 2), "%")

    artifact = {
        "model": model,
        "feature_names": iris.feature_names,
        "target_names": iris.target_names,
        "accuracy": accuracy,
        "n_estimators": N_ESTIMATORS
    }

    print("\nSTEP 4: Saving model")

    joblib.dump(artifact, MODEL_FILE)

    print("Model saved successfully:", MODEL_FILE)
    print("File exists:", os.path.exists(MODEL_FILE))
    print("\nTraining completed successfully.")


if __name__ == "__main__":
    train_model()