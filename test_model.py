import os
import joblib

from sklearn.datasets import load_iris

MODEL_FILE = "model.joblib"


def test_model_file_exists():
    assert os.path.isfile(MODEL_FILE)
    print("TEST 1 PASSED: Model file exists.")


def test_model_loads_correctly():
    artifact = joblib.load(MODEL_FILE)

    assert isinstance(artifact, dict)
    assert "model" in artifact
    assert "accuracy" in artifact
    assert "n_estimators" in artifact

    print("TEST 2 PASSED: Model loaded successfully.")


def test_model_predictions():
    artifact = joblib.load(MODEL_FILE)
    model = artifact["model"]

    iris = load_iris()
    predictions = model.predict(iris.data[:3])

    assert len(predictions) == 3
    assert all(
        prediction in [0, 1, 2]
        for prediction in predictions
    )

    print("TEST 3 PASSED: Predictions are valid.")


if __name__ == "__main__":
    test_model_file_exists()
    test_model_loads_correctly()
    test_model_predictions()

    print("\nAll three automated tests passed!")