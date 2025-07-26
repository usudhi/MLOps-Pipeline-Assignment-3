import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

print("Container verification script started.")

try:
    model = joblib.load('linear_regression_model.joblib')
    print("Model 'linear_regression_model.joblib' loaded successfully.")
    housing = fetch_california_housing()
    _, X_test, _, y_test = train_test_split(housing.data, housing.target, test_size=0.2, random_state=42)
    score = model.score(X_test, y_test)
    print(f"Container verification successful. Model R² Score: {score:.4f}")
except FileNotFoundError:
    print("Error: Model file not found. Ensure train.py ran successfully.")
    exit(1)
except Exception as e:
    print(f"An error occurred during verification: {e}")
    exit(1)