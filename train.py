import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def train_and_save_model():
    """
    Trains a Linear Regression model on the California Housing dataset
    and saves it to a joblib file.
    """
    housing = fetch_california_housing()
    X, y = housing.data, housing.target
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    joblib.dump(model, 'linear_regression_model.joblib')
    print("Model trained and saved as linear_regression_model.joblib")

if __name__ == "__main__":
    train_and_save_model()
