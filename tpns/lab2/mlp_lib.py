
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

def classification_example():
    print("Iris Dataset")

    from sklearn.datasets import load_iris
    iris = load_iris()

    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    mlp = MLPClassifier(
        hidden_layer_sizes=(4),
        activation='relu',
        solver='sgd',
        learning_rate_init=0.05,
        max_iter=100,
        early_stopping=False,
        tol=0,
        n_iter_no_change=10,
        validation_fraction=0.1,
        random_state=1
    )

              
    mlp.fit(X_train, y_train)

    print(f"\nОбучение завершено за {mlp.n_iter_} итераций")

    y_train_pred = mlp.predict(X_train)
    y_test_pred = mlp.predict(X_test)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    print(f"\nРезультаты:")
    print(f"  Точность на обучении: {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"  Точность на тесте:    {test_acc:.4f} ({test_acc*100:.2f}%)")

    return mlp, scaler

def regression_example():
    print("Laptop Price")

    data = pd.read_csv("../Laptop_price.csv")

    brand_dummies = pd.get_dummies(data['Brand'], prefix='Brand')

    X = pd.concat([data.drop(['Brand', 'Price'], axis=1), brand_dummies], axis=1)
    y = data['Price'].values


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    scaler_X = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)

    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()

    mlp = MLPRegressor(
        hidden_layer_sizes=(4),
        activation='relu',
        solver='sgd',
        learning_rate_init=0.01,
        max_iter=500,
        tol=0,
        n_iter_no_change=500,
        early_stopping=False,
    )


    mlp.fit(X_train, y_train_scaled)

    print(f"\nОбучение завершено за {mlp.n_iter_} итераций")

    y_train_pred_scaled = mlp.predict(X_train)
    y_test_pred_scaled = mlp.predict(X_test)

    y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled.reshape(-1, 1)).flatten()
    y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled.reshape(-1, 1)).flatten()

    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    print(f"\nРезультаты:")
    print(f"  RMSE на обучении: {train_rmse/y_train.std():.2f}")
    print(f"  RMSE на тесте:    {test_rmse/y_test.std():.2f}")

    return mlp, scaler_X, scaler_y

if __name__ == "__main__":
    classification_example()

    regression_example()

