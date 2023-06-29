import persistence
import torch
from sklearn.linear_model import LinearRegression


def regression(X, y):
    X = X.numpy()
    y = y.numpy()
    reg = LinearRegression().fit(X, y)
    return reg.score(X, y)


alphas = IO.read_tensor("alphas.pt")
accuracies = IO.read_tensor("accuracies.pt")
print(alphas)
print(accuracies)
print(regression(alphas, accuracies))
