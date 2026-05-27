import numpy as np
from data_loader import load_data
from model import init_params, forward_prop, backward_prop, update_params, get_predictions, get_accuracy

def gradient_descent(X, Y, alpha, iterations, W1=None, b1=None, W2=None, b2=None):
    if W1 is None:
        W1, b1, W2, b2 = init_params()
    m = X.shape[1]
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y, m)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            predictions = get_predictions(A2)
            acc = get_accuracy(predictions, Y)
            print(f"Iteration {i} — Accuracy: {acc:.4f}")
    return W1, b1, W2, b2

if __name__ == "__main__":
    print("Loading data...")
    X_train, Y_train, X_dev, Y_dev = load_data()
    print("Training...")
    W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.10, 1000)
    np.save('W1.npy', W1)
    np.save('b1.npy', b1)
    np.save('W2.npy', W2)
    np.save('b2.npy', b2)
    print("Weights saved!")