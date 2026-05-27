import numpy as np
from model import forward_prop, get_predictions

def load_weights():
    W1 = np.load('W1.npy')
    b1 = np.load('b1.npy')
    W2 = np.load('W2.npy')
    b2 = np.load('b2.npy')
    return W1, b1, W2, b2

def predict_letter(image, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, image)
    prediction = get_predictions(A2)
    letter = chr(prediction[0] + ord('a')).upper()
    return letter

if __name__ == "__main__":
    W1, b1, W2, b2 = load_weights()
    print("Weights loaded, model ready!")