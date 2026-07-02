import base64

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request

from model import forward_prop, get_predictions
from predict import load_weights

app = Flask(__name__)
W1, b1, W2, b2 = load_weights()


def preprocess_canvas_image(data_url):
    _, encoded = data_url.split(",", 1)
    img_bytes = base64.b64decode(encoded)
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)

    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = cv2.findNonZero(thresh)
    if coords is None:
        return None
    x, y, w, h = cv2.boundingRect(coords)
    if w < 5 or h < 5:
        return None

    cropped = thresh[y:y + h, x:x + w]
    resized = cv2.resize(cropped, (28, 28), interpolation=cv2.INTER_AREA)
    return (resized / 255.0).reshape(784, 1)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_route():
    data = request.get_json()
    image = preprocess_canvas_image(data["image"])
    if image is None:
        return jsonify({"letter": None, "confidence": 0})

    _, _, _, A2 = forward_prop(W1, b1, W2, b2, image)
    pred = get_predictions(A2)[0]
    confidence = float(A2[pred][0]) * 100
    letter = chr(pred + ord("a")).upper()
    return jsonify({"letter": letter, "confidence": confidence})


if __name__ == "__main__":
    app.run(debug=True)
