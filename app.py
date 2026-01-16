from flask import Flask, render_template, request
import joblib
import cv2
import numpy as np
import os

from flask import send_from_directory

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

models = {
    "knn": joblib.load("models/knn.pkl"),
    "svm": joblib.load("models/svm.pkl"),
    "logistic": joblib.load("models/logistic.pkl")
}

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (64, 64))
    img = img / 255.0
    img = img.flatten()
    return img.reshape(1, -1)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    image = request.files['image']
    model_name = request.form['model']

    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
    image.save(image_path)

    img_vector = preprocess_image(image_path)
    model = models[model_name]
    prediction = model.predict(img_vector)[0]

    result = "Cat!" if prediction == 0 else "Dog!"

    return render_template(
        "index.html",
        prediction=result,
        image_file=image.filename
    )


if __name__ == "__main__":
    app.run(debug=True)

