import os
import cv2
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

DATASET_DIR = "datasets"
IMAGE_SIZE = 64

X = []
y = []


for label, folder in enumerate(["cats", "dogs"]):
    folder_path = os.path.join(DATASET_DIR, folder)
    for file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, file)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        img = img / 255.0
        X.append(img.flatten())
        y.append(label)

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


knn = KNeighborsClassifier(n_neighbors=5)
svm = SVC(kernel="linear")
logistic = LogisticRegression(max_iter=1000)

knn.fit(X_train, y_train)
svm.fit(X_train, y_train)
logistic.fit(X_train, y_train)


os.makedirs("models", exist_ok=True)
joblib.dump(knn, "models/knn.pkl")
joblib.dump(svm, "models/svm.pkl")
joblib.dump(logistic, "models/logistic.pkl")

print("Models trained and saved successfully")
