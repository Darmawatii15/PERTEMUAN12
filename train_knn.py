# train_knn.py
import numpy as np, joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# load data embedding yg sudah kamu buat
X = np.load("X_train.npy")
y = np.load("y_train.npy", allow_pickle=True)

# Pipeline: scaling + classifier KNN
clf = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=3, metric="euclidean"))
])

clf.fit(X, y)
joblib.dump(clf, "facenet_knn.joblib")
print("Saved facenet_knn.joblib")
