"""
K-Nearest Neighbor Implementation from Scratch

"""

import argparse
import os
import numpy as np
from utils import SimpleDataLoader, SimplePreProcessor
import sys

class KNN:

    def __init__(self, k=1):

        self.k = k      # the number of nearest neighbors to get the vote for
        self.X_train = np.array([])     # KNN is lazy learner that memorizes data  - (2250,3072)
        self.y_train = np.array([])     # class index (0 - c-1) for every training row - (2250,)
        self.num_classes = 0

    def normalize(self, X):

        return X.astype('float64') / 255.0

    def train_test_split(self, X, y, test_size=0.25, seed=42):
        """_summary_
        Randomize the data and slice it to get training and test data

        Args:
            X (np.ndarray): Data of shape (batch_size, flattened_pixels)
            y (np.ndarray): Labels of shape (3000,1)
            test_size (float, optional): _description_. Defaults to 0.25.
            seed (int, optional): _description_. Defaults to 42.

        Returns:
            _type_: _description_
        """
        rng = np.random.default_rng(seed)
        n = X.shape[0]
        indices = rng.permutation(n)

        n_test = int(round(n * test_size))
        test_idx = indices[:n_test]
        train_idx = indices[n_test:]

        return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

    def euclidean(self, A, B):

        A_sq = (A ** 2).sum(axis=1)[:, None]
        B_sq = (B ** 2).sum(axis=1)[None, :]
        cross = A @ B.T

        sq_dist = A_sq - 2.0 * cross + B_sq

        np.maximum(sq_dist, 0, out=sq_dist)
        return np.sqrt(sq_dist)

    def fit(self, X_train, y_train):
        """_summary_
        KNN has no training. All the work happens at prediction time.
        We just store the data and the integer labels.

        Args:
            X_train (np.ndarray): num_train x 3072
            y_train (np.ndarray): num_train array of integer class indices
        """
        self.X_train = X_train      # the training data
        self.y_train = y_train      # integer labels (already encoded)
        self.num_classes = int(y_train.max()) + 1

    def predict(self, X_test, batch_size=256):

        k = min(self.k, self.X_train.shape[0])

        preds = []

        for start in range(0, X_test.shape[0], batch_size):
            batch = X_test[start:start + batch_size]
            distances = self.euclidean(batch, self.X_train)

            neighbor_idx = np.argpartition(distances, kth=k-1, axis=1)[:, :k]
            neighbor_classes = self.y_train[neighbor_idx]

            votes = np.eye(self.num_classes)[neighbor_classes].sum(axis=1)

            preds.append(votes.argmax(axis=1))

        return np.concatenate(preds)

    def evaluate(self, preds, y_test):

        return (preds == y_test).mean()

def main(args):

    # 1. Data loading - Images
    print("Loading images")
    imagePaths = []
    for root, dirs, files in os.walk(args["dataset"]):
        for file in files:
            imagePaths.append(os.path.join(root,file))

    sp = SimplePreProcessor(32,32)  # intantiate preprocessor class
    sdl = SimpleDataLoader([sp])    # intantiated data_loader class
    (data, labels) = sdl.load(imagePaths, verbose=500)
    print(data.shape)       # [batch_size, width, height, n_channels] - (3000,32,32,3)
    data = data.reshape((data.shape[0], 3072))
    print(data.shape)       # [batch_size, width*height*n_channels] - (3000x3072)

    print(f"Feature Matrix Size: {data.nbytes / (1024 * 1024.0):.1f}MB")

    # converting string classes into integer indices once, before splitting,
    # so train and test share the same encoding
    # classes = original string classes, labels = integer label representation
    classes, labels = np.unique(labels, return_inverse=True)

    # 2. Initialize Model
    knn = KNN(k=args["neighbors"])

    # 3. Normalize data
    data = knn.normalize(data)  # bring 0-255 data in 0-1 range

    # 4. Prepare Train/Test dataset
    (X_train, X_test, y_train, y_test) = knn.train_test_split(data, labels, test_size=0.25, seed=42)
    print(f"Train: {X_train.shape} images and {y_train.shape} labels, \
          \nTest: {X_test.shape} images and {y_test.shape} labels")

    # 5. Train/Fit a model
    knn.fit(X_train, y_train)

    # 6. Predict and evaluate the model
    preds = knn.predict(X_test)
    accuracy = knn.evaluate(preds, y_test)

    print(f"Classes: {classes.tolist()}")
    print(f"k={knn.k} -> accuracy: {accuracy * 100:.2f}%")


if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("-d","--dataset",required=True,help="path to input dataset")
    ap.add_argument("-k","--neighbors",type=int,default=5,help="# of nearest neighbors for classification")
    ap.add_argument("-j","--jobs",type=int,default=-1,help="# of jobs for k-NN distance (-1 uses all available cores)")
    args = vars(ap.parse_args())
    main(args)

    sys.exit(0)



