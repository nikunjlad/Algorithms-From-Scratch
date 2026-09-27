"""
KNN Slim implementation

Algorithm:
1. Load Data and labels - X and y
2. Normalize data
3. Split data and labels into train and test - Xtrain,ytrain and xtest,ytest
4. Fit the training data - just get Xtrain,ytrain,num_classes
5. Predict on Test data in batches
6. Evaluate the test predictions against GT labels

"""

import numpy as np

seed = 777
rng = np.random.default_rng(seed)   # create a random number generator

class KNNSlim:

    def __init__(self,k=3):
        self.k = k      # number of nearest neigbhors to find
        self.X_train = np.array([]) # initialize x_train
        self.y_train = np.array([]) # initialize y_train
        self.num_classes = 0        # initialize num classes to 0

    def normalize(self,X):
        """_summary_
        Given a data, normalize it to bring it in 0-1 range

        Args:
            X (np.ndarray): The data we need to normalize. Data is in (num_samples X num_features) shape - (120x3072)
        """
        return X.astype('float') / 255.0    # since our data is in pixels, we divide it by 255

    def train_test_split(self,X,y,test_size):
        """_summary_
        Given a training data and labels, convert them into training and testing given split ratio of test size

        Args:
            X(np.ndarray): Data of shape - (num_samples X num_features) - (120x3072)
            y (np.ndarray): Labels of shape - (num_samples,) - (120,)
            test_size (float): the test ratio as a float value representing fraction of data to be considered test
        """
        num_samples = X.shape[0]    # number of training samples
        indices = rng.permutation(num_samples)      # randomly shuffle training data. Indices is array of training data indices shuffled randomly

        test_split = int(test_size * num_samples)   # number of samples in test_split - 0.25 * 120 = 30
        test_indices = indices[:test_split]         # first 30 indices in indices are test data
        train_indices = indices[test_split:]        # remaining 90 indices are train data

        return X[train_indices],y[train_indices],X[test_indices],y[test_indices]

    def fit(self,X,y,num_classes):
        """_summary_
        KNN has not learning part, so we simply assign them to class variables

        Args:
            X (np.ndarray): data of size - (num_train_samples X num_features) - (90x3072)
            y (np.ndarray): labels of size - (num_train_samples,) - (90,)
            num_classes (int): number of classes
        """
        self.X_train = X
        self.y_train = y
        self.num_classes = num_classes

    def euclidean(self,A,B):
        """_summary_
        This function calculates euclidean distance between 2 matrices
        Euclidean distance = sqrt((A-B)^2) = sqrt(||A||^2 - 2.A.B + ||B||^2)
        Args:
            A (np.ndarray): Test batch matrix of shape - (batch_size X num_features) - (5x3072)
            B (np.ndarray): Training data of shape - (num_train_samples X num_features) - (90x3072)
        """
        A_sq = (A ** 2)             # (5x3072) - square all pixel values of test data
        A_sq = A_sq.sum(axis=1)     # (5,) - sum elements across column features to get a row vector - one per data sample
        A_sq = A_sq[:,None]         # expand column dim - (5x1)

        B_sq = (B ** 2)             # (90x3072) - square all pixel values of training data
        B_sq = B_sq.sum(axis=1)     # (90,) - sum elements across column features to get a row vector - one per data sample
        B_sq = B_sq[None,:]         # (1x90) - prepare a row vector

        dot = A @ B.T             # A = (5x3072), B.T = (3072x90) - A dot(@) B.T = (5x3072) x (3072x90) = (5x90) - its a matrix product

        sq_dist = A_sq - 2.0 * dot + B_sq
        np.maximum(sq_dist,0,out=sq_dist)   # this step ensures, negative values in our distance calculation are clipped to 0

        return np.sqrt(sq_dist)

    def predict(self,X_test,batch_size):
        """_summary_
        This function batches the test data using batch size for KNN euclidean distance calculation

        Args:
            X_test (np.ndarray): The test data which is in (num_test_samples X num_features) shape - (30x3072)
            batch_size (int):   The size of the batch in which to process KNN comparisons
        """

        # first check K value is smaller than number of samples. else take number of samples as the K value
        k = min(self.k, self.X_train.shape[0])
        preds = []

        for start in range(0,X_test.shape[0],batch_size):
            X_test_batch = X_test[start:start + batch_size]     # create batches of 5 test data - (5x3072)
            distances = self.euclidean(X_test_batch, self.X_train)      # distances of 5 test data against 90 training - (5x90)

            # neighbor indices = (5xk) = (5x3)
            neighbor_indices = np.argpartition(distances,kth=k-1,axis=1)[:,:k]      # get the K=3 nearest indices of training data closest to our test data
            neighbor_classes = self.y_train[neighbor_indices]                       # for the training data selected as neighbors, get corresponding labels as well based on indices

            votes = np.eye(self.num_classes)                            # create one-hot encoding of classes - for 3 classes we get 3x3 matrix of [[1,0,0],[0,1,0],[0,0,1]]
            votes = votes[neighbor_classes]                             # index votes with neighbor_classes to get one-hot classes - this gives us (5x3x3) = (batch_size X num_neighbors X num_classes)
            votes = votes.sum(axis=1)                                   # sum across neighbors to get - (batch_size X num_classes)

            preds.append(votes.argmax(axis=1))              # get argument across classes which is maximum

        return np.concatenate(preds)        # convert preds into a single vector

    def evaluate(self,preds,y_test):
        """_summary_
        Evaluate predictions against
        Args:
            preds (np.ndarray): predictions array of shape - (30,)
            y_test (np.ndarray): test labels of shape - (30,)

        Returns:
            float: The fraction that represents the % of values matched correctly
        """
        return (preds == y_test).mean()     # return fraction of labels that match over total labels (30)




def main():

    # assume image of size 32x32x3 pixels. If this image if flattened, we get 3072 pixels
    # let us assume we have 90 training data and 30 test data with total of 120 data
    total_samples = 120
    num_pixels_in_img = 3072
    classes = np.array(["class1","class2","class3"])
    num_classes = classes.shape[0]

    data = rng.integers(low=0,high=256,size=(total_samples,num_pixels_in_img))   # (120x3072)
    labels = rng.integers(low=0,high=num_classes,size=(total_samples,))       # (120,)

    # Initialize KNN
    knn = KNNSlim(k=3)

    # Normalize data
    data = knn.normalize(data)  # convert our data which is in 0-255 range to 0-1 range

    # split our data into training and testing
    X_train,y_train,X_test,y_test = knn.train_test_split(data,labels,test_size=0.25)
    print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)

    # fit the data
    knn.fit(X_train,y_train,num_classes)

    # predict on test data
    preds = knn.predict(X_test,batch_size=5)

    # evaluate the data
    accuracy = knn.evaluate(preds, y_test)

    print(f"Accuracy of KNN: {accuracy * 100.0:.2f}%")
    print(f"Predictions of first 5 test data: {classes[preds][:5]}")



if __name__ == "__main__":
    main()

