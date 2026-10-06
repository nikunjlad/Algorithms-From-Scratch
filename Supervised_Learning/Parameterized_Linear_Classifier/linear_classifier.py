"""
Parameterized Learning

"""

import numpy as np
import cv2
from config import ANIMALS_DATASET

# define class labels
labels = ["dog", "cat", "panda"]    # K labels
rng = np.random.default_rng(42)

# define weights - (KxD) and bias (b) = (Kx1)
W = rng.standard_normal((3, 3072), dtype=np.float32)
b = rng.random(3)

# we just load one image to demonstrate the parameterized operation - y = Wx + b
orig = cv2.imread(f"{ANIMALS_DATASET}/dogs/dogs_00001.jpg")
# cv2.imshow("pic", orig)
# cv2.waitKey(0)
X = cv2.resize(orig, (32,32)).flatten()
print(X.shape)

# scores (y) = W.X' + b  [(3x3072) x (3072x1) + (3x1)]
scores = W.dot(X) + b
print(scores)

for label,score in zip(labels,scores):
    print(f"[INFO] {label}:{score:.2f}")

cv2.putText(orig, f"Label: {labels[np.argmax(scores)]}",
            org=(10,30),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.9,
            color=(0,0,255),
            thickness=2)

cv2.imshow("Image",orig)
cv2.waitKey(0)