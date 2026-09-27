"""
Utility script to preprocess images

"""

import cv2
import os
import numpy as np

class SimplePreProcessor:

    def __init__(self, width, height, inter=cv2.INTER_AREA):
        """
        This class takes width, height and interpolation as input to
        resize an image.

        Args:
            width (int): target width to resize the image to
            height (int): target height to resize the image to
            inter (int, optional): the interpolation method to use for image resize. Defaults to cv2.INTER_AREA.
        """

        self.width = width
        self.height = height
        self.inter = inter

    def preprocess(self, image):
        """_summary_
        This function applies the interpolation to the image and
        resizes it to the dimensions defined by the constructor above

        Args:
            image (np.array): the input image to resize

        Returns:
            np.array: the resized image to return back.
        """
        return cv2.resize(image, (self.width, self.height),
                          interpolation=self.inter)

class SimpleDataLoader:

    def __init__(self, preprocessors=None):

        self.preprocessors = preprocessors
        if self.preprocessors is None:
            self.preprocessors = []

    def load(self, imagePaths, verbose=-1):

        data = []
        labels = []

        for (i, imagePath) in enumerate(imagePaths):

            image = cv2.imread(imagePath)
            label = imagePath.split(os.path.sep)[-2]

            if self.preprocessors is not None:

                for p in self.preprocessors:
                    image = p.preprocess(image)

            data.append(image)
            labels.append(label)

            if verbose > 0 and (i + 1) % verbose == 0:
                print(f"[INFO] Processed {i+1}/{len(imagePaths)}")

        return (np.array(data), np.array(labels))
