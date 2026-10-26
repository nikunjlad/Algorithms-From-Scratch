"""
This file contains implementation of the Feed Forward Network

"""

from keras import Sequential
from keras.layers import Add, Dense, Dropout, Layer, LayerNormalization

class FeedForward(Layer):

    def __init__(self, dff, dModel, dropoutRate=0.1, **kwargs):

        super().__init__(**kwargs)
        self.supports_masking = True

        # initialize Sequential models
        self.seq = Sequential([
            Dense(units=dff, activation="relu"),
            Dense(units=dModel),
            Dropout(rate=dropoutRate)
        ])
        self.seq.supports_masking = True

        # initialize Addition and Layer Normalization layers
        self.add = Add()
        self.layernorm = LayerNormalization()

    def call(self, x):

        x = self.add([x, self.seq(x)])

        x = self.layernorm(x)

        return x

