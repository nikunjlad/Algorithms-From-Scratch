"""
The encoder of the transformer
"""

import tensorflow as tf
from keras.layers import Dropout, Layer
from .attention import GlobalSelfAttention
from .feed_forward import FeedForward
from .positional_encoding import PositionalEmbedding

class EncoderLayer(Layer):

    def __init__(self, dModel, numHeads, dff, dropOutRate=0.1, **kwargs):

        super().__init__(**kwargs)
        self.supports_masking = True
        self.globalSelfAttention = GlobalSelfAttention(
            num_heads=numHeads,
            key_dim=dModel // numHeads,
            dropout=dropOutRate
        )

        self.ffn = FeedForward(dff=dff, dModel=dModel, dropoutRate=dropOutRate)

    def call(self, x):

        x = self.globalSelfAttention(x)

        x = self.ffn(x)

        return x

class Encoder(Layer):

    def __init__(self, numLayers, dModel, numHeads, sourceVocabSize, maximumPositionEncoding, dff, dropOutRate=0.1, **kwargs):

        super().__init__(**kwargs)

        self.dModel = dModel
        self.numLayers = numLayers

        self.positionalEmbedding = PositionalEmbedding(
            vocabSize=sourceVocabSize,
            dModel=dModel,
            maximumPositionEncoding=maximumPositionEncoding
        )

        self.encodingLayers = [
            EncoderLayer(dModel=dModel, dff=dff, numHeads=numHeads, dropOutRate=dropOutRate) for _ in range(numLayers)
        ]

        self.dropout = Dropout(rate=dropOutRate)

    def call(self, x):

        x = self.positionalEmbedding(x)

        x = self.dropout(x)

        for encoderLayer in self.encodingLayers:
            x = encoderLayer(x=x)

        return x


