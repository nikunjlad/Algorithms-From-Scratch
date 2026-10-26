"""
The Decoder of the Transformer
"""

import tensorflow as tf
from keras.layers import Dropout, Layer
from .attention import CausalSelfAttention, CrossAttention
from .feed_forward import FeedForward
from .positional_encoding import PositionalEmbedding

class DecoderLayer(Layer):

    def __init__(self, dModel, numHeads, dff, dropOutRate=0.1, **kwargs):

        super().__init__(**kwargs)
        self.supports_masking = True

        self.causalSelfAttention = CausalSelfAttention(
            num_heads=numHeads,
            key_dim=dModel // numHeads,
            dropout=dropOutRate
        )

        self.crossAttention = CrossAttention(
            num_heads=numHeads,
            key_dim=dModel // numHeads,
            dropout=dropOutRate
        )

        self.ffn = FeedForward(dff=dff, dModel=dModel, dropoutRate=dropOutRate)

    def call(self, x, context, contextMask=None):

        x = self.causalSelfAttention(x=x)
        x = self.crossAttention(x=x, context=context, contextMask=contextMask)

        self.lastAttentionScores = self.crossAttention.lastAttentionScores

        x = self.ffn(x)
        return x


class Decoder(Layer):

    def __init__(self, numLayers, dModel, numHeads, targetVocabSize, maximumPositionEncoding, dff, dropOutRate=0.1, **kwargs):

        super().__init__(**kwargs)

        self.dModel = dModel
        self.numLayers = numLayers

        self.positionalEmbedding = PositionalEmbedding(
            vocabSize=targetVocabSize,
            dModel=dModel,
            maximumPositionEncoding=maximumPositionEncoding
        )

        self.decoderLayers = [
            DecoderLayer(dModel=dModel, dff=dff, numHeads=numHeads, dropOutRate=dropOutRate) for _ in range(numLayers)
        ]

        self.dropout = Dropout(rate=dropOutRate)

    def call(self, x, context, contextMask=None):

        x = self.positionalEmbedding(x=x)

        x = self.dropout(x)

        for decoderLayer in self.decoderLayers:
            x = decoderLayer(x=x, context=context, contextMask=contextMask)

        self.lastAttentionScores = self.decoderLayers[-1].lastAttentionScores

        return x

