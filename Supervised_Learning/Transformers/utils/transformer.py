"""
The Transformer Architecture

"""

import tensorflow as tf
from keras import Model
from keras.layers import Dense
from keras.metrics import Mean
from .decoder import Decoder
from .encoder import Encoder

class Transformer(Model):

    def __init__(self, encNumLayers, decNumLayers, dModel, numHeads, dff, sourceVocabSize, targetVocabSize,
                 maximumPositionEncoding, dropOutRate=0.1, **kwargs):

        super().__init__(**kwargs)

        self.encoder = Encoder(
            numLayers=encNumLayers,
            dModel=dModel,
            numHeads=numHeads,
            sourceVocabSize=sourceVocabSize,
            maximumPositionEncoding=maximumPositionEncoding,
            dff=dff,
            dropOutRate=dropOutRate
        )

        self.decoder = Decoder(
            numLayers=decNumLayers,
            dModel=dModel,
            numHeads=numHeads,
            targetVocabSize=targetVocabSize,
            maximumPositionEncoding=maximumPositionEncoding,
            dff=dff,
            dropOutRate=dropOutRate
        )

        self.finalLayer = Dense(units=targetVocabSize)

    def call(self, inputs):

        (source, target) = inputs
        sourceMask = tf.not_equal(source,0)

        encoderOutput = self.encoder(x=source)
        decoderOutput = self.decoder(x=target, context=encoderOutput, contextMask=sourceMask)

        shape = tf.shape(decoderOutput)
        flat = tf.reshape(decoderOutput, [-1, decoderOutput.shape[-1]])
        logits = self.finalLayer(flat)
        logits = tf.reshape(logits, [shape[0], shape[1], -1])

        try:
            del logits._keras_mask
        except AttributeError:
            pass

        return logits

