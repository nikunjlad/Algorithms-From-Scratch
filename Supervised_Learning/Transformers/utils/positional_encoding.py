"""
This file provides code for positionally encoding our source and target sentences to preserve order of
words into our layers. A position is the index of the word in a sentence. For instance, For a sentence - "This is a dog",
the words are in position - 0,1,2,3 where these are the indices (position) of each word in the sentence. Encoding a position,
means encoding these indices such that the words known their order in the sentence. Without this encoding, we won't know if the word
say "dog" is at position 3 or 2 or 1.

"""

import numpy as np
import tensorflow as tf
from keras.layers import Embedding, Layer

def positional_encoding(length, depth):
    """_summary_
    This function takes in the number of words in a sentence (denoted as length) and the depth of each word (aka dimension referred to
    here as depth). This helps to
    Args:
        length (_type_): _description_
        depth (_type_): _description_

    Returns:
        _type_: _description_
    """
    # depth is the dimension of each position.
    # We divide by 2 since half of our depth which are odd indices are encoded using sin() while rest even half are encoded by cos()
    depth = depth / 2

    # define positions
    positions = np.arange(length)[:,np.newaxis]         # An array of indices (positions) - one per row
    depths = np.arange(depth)[np.newaxis,:] / depth     # this is i / depth / 2 == 2i / depth

    # calculate angle rates and radians
    angleRates = 1 / (10000 ** depths)
    angleRads = positions * angleRates

    posEncoding = np.concatenate([np.sin(angleRads), np.cos(angleRads)], axis=1)

    return tf.cast(posEncoding, dtype=tf.float32)

class PositionalEmbedding(Layer):

    def __init__(self, vocabSize, dModel, maximumPositionEncoding, **kwargs):
        """_summary_
        This class helps in creating positional embedding

        Args:
            vocabSize (int): The size of the input vocabulary which is the tokens which we can have as maximum number
            dModel (int): The dimension of the input which is the constant across outputs
            maximumPositionEncoding (int): The maximum length of the sentence in the dataset
        """
        super().__init__(**kwargs)

        # initialize the embedding layer
        self.embedding = Embedding(input_dim=vocabSize, output_dim=dModel, mask_zero=True)

        # initialze the positional embedding function
        self.posEncoding = positional_encoding(length=maximumPositionEncoding, depth=dModel)

        self.dModel = dModel

    def compute_mask(self, *args, **kwargs):

        return self.embedding.compute_mask(*args, **kwargs)

    def call(self, x):

        # get the length of input sequence
        seqLen = tf.shape(x)[1]

        # embed the input and scale the embeddings
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dModel, tf.float32))

        # add positional encoding with scaled embeddings
        x += self.posEncoding[tf.newaxis, :seqLen, :]

        return x