"""
This file defines the various functions we use inside the Transformer architecture
"""
import tensorflow as tf
from keras.layers import Add, Layer, LayerNormalization, MultiHeadAttention

class BaseAttention(Layer):
    """_summary_
    The Base Attention module is the source of all attentions. All other attention modules are sub-classed from this module
    """

    def __init__(self, **kwargs):
        # **kwargs used to initialize the MultiHeadAttention layer for all sub classed modules
        super().__init__()

        # initialize multi head attention layer, layer normalization and addition layer
        self.supports_masking = True
        self.mha = MultiHeadAttention(**kwargs)
        self.layernorm = LayerNormalization()
        self.add = Add()

class CrossAttention(BaseAttention):

    def call(self, x, context, contextMask=None):
        # apply multi head attention to the query and context inputs
        (attentionOutputs, attentionScores) = self.mha(query=x,
                                                       key=context,
                                                       value=context,
                                                       key_mask=contextMask,
                                                       value_mask=contextMask,
                                                       return_attention_scores=True
                                                       )

        # store attention scores for visualization later
        self.lastAttentionScores = attentionScores

        # apply residual connection and layer norm
        x = self.add([x, attentionOutputs])
        x = self.layernorm(x)

        return x

class GlobalSelfAttention(BaseAttention):

    def call(self, x):

        # apply self attention head
        attentionOutputs = self.mha(query=x,
                                    key=x,
                                    value=x
                                    )

        x = self.add([x, attentionOutputs])
        x = self.layernorm(x)

        return x

class CausalSelfAttention(BaseAttention):

    def call(self, x):

        attentionOutputs = self.mha(query=x,
                                    key=x,
                                    value=x,
                                    use_causal_mask=True
                                    )

        x = self.add([x, attentionOutputs])
        x = self.layernorm(x)

        return x