"""
This file helps us define configurations which we plan to use in our architecture

"""
from config import ENGLISH_FRENCH_DATASET

# Name of the dataset text file
DATA_FNAME = f"{ENGLISH_FRENCH_DATASET}/fra.txt"

# define the batch size of our data.
BATCH_SIZE = 96

# Define vocabulary size of source and target text processors
# lets our text vectorization layer know amount of vocabulary to generate
# vocabulary are the number of tokens which are the smallest unit of vocabulary
# for example - word "unbelivable" can be tokens - "un", "beliv" and "able" and these are part of the vocabulary
SOURCE_VOCAB_SIZE = 15_000
TARGET_VOCAB_SIZE = 15_000

# Positional encoding dimension
MAX_POS_ENCODING = 2048

# number of encoder and decoder blocks
ENCODER_NUM_LAYERS = 6
DECODER_NUM_LAYERS = 6

# define model dimension aka the dimension of intermediate outputs
# essentially, when we input the number of tokens and the dimension of those token vectors, we use this
# it is the D_x (dimension of X) for an input X of shape (N_x, D_x)
D_MODEL = 512

# Feed Forward network dimension
# The FFN block that resides after the MHA block is a 2 layer MLP. This paramter tells the dimension of the hidden layer
DFF = 2048

# Define the number of heads of the Multi-Head attention.
# This splits the dimension so all tokens are processed in parallel but with different Q, K and V matrices
NUM_HEADS = 8

# define other training parameters like epochs, drop_rate, output directory
DROP_RATE = 0.1
EPOCHS = 25
OUTPUT_DIR = "output"







