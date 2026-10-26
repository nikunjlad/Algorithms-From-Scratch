"""
Transformers Training Script

"""
import os

# 0 = all logs, 1 = filter INFO, 2 = filter WARNING, 3 = filter ERROR
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Turn off oneDNN floating point warnings if desired
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import sys
import logging
import tensorflow as tf
import keras

tf.keras.utils.set_random_seed(42)
from keras.layers import TextVectorization
from keras.optimizers import Adam
from utils import config
from utils.transformer import Transformer
from utils.translator import Translator
from utils.rate_schedule import RateScheduler
from utils.loss_accuracy import masked_accuracy, masked_loss
from utils.dataset import load_data, make_dataset, splitting_dataset, tf_lower_and_split_punct

gpus = tf.config.list_physical_devices('GPU')
print(gpus)

def configure_logger():
    """_summary_
    This function helps to configure logger for logging to the console with the minimum log level to be DEBUG

    Returns:
        logging.logger: The logger that is configured with a stream handler which prints the log-level and message to the terminal
    """
    logger = logging.getLogger("TransformerLogger")     # configure logger
    logger.setLevel(logging.DEBUG)      # Set lowest level to capture all logs

    # 2. Create a console handler and direct it to standard output
    console_handler = logging.StreamHandler(sys.stdout)     # stream the logs to the console
    console_handler.setLevel(logging.DEBUG)

    # 3. Define a clean, scannable format: [Time] [Level] Message
    log_format = logging.Formatter(fmt="[%(levelname)s] %(message)s")
    console_handler.setFormatter(log_format)

    # 4. Add the handler to your logger
    logger.addHandler(console_handler)

    return logger


def main():

    # configure logger
    logger = configure_logger()

    # load the data
    logger.info(f"Loading the data from {config.DATA_FNAME}")
    (source, target) = load_data(fname=config.DATA_FNAME)

    # split the datasets in train, val and test
    logger.info(f"Splitting the dataset into train, val and test")
    (train, val, test) = splitting_dataset(source=source, target=target)

    # Configure text processor for source sentences - pre-processes texts and sets the max tokens for our dataset
    logger.info(f"Configuring Text processor for the source sentences")
    sourceTextProcessor = TextVectorization(standardize=tf_lower_and_split_punct, max_tokens=config.TARGET_VOCAB_SIZE)
    sourceTextProcessor.adapt(train[0])     # adapt the text processor for the training sources

    # Configure text processor for the target sentences
    logger.info(f"Configuring Text processor for the target sentences")
    targetTextProcessor = TextVectorization(standardize=tf_lower_and_split_punct, max_tokens=config.TARGET_VOCAB_SIZE)
    targetTextProcessor.adapt(train[1])     # adapt the text processor for the training targets

    # Build the training, vlidation and test data pipeline
    logger.info(f"Building Tensorflow Data input pipeline")
    trainDs = make_dataset(splits=train, batchSize=config.BATCH_SIZE, train=True,
                           sourceTextProcessor=sourceTextProcessor, targetTextProcessor=targetTextProcessor)
    valDs = make_dataset(splits=val, batchSize=config.BATCH_SIZE, train=False,
                         sourceTextProcessor=sourceTextProcessor, targetTextProcessor=targetTextProcessor)
    testDs = make_dataset(splits=test, batchSize=config.BATCH_SIZE, train=False,
                          sourceTextProcessor=sourceTextProcessor, targetTextProcessor=targetTextProcessor)

    logger.info(f"Building the Transformer Model")
    transformerModel = Transformer(
        encNumLayers=config.ENCODER_NUM_LAYERS,
        decNumLayers=config.DECODER_NUM_LAYERS,
        dModel=config.D_MODEL,
        numHeads=config.NUM_HEADS,
        dff=config.DFF,
        sourceVocabSize=config.SOURCE_VOCAB_SIZE,
        targetVocabSize=config.TARGET_VOCAB_SIZE,
        maximumPositionEncoding=config.MAX_POS_ENCODING,
        dropOutRate=config.DROP_RATE,
    )

    logger.info(f"Compiling the Transformer model")
    learningRate = RateScheduler(dModel=config.D_MODEL)
    optimizer = Adam(learning_rate=learningRate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    transformerModel.compile(loss=masked_loss, optimizer=optimizer, metrics=[masked_accuracy], jit_compile=False)

    transformerModel.fit(trainDs, epochs=config.EPOCHS, validation_data=valDs)

    # infer on a sentence
    translator = Translator(
        sourceTextProcessor=sourceTextProcessor,
        targetTextProcessor=targetTextProcessor,
        transformer=transformerModel,
        maxLength=50,
    )

    logger.info(f"Serialize the inference translator to disk")
    tf.saved_model.save(
        obj=translator,
        export_dir="translator",
    )

if __name__ == "__main__":
    main()
    sys.exit(0)
