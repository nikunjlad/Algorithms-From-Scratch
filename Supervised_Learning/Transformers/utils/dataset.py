"""
This file helps us prepare the dataset

"""

import random
import tensorflow as tf
import tensorflow_text as tf_text

# define a model level autotune
_AUTO = tf.data.AUTOTUNE

def load_data(fname):
    """_summary_
    This function takes in the dataset file as input which contains source and target sentence as inputs. It parses it
    and returns a list of source and target sentences as parsed from the dataset files. Dataset file has extra metadata which is
    stripped as part of the parsing operation.

    Args:
        fname (string): The path to filename which is the dataset containing source sentence and target translation pairs

    Returns:
        tuple(list, list): The list of source and target sentences parsed from the input dataset file
    """
    with open(fname, "r", encoding="utf-8") as textFile:
        # the source and the target sentence is demarcated with tab,
        # iterate over each line and split the sentences to get
        # the individual source and target sentence pairs
        lines = textFile.readlines()
        pairs = [line.split("\t")[:-1] for line in lines]
        # randomly shuffle the pairs
        random.shuffle(pairs)
        # collect the source sentences and target sentences into respective lists
        source = [src for src, _ in pairs]      # list of source sentences
        target = [trgt for _, trgt in pairs]    # list of target sentences

    return (source, target)

def splitting_dataset(source, target):
    """_summary_
    This function takes in a source and target sentences list and splits them into training, validation and test splits

    Args:
        source (list): A list of strings containing the source sentences to be translated
        target (list): A list of strings containing the target sentences which are translations of the source

    Returns:
        tuple(tuple): A tuple of tuples containing training, validation and test splits of the source and target lists
    """
    # calculate the training and validation size
    trainSize = int(len(source) * 0.8)      #
    valSize = int(len(source) * 0.1)
    # split the inputs into train, val, and test
    (trainSource, trainTarget) = (source[:trainSize], target[:trainSize])
    (valSource, valTarget) = (source[trainSize : trainSize + valSize],
                              target[trainSize : trainSize + valSize])
    (testSource, testTarget) = (source[trainSize + valSize :],
                                target[trainSize + valSize :])
    # return the splits
    return ((trainSource, trainTarget),
            (valSource, valTarget),
            (testSource, testTarget))

def make_dataset(splits, batchSize, sourceTextProcessor, targetTextProcessor, train=False):
    """_summary_
    This function takes the source and target sentence lists based on the splits we provide and converts them into a TF dataset
    The dataset is batched using batchSize and pre-processed using the source and target text processor
    Additionally, training split is shuffled for training

    Args:
        splits (tuple): A (source, target) tuple holding list of sentences for the splits - training, validation and testing
        batchSize (int): The integer to use for batching our data
        sourceTextProcessor (object): A function that we use to apply on our source list of sentences
        targetTextProcessor (object): A function that we use to apply on our target list of sentences
        train (bool, optional): A flag if set to true indicates this function is being used to process training split. Defaults to False.

    Returns:
        tf.data.Dataset: A Tensorflow dataset pipeline generated for the split provided
    """
    # parse the split into source and target lists of sentences
    (source, target) = splits

    # build a tensorflow dataset
    dataset = tf.data.Dataset.from_tensor_slices((source, target))

    # given a source and target list, pre-process the sentences using the Text Processors
    # The decoder takes in targetInputs and targetOutputs. We derive them from the targetBuffer
    # targetInputs are all tokens (words) without last token
    # targetOutputs are all tokens (words) starting from 1st token to last.
    # recall we map tokens to next words, hence a shift right operation
    def prepare_batch(source, target):
        source = sourceTextProcessor(source)
        targetBuffer = targetTextProcessor(target)
        targetInput = targetBuffer[:, :-1]
        targetOutput = targetBuffer[:, 1:]
        return (source, targetInput), targetOutput

    # if training dataset, shuffle batch and prefetch it
    if train:
        dataset = (
            dataset.shuffle(dataset.cardinality())
            .batch(batchSize)
            .map(prepare_batch, _AUTO)
            .prefetch(_AUTO)
        )
    else:
        dataset = dataset.batch(batchSize).map(prepare_batch,_AUTO).prefetch(_AUTO)

    return dataset

def tf_lower_and_split_punct(text):
    """_summary_
    This function takes text and pre-processes it while preserving the semantic meaning of it. This includes -
    perfoming NFKD processing, converting text to lower case, preserving only specfic punctuations and sanitizing against unwanted
    emojis and characters, adding spaces around punctuations and wrapping sentences within START and END markers

    Args:
        text (string): The sentence to be processed

    Returns:
        string: The processed sentence to be used for training the model

    """
    
    # split accented characters
    # essentially, it is a way to preserve the accent marks that are present in languages like French
    # without this, the text would not have accent and will miss out on critical language specific features
    text = tf_text.normalize_utf8(text, "NFKD")     # NFKD - Normalization Form Compatibility Decomposition
    text = tf.strings.lower(text)                   # convert text into lower case

    # keep space, a to z, and selected punctuations (periods, question marks, exclamation marks and commas)
    text = tf.strings.regex_replace(text, "[^ a-z.?!,]", "")       # keep lower case letters, space, some punctuations
    # add spaces around punctuation
    text = tf.strings.regex_replace(text, "[.?!,]", r" \0 ")       # after punctuations add spaces

    # strip whitespace and add [START] and [END] tokens
    text = tf.strings.strip(text)       # remove unwanted white spaces at the edges of the sentences (start and end)
    text = tf.strings.join(["[START]", text, "[END]"], separator=" ")       # wrap the sentences with [START] and [END] markers

    # return the processed text
    return text



