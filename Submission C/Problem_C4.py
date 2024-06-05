# =====================================================================================================
# PROBLEM C4
#
# Build and train a classifier for the sarcasm dataset.
# The classifier should have a final layer with 1 neuron activated by sigmoid.
#
# Do not use lambda layers in your model.
#
# Dataset used in this problem is built by Rishabh Misra (https://rishabhmisra.github.io/publications).
#
# Desired accuracy and validation_accuracy > 75%
# =======================================================================================================

import json
import tensorflow as tf
import numpy as np
import urllib
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('acc') > 0.76 and logs.get('val_acc') > 0.76):
            print("\nTarget has accomplished, mission success")
            self.model.stop_training = True

def solution_C4():
    data_url = 'https://github.com/dicodingacademy/assets/raw/main/Simulation/machine_learning/sarcasm.json'
    urllib.request.urlretrieve(data_url, 'sarcasm.json')

    # DO NOT CHANGE THIS CODE
    # Make sure you used all of these parameters or test may fail
    vocab_size = 1000
    embedding_dim = 16
    max_length = 120
    trunc_type = 'post'
    padding_type = 'post'
    oov_tok = "<OOV>"
    training_size = 20000

    sentences = []
    labels = []

    with open("./sarcasm.json", 'r') as f:
        datastore = json.load(f)

    for item in datastore:
        sentences.append(item['headline'])
        labels.append(item['is_sarcastic'])

    # YOUR CODE HERE
    training_sentences, validation_sentences = sentences[: training_size], sentences[training_size:]
    training_labels, validation_labels = labels[: training_size], labels[training_size:]

    # Fit your tokenizer with training data
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(training_sentences)

    train_sequences = tokenizer.texts_to_sequences(training_sentences)
    train_padded = pad_sequences(train_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

    valid_sequences = tokenizer.texts_to_sequences(validation_sentences)
    valid_padded = pad_sequences(valid_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

    training_labels = np.array(training_labels)
    validation_labels = np.array(validation_labels)

    model = tf.keras.Sequential([
        # YOUR CODE HERE. DO not change the last layer or test may fail
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    callback = myCallback()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

    model.fit(train_padded, training_labels, validation_data=(valid_padded, validation_labels),
              verbose=2, epochs=100, callbacks=callback)
    return model


# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model = solution_C4()
    model.save("model_C4.h5")
