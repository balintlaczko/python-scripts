# %%
# load dependencies

import tensorflow as tf
import numpy as np
import os
import time
import pandas as pd

tf.enable_eager_execution()

# %%
# def to get pitches from a csv


def get_pitches_from_csv(csvdir, num):
    # zero-padded number of the variation
    _chosen_str = "%02d" % num
    # corresponding csv file
    _csv_filename = "988-v%s.csv" % _chosen_str
    # the full path for it
    _csv_path = os.path.join(csvdir, _csv_filename)
    # read it
    _csv = pd.read_csv(_csv_path, error_bad_lines=False, warn_bad_lines=False)
    # name the columns
    _csv.columns = ["Track", "Time", "Header", "Manual", "Pitch", "Velocity"]
    # get the notes (only note-on MIDI values from the right hand)
    _notes = _csv[(_csv['Header'] == " Note_on_c") & (
        _csv["Velocity"] != 0) & (_csv["Manual"] == " 0")]
    # take the "pitch" column
    _pitches = _notes[['Pitch']]
    _pitches = _pitches.values.astype('int')
    return _pitches.flatten()

# %%
# def to get pitches from all input files


def get_pitches_from_all(num_files, filesdir):
    _all_pitches = []
    _all_pitcharrays = []
    # iterate through all csv files
    for i in range(1, num_files+1, 1):
        # zero-padded number of the variation
        _chosen_str = "%02d" % i
        # corresponding csv file
        _csv_filename = "988-v%s.csv" % _chosen_str
        # the full path for it
        _csv_path = os.path.join(filesdir, _csv_filename)
        # read it
        _csv = pd.read_csv(_csv_path, error_bad_lines=False,
                           warn_bad_lines=False)
        # name the columns
        _csv.columns = ["Track", "Time", "Header",
                        "Manual", "Pitch", "Velocity"]
        # get the notes (only note-on MIDI values from the right hand)
        _notes = _csv[(_csv['Header'] == " Note_on_c") & (
            _csv["Velocity"] != 0) & (_csv["Manual"] == " 0")]
        # take the "pitch" column
        _pitches = _notes[['Pitch']]
        # convert them to floats
        _pitches = _pitches.values.astype('int')
        _pitches = _pitches.flatten()
        # add them as an array to a collection
        _all_pitcharrays.append(_pitches)
        # and also the individual pitch values to a flattened array (will be used for scaling)
        for k in range(len(_pitches)):
            _all_pitches.append(_pitches[k])

    return _all_pitches, _all_pitcharrays


# %%
# def to get pitches from all input files (all manuals!)


def get_pitches_from_all_allmanuals(num_files, filesdir):
    _all_pitches = []
    _all_pitcharrays = []
    # iterate through all csv files
    for i in range(1, num_files+1, 1):
        # zero-padded number of the variation
        _chosen_str = "%02d" % i
        # corresponding csv file
        _csv_filename = "988-v%s.csv" % _chosen_str
        # the full path for it
        _csv_path = os.path.join(filesdir, _csv_filename)
        # read it
        _csv = pd.read_csv(_csv_path, error_bad_lines=False,
                           warn_bad_lines=False)
        # name the columns
        _csv.columns = ["Track", "Time", "Header",
                        "Manual", "Pitch", "Velocity"]
        # get the notes (only note-on MIDI values from the right hand)
        _notes = _csv[(_csv['Header'] == " Note_on_c") & (
            _csv["Velocity"] != 0)]  # & (_csv["Manual"] == " 0")]
        # take the "pitch" column
        _pitches = _notes[['Pitch']]
        # convert them to floats
        _pitches = _pitches.values.astype('int')
        _pitches = _pitches.flatten()
        # add them as an array to a collection
        _all_pitcharrays.append(_pitches)
        # and also the individual pitch values to a flattened array (will be used for scaling)
        for k in range(len(_pitches)):
            _all_pitches.append(_pitches[k])

    return _all_pitches, _all_pitcharrays


# %%
# def to split a sequence into inputs and targets

def split_input_target(chunk):
    input_pitches = chunk[:-1]
    target_pitches = chunk[1:]
    return input_pitches, target_pitches

# %%
# def to make the dataset


def make_dataset(sqlen, filedir):
    print("Building dataset...")
    _pitches, _pitcharray = get_pitches_from_all_allmanuals(30, filedir)

    _pitch_lowest = min(_pitches)
    _pitch_highest = max(_pitches)
    _pitch_vocab = sorted(set(_pitches))
    _pitch_newmax = max(_pitches - _pitch_lowest)

    _pitch_dataset = tf.data.Dataset.from_tensor_slices(
        (_pitcharray[0] - _pitch_lowest))

    _sequences = _pitch_dataset.batch(sqlen+1, drop_remainder=True)

    _dataset = _sequences.map(split_input_target)

    print("Sequences:", len(list(_dataset)))

    for i in range(1, len(_pitcharray), 1):
        _temp_pitch_dataset = tf.data.Dataset.from_tensor_slices(
            (_pitcharray[i] - _pitch_lowest))
        _temp_sequences = _temp_pitch_dataset.batch(
            sqlen+1, drop_remainder=True)
        _temp_dataset = _temp_sequences.map(split_input_target)
        _dataset = _dataset.concatenate(_temp_dataset)
        print("Sequences:", len(list(_dataset)))

    print("")
    return _pitch_lowest, _pitch_vocab, _pitch_newmax, _dataset


# %%
# def to make the dataset with overlaps

# the idea with this is to fix the issue that with large sqlen parameters,
# we potentially discard (since drop_remainder=True) up to sqlen pitches
# ...which is a lot when for example there are 100 pitches (like in a slow
# movement), sqlen=64, so we discard the last 36 pitches which is more than
# a third of the entire data... Question with this method whether it is
# problematic (or beneficial?) introduce redundancy with a lot of shifted
# windowed sequences. My guess is that this could be a good trick when
# there is only a very small dataset available, and it's absolutely
# unnecessary when there is enough data in the first place. We'll see.

# UPDADTE: in this file, it's done more extremely, with a shifting window loop


def make_dataset_w_overlaps(sqlen, filedir):
    print("Building dataset...")

    _pitches, _pitcharray = get_pitches_from_all_allmanuals(30, filedir)

    _pitch_lowest = min(_pitches)
    _pitch_highest = max(_pitches)
    _pitch_vocab = sorted(set(_pitches))
    _pitch_newmax = max(_pitches - _pitch_lowest)

    _pitch_dataset = tf.data.Dataset.from_tensor_slices(
        (_pitcharray[0] - _pitch_lowest))

    _remainder = len(_pitcharray[0]) % (sqlen+1)

    for j in range(1, _remainder+1, 1):

        _offset_pitch_dataset = tf.data.Dataset.from_tensor_slices(
            (_pitcharray[0][j:] - _pitch_lowest))

        _pitch_dataset = _pitch_dataset.concatenate(_offset_pitch_dataset)

    _sequences = _pitch_dataset.batch(sqlen+1, drop_remainder=True)

    _dataset = _sequences.map(split_input_target)

    print("Sequences:", len(list(_dataset)))

    for i in range(1, len(_pitcharray), 1):
        _temp_pitch_dataset = tf.data.Dataset.from_tensor_slices(
            (_pitcharray[i] - _pitch_lowest))

        _remainder = len(_pitcharray[i]) % (sqlen+1)

        for j in range(1, _remainder+1, 1):

            _offset_temp_pitch_dataset = tf.data.Dataset.from_tensor_slices(
                (_pitcharray[i][j:] - _pitch_lowest))

            _temp_pitch_dataset = _temp_pitch_dataset.concatenate(
                _offset_temp_pitch_dataset)

        _temp_sequences = _temp_pitch_dataset.batch(
            sqlen+1, drop_remainder=True)

        _temp_dataset = _temp_sequences.map(split_input_target)

        _dataset = _dataset.concatenate(_temp_dataset)

        print("Sequences:", len(list(_dataset)))

    print("")

    return _pitch_lowest, _pitch_vocab, _pitch_newmax, _dataset


# %%
# make the dataset
seq_length = 64
csv_directory = "C:/Users/NOTAM/Desktop/Balint/goldberg/Goldberg-midicsv"

pitch_lowest, pitch_vocab, pitch_newmax, dataset = make_dataset_w_overlaps(
    seq_length, csv_directory)

# %%
# shuffle and chunk dataset

# Batch size
BATCH_SIZE = 64

# Buffer size to shuffle the dataset
# (TF data is designed to work with possibly infinite sequences,
# so it doesn't attempt to shuffle the entire sequence in memory. Instead,
# it maintains a buffer in which it shuffles elements).
BUFFER_SIZE = 100000

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)


# %%
# build network

# Length of the vocabulary
vocab_for_net = pitch_newmax + 1

# The embedding dimension
embedding_dim = 256

# Number of RNN units
rnn_units = 256


def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                  batch_input_shape=[batch_size, None]),
        tf.keras.layers.CuDNNLSTM(rnn_units,
                                  return_sequences=True,
                                  stateful=True,
                                  recurrent_initializer='glorot_uniform'),
        tf.keras.layers.CuDNNLSTM(rnn_units,
                                  return_sequences=True,
                                  stateful=True,
                                  recurrent_initializer='glorot_uniform'),
        tf.keras.layers.CuDNNLSTM(rnn_units,
                                  return_sequences=True,
                                  stateful=True,
                                  recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model


model = build_model(
    vocab_size=vocab_for_net,
    embedding_dim=embedding_dim,
    rnn_units=rnn_units,
    batch_size=BATCH_SIZE)

model.summary()

# %%
# def for the loss function


def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)


# %%
# compile the model

# model.load_weights("best_goldberg_tf_stacked_wshift3.h5")
# print("Loaded weights.")
model.compile(optimizer='adam', loss=loss)
print("Compiled.")

# %%
# create checkpoints

# Directory where the checkpoints will be saved
checkpoint_dir = './training_checkpoints_lstm'
logdir = r"C:\Users\NOTAM\Desktop\Balint\tensorboard_logs\stacked_wshift3\v1"
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix)

best = tf.keras.callbacks.ModelCheckpoint('best_goldberg_tf_stacked_wshift3.h5', monitor='loss',
                                          mode='min', verbose=1, save_best_only=True)
last = tf.keras.callbacks.ModelCheckpoint('last_goldberg_tf_stacked_wshift3.h5', monitor='loss',
                                          mode='min', verbose=0, save_freq=1)

earlystopping = tf.keras.callbacks.EarlyStopping(
    monitor='loss', patience=30)

tensorboard = tf.keras.callbacks.TensorBoard(log_dir=logdir)


# %%
# train!

EPOCHS = 1000

history = model.fit(dataset, epochs=EPOCHS, callbacks=[
                    best, earlystopping, tensorboard])
# NB saving last model creates a huge bottleneck

# %%
# make a new model for generating stuff

model2play = build_model(vocab_for_net, embedding_dim, rnn_units, batch_size=1)

model2play.load_weights("best_goldberg_tf_stacked_wshift3.h5")

model2play.build(tf.TensorShape([1, None]))

model2play.summary()

# %%
# def to generate melody


def generate_melody(model, start_pitches, lowest, melody_length, temper):

    start_pitches = [np.random.randint(start_pitches[0], start_pitches[1])]

    # Converting our start string to numbers (vectorizing)
    input_eval = start_pitches - lowest
    input_eval = tf.expand_dims(input_eval, 0)

    # Empty string to store our results
    melody_generated = []

    melody_generated.append(start_pitches[0] - lowest)

    # Low temperatures results in more predictable text.
    # Higher temperatures results in more surprising text.
    # Experiment to find the best setting.
    temperature = temper

    # Here batch size == 1
    model.reset_states()
    for i in range(melody_length):
        predictions = model(input_eval)
        # remove the batch dimension
        predictions = tf.squeeze(predictions, 0)

        # using a categorical distribution to predict the word returned by the model
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(
            predictions, num_samples=1)[-1, 0].numpy()

        # We pass the predicted word as the next input to the model
        # along with the previous hidden state
        input_eval = tf.expand_dims([predicted_id], 0)

        melody_generated.append(predicted_id)

    return melody_generated + lowest


# %%
# generate a new melody


def spit_out_txt(stimuli, lines, linelength, tempr):
    preds = [generate_melody(model2play, start_pitches=stimuli,
                             lowest=pitch_lowest, melody_length=linelength, temper=tempr) for i in range(lines)]
    target_dir = r"C:\Users\NOTAM\Desktop\Balint\goldberg\Predictions\tf_stacked"
    target_file = "goldberg_predictions_tf_stacked_wshift3_rand%d-%d_temp%f.txt" % (
        stimuli[0], stimuli[1], tempr)
    fullname = os.path.join(target_dir, target_file)
    outfile = open(fullname, "w")
    for i in range(len(preds)):
        for j in range(len(preds[i])):
            outfile.write("%d " % preds[i][j])
        outfile.write("\n")
    outfile.close()
    print("Melodies were written to", target_file, "\n")


# %%
spit_out_txt([55, 79], 100, 128, 1.0)
spit_out_txt([55, 79], 100, 128, 1.25)
spit_out_txt([55, 79], 100, 128, 1.5)
spit_out_txt([55, 79], 100, 128, 1.75)
spit_out_txt([55, 79], 100, 128, 2.0)


# %%
