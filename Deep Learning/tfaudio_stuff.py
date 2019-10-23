# %%
import tensorflow as tf
tf.enable_eager_execution()

# %%

filename = "C:/Users/User/Desktop/stretched_stereo.wav"

raw_audio = tf.io.read_file(filename)
waveform = tf.audio.decode_wav(raw_audio)
sr = waveform[1].numpy()

# %%
print("My cool samples:", waveform[0].numpy())
print("My cool samplerate:", sr)


# %%
dataset = tf.data.Dataset.from_tensor_slices(waveform[0])

# %%
# def to split a sequence into inputs and targets


def split_input_target(chunk):
    input_samples = chunk[:-1]
    target_samples = chunk[1:]
    return input_samples, target_samples


# %%

seq_length = sr / 100

sequences = dataset.batch(seq_length+1, drop_remainder=True)

dataset = sequences.map(split_input_target)

# %%
for input_example, target_example in dataset.take(1):
    print('Input data: ', input_example.numpy())
    print('Target data:', target_example.numpy())


# %%
# Batch size
BATCH_SIZE = 100

# Buffer size to shuffle the dataset
# (TF data is designed to work with possibly infinite sequences,
# so it doesn't attempt to shuffle the entire sequence in memory. Instead,
# it maintains a buffer in which it shuffles elements).
BUFFER_SIZE = 1000000

#dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
dataset = dataset.cache().shuffle(BUFFER_SIZE).batch(
    BATCH_SIZE, drop_remainder=True)  # .repeat()

# %%
dataset

# %%
# build network


# Number of RNN units
rnn_units = 256


def build_model(sqlen, feats, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.CuDNNLSTM(rnn_units,
                                  batch_input_shape=(
                                      BATCH_SIZE, sqlen, feats),
                                  return_sequences=True,
                                  stateful=True,
                                  recurrent_initializer='glorot_uniform'),
        tf.keras.layers.CuDNNLSTM(rnn_units,
                                  batch_input_shape=(
                                      BATCH_SIZE, sqlen, feats),
                                  return_sequences=True,
                                  stateful=True,
                                  recurrent_initializer='glorot_uniform'),
        tf.keras.layers.CuDNNLSTM(rnn_units,
                                  batch_input_shape=(
                                      BATCH_SIZE, sqlen, feats),
                                  return_sequences=True,
                                  stateful=True,
                                  recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(feats)
    ])
    return model


model = build_model(
    sqlen=seq_length,
    feats=waveform[0].numpy().shape[1],
    rnn_units=rnn_units,
    batch_size=BATCH_SIZE)

model.summary()

# %%
# compile the model
model.compile(optimizer='adam', loss='mean_squared_error')
print("Compiled.")

# %%
# checkpoints

logdir = r"C:\Users\NOTAM\Desktop\Balint\tensorboard_logs\tfaudio\v1"

best = tf.keras.callbacks.ModelCheckpoint('best_tfaudio.h5', monitor='loss',
                                          mode='min', verbose=1, save_best_only=True)

earlystopping = tf.keras.callbacks.EarlyStopping(
    monitor='loss', patience=30)

tensorboard = tf.keras.callbacks.TensorBoard(log_dir=logdir)

# %%
# train!
EPOCHS = 1000

history = model.fit(dataset, epochs=EPOCHS, callbacks=[
                    best, earlystopping, tensorboard])
