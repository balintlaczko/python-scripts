# %%
# load dependencies
from datetime import datetime
import itertools
import random
import os
import math
import numpy as np
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from scipy.ndimage.interpolation import shift
# import plaidml.keras
# plaidml.keras.install_backend()
import keras


# %%
# load dataset
dataset = read_csv('elephants.csv')
values = dataset.values
# ensure all data is float
values = values.astype('float32')
values = values[::100]  # subsampling every 100th entry...
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
#np.savetxt("elephants_scaled2.csv", scaled, delimiter=",")
print("Samples in data:", len(scaled))


# %%
# export dataset as coll for Max
def export_dataset_coll():
    target_dir = r"C:\Users\User\Desktop"
    target_file = "elephant_dataset.txt"
    fullname = os.path.join(target_dir, target_file)
    outfile = open(fullname, "w")
    for i in range(len(scaled)):
        outfile.write("%d, %f %f;\n" % (i+1, scaled[i, 0], scaled[i, 1]))
    outfile.close()
    print("Dataset written to", target_file, "\n")

# export_dataset_coll()


# %%
# reframe dataset as timeseries
num_inputs = 6000
seq_len = 50
#num_inputs = len(scaled) - seq_len - 3

dataX = []
dataY = []
for i in range(num_inputs):
    start = np.random.randint(len(scaled)-seq_len-2)
    #start = i
    end = start+seq_len-1
    sequence_in = scaled[start:end+1]
    sequence_out = scaled[end + 1]
    dataX.append(sequence_in)
    dataY.append(sequence_out)
dataX = np.array(dataX)
dataY = np.array(dataY)
print("Created", num_inputs, "input sequences, each", seq_len, "samples long.")


# %%
# design the network
neurons = 32
batches = 16
dropout = 0.2
#reg = keras.regularizers.L1L2(l1=0.01, l2=0.01)
model = keras.models.Sequential()
model.add(keras.layers.LSTM(neurons, dropout=dropout, input_shape=(seq_len, 2)))
model.add(keras.layers.Dense(2))


# %%
# training time!
# load weights
model.load_weights("best_model_elephants.h5")
print("Loaded weights.")
# compile
model.compile(loss='mean_squared_error', optimizer='adam')
print("Compiled.")
losses = []


def plotit(losses, newloss, epoch):
    losses.append(newloss)
    if (epoch+1) % 100 == 0:
        # pyplot.close()
        pyplot.plot(losses, label='epoch '+str(epoch+1))
        pyplot.legend()
        pyplot.show()

# make a prediction after each epoch


def generate_trajectory(dataX, model, steps, epoch, loss):
    # generate trajectory
    generated = []
    initial = dataX[random.randint(0, len(dataX)-1)]
    initial = initial.reshape((1, seq_len, 2))
    for i in range(steps):
        pred = np.clip(model.predict(initial), 0, 1)[0]
        generated.append(pred[0])
        generated.append(pred[1])
        initial = shift(initial[0], [-1, 0], cval=0.)
        initial[-1, 0], initial[-1, 1] = pred
        initial = initial.reshape((1, seq_len, 2))
    # reshape it as a coll, save it to txt
    target_dir = r"C:\Users\User\Desktop\Predictions"
    target_file = "elephant_prediction_%d_%f.txt" % ((epoch+1), loss)
    fullname = os.path.join(target_dir, target_file)
    outfile = open(fullname, "w")
    for i in range(math.floor(len(generated)/2)):
        outfile.write("%d, %f %f;\n" % (i+1, generated[i*2], generated[i*2+1]))
    outfile.close()
    print("         Trajectory written to", target_file, "\n")


def on_epoch_end(epoch, _):
    epoch += 11281  # offset with number of previously completed epochs
    plotit(losses, _["loss"], epoch)
    if (epoch + 1) % 100 == 0:
        print('----- Generating trajectory after Epoch: %d' % (epoch + 1))
        generate_trajectory(dataX, model, 1000, epoch, _["loss"])


# checkpoints
gen = keras.callbacks.LambdaCallback(on_epoch_end=on_epoch_end)
es = keras.callbacks.EarlyStopping(
    monitor='loss', mode='min', verbose=1, patience=500)
best = keras.callbacks.ModelCheckpoint('best_model_elephants.h5', monitor='loss',
                                       mode='min', verbose=1, save_best_only=True)
last = keras.callbacks.ModelCheckpoint('last_model_elephants.h5', monitor='loss',
                                       mode='min', verbose=0, period=1)

# %%
# fit network
history = model.fit(dataX, dataY, epochs=100000, batch_size=batches,
                    verbose=1, shuffle=False, callbacks=[es, best, last, gen])


# %%
# predict trajectory with the new best weights
def generate_with_best(model, epoch, loss):
    print("Reloading best weights...")
    model.load_weights("best_model_elephants.h5")
    print("Generating trajectory...")
    generate_trajectory(dataX, model, 1000, epoch-1, loss)

# predict trajectory with the last weights


def generate_with_last(model, epoch, loss):
    print("Reloading last weights...")
    model.load_weights("last_model_elephants.h5")
    print("Generating trajectory...")
    generate_trajectory(dataX, model, 1000, epoch-1, loss)


# %%
for i in range(10):
    generate_with_best(model, 20000, i)
