
# %%
# Imports and init
import numpy as np
import librosa
import os
import os.path

mean_flatness_array = []
d_files_and_means = {}
sound_folder = "C:/Users/User/Desktop/Librosa_sorter/Audio_files"

print("Initialized.")

# %%
# Collect all WAVE files from folder
wavfiles_wtree = []
for dirpath, dirnames, filenames in os.walk(sound_folder):
    for filename in [f for f in filenames if f.endswith(".wav")]:
        wavfiles_wtree.append(str(os.path.join(dirpath, filename)))
wavfiles = []
for i, fn in enumerate(wavfiles_wtree):
    path, fil = os.path.split(fn)
    wavfiles.append(fil)
num_wavfiles = len(wavfiles)
print("Found %i .wav files in this folder." % num_wavfiles)

# %%
# Get mean spectral flatness of all files
print("Measuring spectral flatness in all files...")
for i, filename in enumerate(wavfiles_wtree):
    # Load file
    path = filename
    y, sr = librosa.load(path, sr=None, mono=False)
    #print("Loaded", filename)
    # Convert to mono (if necessary)
    if (y.shape[0] > 1):
        #print("File has {} channels, converting to mono...".format(y.shape[0]))
        y_mono = librosa.to_mono(y)
    else:
        pass
        #print("File is mono.")
    # Calculate spectral flatness
    #print("Calculating spectral flatness...")
    flatness = librosa.feature.spectral_flatness(y=y_mono)
    # Save mean value with filename
    #print("Saving mean value...")
    mean_flatness_array.append(np.mean(flatness))
    d_files_and_means[mean_flatness_array[i]] = wavfiles[i]
print("Done.")

# %%
#print("Mean flatnesses:", mean_flatness_array)
mean_flatness_array = np.sort(mean_flatness_array)
#print("Sorted:", mean_flatness_array)
print("Sorted all values into ascending order.")

# %%
# Rename files based on spectral flatness order
print("Renaming files based on order...")
for i, meanflatness in enumerate(mean_flatness_array):
    # get the corresponding file from the dictionary
    corresponding_file = d_files_and_means[meanflatness]
    # the new prefix
    prefix = str(i).zfill(2) + "_"
    #print("The new prefix:", prefix)
    # check if it has already a prefix (from a previous run)
    try:
        prev_prefix = int(corresponding_file[0:2])
        #print("Previous prefix found:", prev_prefix)
        newname = prefix + corresponding_file[3:]
    except:
        #print("No previous prefix found.")
        newname = prefix + corresponding_file

    #print("The new name will be:", newname)

    # and rename it
    os.rename(os.path.join(sound_folder, corresponding_file),
              os.path.join(sound_folder, newname))
print("Finished!")
# %%
