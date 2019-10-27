# python-scripts
## Python scripts for various purposes (deep learning, audio analysis, Reaper scripts) mostly for my own use

Contents:
- Audio Analysis:
  - librosa_sorter.py | sorts/renames audio files in a folder in ascending order based on average spectral flatness
- Deep Learning:
  - gest-LSTM-keras_elephants.py | trains an LSTM network to predict XY coordinates based on a dataset of geo-tracked elephants
  - goldberg_tf_LSTM_stacked_wshift3.py | trains a stacked, stateful LSTM network to predict MIDI pitches based on J.S. Bach's Goldberg Variations 
  - tfaudio_stuff.py | trains a stacked, stateful LSTM network to generate audio based on another audio file (work-in-progress...)
- ReaScripts:
  - bl_armAllAutos.py | arms all automation lines in a project
  - bl_autonameChildsInParent.py | generates names of all child tracks in a parent track (as <parent_name><child_index>)
  - bl_autoRouteToParent.py | routes selected child tracks to consequtive channels of the parent track, unrouting all other (unselected) child tracks
  - bl_changeSourceFiles.py | changes the source files of all items on a selected track (useful for batch processing/swapping)
  - bl_parentSendSort.py | routes all child tracks in a selected parent track to consequtive  channels
  - bl_resetParentSends.py | resets the routing of child tracks inside a parent track to the defaults
  - bl_generateMcfxConvConf.py | generates .conf files for an mcfx_convolver in order to use it as a HOA IR reverb
  

