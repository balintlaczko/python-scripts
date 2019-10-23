# import the reaper API
from reaper_python import *

# how many tracks are there in the project?
tracksInProject = RPR_CountTracks(0)

# (0=trim/off, 1=read, 2=touch, 3=write, 4=latch)

for i in range(tracksInProject):
    currentTrack = RPR_GetTrack(0, i)
    RPR_SetMediaTrackInfo_Value(currentTrack, "I_AUTOMODE", 1)
