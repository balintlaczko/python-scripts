# import the reaper API
from reaper_python import *
# how many selected tracks are there?
numSelected = RPR_CountSelectedTracks(0)
# make sure that only one track is selected
if numSelected != 1:
    RPR_ShowConsoleMsg("Select ONLY the folder track!")
else: # if only one track is selected
    parentTrack = RPR_GetSelectedTrack(0, 0) # assuming that we selected the parent track
    # how many receives does the parent track have now?
    parentNumReceives = RPR_GetTrackNumSends(parentTrack, -1)
    # delete all of them
    for k in range(parentNumReceives):
        RPR_RemoveTrackSend(parentTrack, -1, 0)
    # set it (back) to have 2 channels
    RPR_SetMediaTrackInfo_Value(parentTrack, "I_NCHAN", 2)
    # cycle through the tracks of the project. any child of this parent will be set to have
    # parent send on the parent's first 2 channels
    numChilds = 0
    tracksInProject = RPR_CountTracks(0)
    for i in range(tracksInProject):
        currentTrack = RPR_GetTrack(0, i)
        if RPR_GetParentTrack(currentTrack) == parentTrack:
            RPR_SetMediaTrackInfo_Value(currentTrack, "B_MAINSEND", 1)
            RPR_SetMediaTrackInfo_Value(currentTrack, "C_MAINSEND_OFFS", 0)
            numChilds += 1
    # if our "parent" track doesn't have any childs, then we probably selected a child 
    # instead of the parent, so let's notify the user, and select the "real" parent track
    if numChilds == 0:
        theRealParent = RPR_GetParentTrack(parentTrack)
        RPR_ShowConsoleMsg("This track is probably a child track. I select the folder track for you.")
        for i in range(tracksInProject):
            currentTrack = RPR_GetTrack(0, i)
            RPR_SetMediaTrackInfo_Value(currentTrack, "I_SELECTED", 0)
        RPR_SetMediaTrackInfo_Value(theRealParent, "I_SELECTED", 1)