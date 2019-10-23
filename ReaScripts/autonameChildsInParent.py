# import the reaper API
from reaper_python import *
# how many selected tracks are there?
numSelected = RPR_CountSelectedTracks(0)
# make sure that only one track is selected
if numSelected != 1:
    RPR_ShowConsoleMsg("Select ONLY the folder track!")
else: # if only one track is selected
    parentTrack = RPR_GetSelectedTrack(0, 0) # assuming that we selected the parent track
    listOfChilds = []
    # how many tracks are there in the project?
    tracksInProject = RPR_CountTracks(0)
    # iterate through the whole project and ask every track:
    for i in range(tracksInProject):
        currentTrack = RPR_GetTrack(0, i)
        if RPR_GetParentTrack(currentTrack) == parentTrack: #is this your parent?
            listOfChilds.append(currentTrack) #if yes, I put you on my list
    # now we count how many childs our parent track has
    numChilds = len(listOfChilds)
    # if our "parent" track doesn't have any childs, then we probably selected a child 
    # instead of the parent, so let's notify the user, and select the "real" parent track
    if numChilds == 0:
        theRealParent = RPR_GetParentTrack(parentTrack)
        RPR_ShowConsoleMsg("This track is probably a child track. I select the folder track for you.")
        for i in range(tracksInProject):
            currentTrack = RPR_GetTrack(0, i)
            RPR_SetMediaTrackInfo_Value(currentTrack, "I_SELECTED", 0)
        RPR_SetMediaTrackInfo_Value(theRealParent, "I_SELECTED", 1)
    else: # if it's really a parent track...
        # what's the name of our parent track?
        parentName = RPR_GetSetMediaTrackInfo_String(parentTrack, "P_NAME", "", 0)[3]
        # set the names of all childs
        for i in range(numChilds):
            currentTrack = listOfChilds[i]
            currentName = parentName + str(i + 1)
            RPR_GetSetMediaTrackInfo_String(currentTrack, "P_NAME", currentName, 1)