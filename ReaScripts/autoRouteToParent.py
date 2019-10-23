# import the reaper API
from reaper_python import *
# how many selected tracks are there?
selTracks = RPR_CountSelectedTracks(0)
# which one is the first selected track?
firstSelTrack = RPR_GetSelectedTrack(0, 0)
# which is the parent track of that one?
parentTrack = RPR_GetParentTrack(firstSelTrack)
# how many receives does the parent track have now?
parentNumReceives = RPR_GetTrackNumSends(parentTrack, -1)
# delete all of them
for k in range(parentNumReceives):
    RPR_RemoveTrackSend(parentTrack, -1, 0)
# how many tracks are there in the project?
tracksInProject = RPR_CountTracks(0)
# let's just make an empty list, we will fill it up later
listOfChilds = []
# unfortunately there is no way to query how much childs a parent track
# has, so we have to do it "by hand":
#iterate through the whole project and ask every track:
for i in range(tracksInProject):
    currentTrack = RPR_GetTrack(0, i)
    if RPR_GetParentTrack(currentTrack) == parentTrack: #is this your parent?
        listOfChilds.append(currentTrack) #if yes, I put you on my list
# now we count how many childs our parent track has
numChilds = len(listOfChilds)
# switch off parent send to all child tracks
for idx, child in enumerate(listOfChilds):
    RPR_SetMediaTrackInfo_Value(child, "B_MAINSEND", 0)
# if we have an even number of selected tracks, set channel count on 
# the parent track to that number, otherwise +1 (necessary because it 
# can only have an even number of channels...)
if selTracks % 2 == 0:
    RPR_SetMediaTrackInfo_Value(parentTrack, "I_NCHAN", selTracks)
else:
    RPR_SetMediaTrackInfo_Value(parentTrack, "I_NCHAN", selTracks + 1)
# create a mono send to each selected track to the folder
# all other tracks will be unrouted, so they won't be in the render 
for i in range(selTracks):
    currentTrack = RPR_GetSelectedTrack(0, i)
    RPR_CreateTrackSend(currentTrack, parentTrack)
    RPR_SetTrackSendInfo_Value(currentTrack, 0, 0, "I_SRCCHAN", 1024)
    RPR_SetTrackSendInfo_Value(currentTrack, 0, 0, "I_DSTCHAN", i + 1024)