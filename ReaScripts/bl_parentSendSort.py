# This script allows you to quickly sort tracks in a folder (parent) track, 
# so that they are routed to subsequent channels on the folder.
# For example you have 30 mono tracks and you want to render them as one
# 30-channel file. You just simply make a new track above the first one, set 
# it to be a folder, and then close the folder on the last track you want to
# be included (so all the tracks you want to render are part of this folder).
# Then you select the folder track and run this script -for example with a 
# keyboard shortcut. If you now render out the folder track as a multichannel 
# file, all the included tracks will be routed correctly on subsequent
# channels.   

# import reaper API
from reaper_python import *
# name our selected track to be "the" parent (since you are supposed to
# select the parent track you want to sort)
parentTrack = RPR_GetSelectedTrack(0, 0)
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
# set the channel count on the parent to an even number closest to (rounded up)
# the number of childs in it - reaper limitation: a track can only have even
# number of channels
if numChilds % 2 == 0:
    RPR_SetMediaTrackInfo_Value(parentTrack, "I_NCHAN", numChilds)
else:
    RPR_SetMediaTrackInfo_Value(parentTrack, "I_NCHAN", numChilds + 1)
# and now we switch on parent send on every child, and offset them appropriately
# so that child-1 goes to 1st channel on parent, child-2 goes to 2nd, etc...
for idx, child in enumerate(listOfChilds):
    RPR_SetMediaTrackInfo_Value(child, "B_MAINSEND", 1)
    RPR_SetMediaTrackInfo_Value(child, "C_MAINSEND_OFFS", idx)
