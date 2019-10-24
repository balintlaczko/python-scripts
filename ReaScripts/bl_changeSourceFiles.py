from reaper_python import *
import os


def BR_SetTakeSourceFromFile(p0, p1, p2):
    a = rpr_getfp('BR_SetTakeSourceFromFile')
    f = CFUNCTYPE(c_byte, c_uint64, c_char_p, c_byte)(a)
    t = (rpr_packp('MediaItem_Take*', p0), rpr_packsc(p1), c_byte(p2))
    r = f(t[0], t[1], t[2])
    return r


# the selected track
selectedTrack = RPR_GetSelectedTrack(0, 0)
# the number of items on that track
numItems = RPR_GetTrackNumMediaItems(selectedTrack)

# we will search for files ending with this
# change here if needed
suffix = "_bre-plos.wav"

# to count how many sources we change
changed = 0

# cycle through all the items on the selected track,
# get their -> take -> source -> file name,
# and if it's not already ending with the suffix,
# then change the source file for the take, and rename take
for i in range(numItems):
    currentItem = RPR_GetTrackMediaItem(selectedTrack, i)
    currentTake = RPR_GetActiveTake(currentItem)
    currentSource = RPR_GetMediaItemTake_Source(currentTake)
    currentFile = RPR_GetMediaSourceFileName(currentSource, "", 512)
    currentName = currentFile[1]
    # only if we haven't changed it already
    if not currentName.endswith(suffix):
        newName = currentName[:-4] + suffix
        BR_SetTakeSourceFromFile(currentTake, newName, 0)
        # extract file name from full path for naming our take
        path, newName = os.path.split(newName)
        RPR_GetSetMediaItemTakeInfo_String(currentTake, "P_NAME", newName, 1)
        changed += 1

# rebuild all peaks
RPR_Main_OnCommand(40048, 0)
# update the arrange view
RPR_UpdateArrange()
# say "done"
RPR_ShowConsoleMsg("DONE, Changed {} item sources!\n".format(changed))
