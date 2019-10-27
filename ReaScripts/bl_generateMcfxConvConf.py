"""
This script is for generating .conf files for an mcfx_convolver
to create IR-reverbs (or other weirder stuff) for Higher-Order
Ambisonics. Upon execution, the script
1.) let's the user choose the IR file through a browser
2.) asks for the desired HOA order
3.) generates a .conf file in the same folder as the IR
"""

# import the api
from reaper_python import *
import os

# returns number of ambisonic spherical harmonics of a given HOA order in 3D


def calcNumChannels(order):
    return (order+1)**2

# given the name of the config file, the absolute path of the IR and the HOA order,
# this generates the necessary .conf file for mcfx_convolver


def generateConf(confname, ir_fullpath, order):
    outfile = open(confname, "w")
    num_ch = calcNumChannels(order)
    outfile.write("/convolver/new %d %d 256 204800 1.0\n" % (num_ch, num_ch))

    for i in range(num_ch):
        outfile.write("/input/name %d In.%d\n" % (i+1, i+1))

    for i in range(num_ch):
        outfile.write("/output/name %d Out.%d\n" % (i+1, i+1))

    for i in range(num_ch):
        outfile.write("/impulse/read %d %d 1 0 0 0 1 %s\n" %
                      (i+1, i+1, ir_fullpath))

    outfile.close()

    RPR_ShowConsoleMsg("Config file written to %s" % (confname))


# show a dialog where the user can select the IR file
filenameNeed, title, defext = "", "", ""
retval_ir, filenameNeed, title, defext = RPR_GetUserFileNameForRead(
    filenameNeed, title, defext)

# if not cancelled
if retval_ir:
    fileName = filenameNeed.replace("\\", "/")
    target_dir, ir_name_wext = os.path.split(fileName)
    ir_name, ir_ext = os.path.splitext(ir_name_wext)
    # pop-up dialog for selecting the desired HOA order
    retval_hoaord, _t, _num, _cap, response, _maxlen = RPR_GetUserInputs(
        "Select HOA Order", 1, "HOA Order:", "7", 100)
    # if not cancelled
    if retval_hoaord:
        hoa_order = int(response)
        if hoa_order < 1:
            hoa_order = 1
        elif hoa_order > 7:
            hoa_order = 7

        target_file = "%s_%dth_order.conf" % (ir_name, hoa_order)
        target_fullPath = os.path.join(target_dir, target_file)
        target_fullPath = target_fullPath.replace("\\", "/")
        # write the conf file
        generateConf(target_fullPath, fileName, hoa_order)
