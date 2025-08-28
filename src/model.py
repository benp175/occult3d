#
#	model.py
#
#	Model of a 2d ellipse
#
#	Benjamin Proudfoot
#	04/03/24
#

import numpy as np
import sora
from astropy.coordinates import SkyCoord
from astropy.time import Time
import commentjson as json
import sys

def initialize(params, runprops):
    # Takes input parameter array (from emcee) and generates a shape model using the body class in sora
	
    # Ensures params has the correct elements
    if params.size != 5:
        # error
        print("Too few/many params in params")
        sys.exit()
	
    # Unpack shape, orientation, and rotation period
    a,b = params[:2]
    f,g = params[2:4]
    pa = params[4]

    # Output
    return body, fgs
