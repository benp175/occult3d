#
#    data.py
#    
#    Conditions data for use in occult3d
#
#    Benjamin Proudfoot
#    04/16/24

import numpy as np
import sora
import sys
from sora.prediction import prediction
from sora import Occultation, Body, Star, LightCurve, Observer
import shapely.geometry as geometry
import pandas as pd

def condition_data(runprops):

    # Set up lists for holding outputs from loops
    allpositive = []
    allpositive_err = []
    allnegative = []
    #allouter = []
    #allouter_err = []
    occobj = []
    allvelocity = []

    # Get the details of the events
    occults = pd.read_csv(runprops.get("occultation data"))
    chords = pd.read_csv(runprops.get("chord data"), comment = "#")

    # Run a few checks on the inputs
    if runprops.get("nevents") != occults["Date"].nunique():
        print("Number of events in occultations.csv and runprops.txt do not match")
        sys.exit()
    if runprops.get("nevents") != chords["Date"].nunique():
        print("Number of events in chords.csv and runprops.txt do not match")
        sys.exit()
    
    # First postdict the occultations to use as a starting point
    # These predictions don't really affect the outcomes too much
    for i in range(runprops.get("nevents")):
        event = occults.iloc[i]
        eventname = event["Date"]
        eventchords = chords.loc[chords["Date"] == eventname]
        
        nchords = eventchords.shape[0]
        names = eventchords["Location"].to_numpy()
        lons = eventchords["Longitude"].to_numpy()
        lats = eventchords["Latitude"].to_numpy()
        heights = eventchords["Height"].to_numpy()
        initial_times = eventchords["Start"].to_numpy()
        end_times = eventchords["Stop"].to_numpy()
        immersions = eventchords["Immersion"].to_numpy()
        immersion_errs = eventchords["Immersion error"].to_numpy()
        emersions = eventchords["Emersion"].to_numpy()
        emersion_errs = eventchords["Emersion error"].to_numpy()

        time = event["Date"]
        starcoord = event["Star coordinates"]
        
        body = Body(name = runprops.get("name"), database = None, spkid = runprops.get("spkid"), 
                              ephem = ["../ephem/"+runprops.get("ephem"),"../ephem/de438s.bsp"], H = runprops.get("H_mag"))
        star = Star(coord = starcoord)
        occ = Occultation(star = star, body = body, time = time)
        runprops["distance" + str(i)] = occ.dist.value*149597870.7
        runprops["occ_coord"] = body.ephem.get_position(time, observer = "geocenter")
        runprops["body_obj"] = body

        runprops["lc_coord"] = body.ephem.get_position(runprops.get("lc_time"), observer = "geocenter")
        

        # Now use the prediction to convert light curve times to f,g
        positive = []
        positive_err = []
        negative = []
        velocity = []
        #outer = []
        #outer_err = []
        for j in range(nchords):
            obs = Observer(name = names[j], lon = lons[j], lat = lats[j], height = heights[j])

            # If chord is negative do not use immersion and emersion
            if np.isnan(immersion_errs[j]):
                lc = LightCurve(name = names[j], initial_time = initial_times[j], end_time = end_times[j])
            else:
                lc = LightCurve(name = names[j], initial_time = initial_times[j], end_time = end_times[j],
                                immersion = immersions[j], immersion_err = immersion_errs[j],
                                emersion = emersions[j], emersion_err = emersion_errs[j])

            # Define the chord inside sora
            chord = occ.chords.add_chord(observer = obs, lightcurve = lc)

            # Now convert the chords to f,g or shapely linestrings for negatives
            if chord.status() == "positive":
                # Get the positive portion of the chord
                fi, gi, vfi, vgi = chord.get_fg(time='immersion', vel=True)
                erri = chord.lightcurve.immersion_err
                #erri = np.linalg.norm([vfi, vgi])*chord.lightcurve.immersion_err
                
                positive.append([fi,gi])
                positive_err.append(erri)
                fe, ge, vfe, vge = chord.get_fg(time='emersion', vel=True)
                erre = chord.lightcurve.emersion_err
                #erre = np.linalg.norm([vfe, vge])*chord.lightcurve.emersion_err
                
                positive.append([fe,ge])
                positive_err.append(erre)

                velocity.append([vfi,vgi])
                velocity.append([vfe,vge])

                # Now get the negative portion of the chord (outer)
                #f_all1, g_all1, junk1, junk2 = chord.path(segment='outer', 
                #                                            step = (chord.lightcurve.immersion_err) )
                #junk1, junk2, f_all2, g_all2 = chord.path(segment='outer', 
                #                                            step = (chord.lightcurve.emersion_err) )
                #coords1 = np.array([f_all1,g_all1]).T
                #coords2 = np.array([f_all2,g_all2]).T
                #out1 = LineString(coords1)
                #out2 = LineString(coords2)
                #outer.append(coords1)
                #outer.append(coords2)
                #outer_err.append(erri)
                #outer_err.append(erre)
                
            if chord.status() != "positive":
                f_all, g_all = chord.path(segment='full', step=1)
                coords = np.array([f_all,g_all]).T
                #neg = LineString(coords)
                negative.append(coords)


        # Now add all the chords from this event to multievent lists
        allpositive.append(positive)
        allpositive_err.append(positive_err)
        allnegative.append(negative)
        #allouter.append(outer)
        #allouter_err.append(outer_err)
        allvelocity.append(velocity)

        # Additionally output the occultation objects for later use
        occobj.append(occ)

    # Now that all events and chords have been analyzed, return data in useable form
    return allpositive, allpositive_err, allnegative, occobj, allvelocity







