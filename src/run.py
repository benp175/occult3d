#
#    run.py
#
#    Runs occult3d
#
#    Benjamin Proudfoot
#    04/17/24

from setup import ReadJson
import setup
import model
import likelihood
from likelihood import probability
import data
import analysis

import sys
import shutil
import os
from tqdm import tqdm
import datetime
import numpy as np
import emcee
import pandas as pd

def run():
    # Read runprops and set up place for outputs to go
    runprops = ReadJson("runprops.txt").outProps()

    # Create results folder
    x = datetime.datetime.now()
    date = str(x.strftime("%Y"))+"-"+str(x.strftime("%m"))+"-"+str(x.strftime("%d"))+"_"+str(x.strftime("%H"))+"."+str(x.strftime("%M"))
    resultspath = "../../results/"+runprops.get('name')+"_"+date
    if not os.path.exists(resultspath):
        os.makedirs(resultspath)
        #print("remove this before actually running!")
    else:
        print("Output directory already exists. Stuff might be messed up!")
        #sys.exit()

    # Copy all inputs into the newly minted results folder
    shutil.copy("runprops.txt", resultspath + "/runprops.txt")
    shutil.copy(runprops.get("init guess"), resultspath + "/" + runprops.get("name") + "_init_guess.csv")
    shutil.copy(runprops.get("chord data"), resultspath + "/" + runprops.get("chord data"))
    shutil.copy(runprops.get("occultation data"), resultspath + "/" + runprops.get("occultation data"))
    
    # Load in data and condition it
    allpositive, allpositive_err, allnegative, occobj, allvelocity = data.condition_data(runprops)

    # Check number of events. Implement multi-event in the future!
    nevents = runprops.get("nevents")
    if nevents > 1:
        print("Not currently able to do more than one event. Quitting.")
        sys.exit()
    
    # Set up p0 and get all necessary numbers
    p0 = setup.init_guess(runprops)
    nwalkers = p0.shape[0]
    ndims = p0.shape[1]
    nsteps = runprops.get("nsteps")
    nburnin = runprops.get("nburnin")

    # Name the parameter for utility later on
    paramnames = ["a","b","c","ra","dec","phi","f","g"]

    # Now move to src directory to run things
    os.chdir("../../src")
    resultspath = "../results/"+runprops.get('name')+"_"+date
    runprops["results path"] = resultspath

    # Check p0 using standard methods to check for -infs
    # May just need to use priors... that would improve performance
    maxreset = 10000
    reset = 0
    for i in tqdm(range(nwalkers)):
        lp = probability(p0[i,:], allpositive, allpositive_err, allnegative, occobj, allvelocity, runprops)
        while (lp == -np.inf):
            p = np.random.uniform()
            p0[i,:] = (p*p0[np.random.randint(nwalkers),:] + (1-p)*p0[np.random.randint(nwalkers),:])
            lp = probability(p0[i,:], allpositive, allpositive_err, allnegative, occobj, allvelocity, runprops)
            reset += 1
            if reset > maxreset:
                print("Bad inputs! Priors are -inf.")
                sys.exit()

    print("Number of resets: ", reset)

    # Set up emcee
    backend = emcee.backends.HDFBackend(resultspath + "/chain.h5")
    if runprops.get("multi_sample"):
        moves=[(emcee.moves.DEMove(), 0.8),(emcee.moves.DESnookerMove(), 0.2)]
    else:
        moves = None
    sampler = emcee.EnsembleSampler(nwalkers, ndims, probability, args = [allpositive, allpositive_err, allnegative, occobj, allvelocity, runprops], backend = backend, moves = moves)
                
    # Run emcee burn in
    state = sampler.run_mcmc(p0, nburnin, progress = True, skip_initial_state_check = True)

    # Clustering?
    if runprops.get("use_clustering"):
        sampler, state = likelihood.clustering(sampler, state, paramnames, probability, backend, allpositive, allpositive_err, allnegative, occobj, allvelocity, runprops)
    
    # Run emcee sampling
    sampler.run_mcmc(state, nsteps, progress = True, skip_initial_state_check = True)

    # Analyze output/make plots
    analysis.analysis(sampler, allpositive, allpositive_err, allnegative, occobj, paramnames, runprops)

if __name__ == '__main__':
    run()