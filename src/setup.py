import commentjson as json
import pandas as pd
import numpy as np

# chain = (nwalkers, nlink, ndim)

class ReadJson(object):
    def __init__(self, filename):
        #print('Read the runprops.txt file')
        self.data = json.load(open(filename))
    def outProps(self):
        return self.data

def init_guess(runprops):
    # Load in the start guess
    guess = pd.read_csv(runprops.get("init guess"), index_col = 0)

    # Get the required dimensions
    ndims = guess.shape[0]
    nwalkers = runprops.get("nwalkers")

    # Now create p0
    means = guess["mean"].to_numpy()
    stds = guess["stddev"].to_numpy()
    p0 = np.random.normal(means, stds, (nwalkers, ndims))

    # If user requests a Maclaurin, make a = b
    if runprops.get("maclaurin"):
        p0[:,1] = p0[:,0]
        print("\nWARNING!: Fitting a Maclaurin spheroid (a = b).\n")

    # Return p0
    return p0