#
#    likelihood.py
#
#    Likelihood related functions for occult3d
#    Benjamin Proudfoot
#    04/16/24
#

import numpy as np
import shapely.geometry as geometry
import sora
from sora.stats import Parameters
import pandas as pd
import emcee
import scipy
import sys
from astropy.time import Time
from astropy.coordinates import SkyCoord

# Data is formatted as follows:
# positive is an Nevent-element array of Nchord x 2 array with f,g coordinates of points around the limb
# positive_err is an Nevent-element array of Nchord-element arrays with uncertainties of points around the limb
# negative is an array of shapely linestrings of the most constraining negative chords
# This data needs to be premade before any fitting occurs

# This gets the limb shape and angle from Magnusson 1986
def magnusson(a,b,c,aspect,phase):
    # phase angle of 90/270 are minima
    asp = np.deg2rad(aspect)
    rp = np.deg2rad(phase)
    
    A = (b**2 * c**2 * np.sin(asp)**2 * np.sin(rp)**2) + (a**2 * c**2 * np.sin(asp)**2 * np.cos(rp)**2) + (a**2 * b**2 * np.cos(asp)**2)
    B = -1*(a**2 * (np.cos(asp)**2 * np.sin(rp)**2 + np.cos(rp)**2) + b**2 * (np.cos(asp)**2 * np.cos(rp)**2 + np.sin(rp)**2) + c**2 * (np.sin(asp)**2))

    u = (2*A / (-B - (B**2 - 4*A)**0.5))**0.5
    v = (2*A / (-B + (B**2 - 4*A)**0.5))**0.5
    
    top = 2*np.cos(asp)*np.cos(rp)*np.sin(rp)* (b**2 - a**2)
    bot = a**2 * (np.cos(asp)**2 * np.sin(rp)**2 - np.cos(rp)**2) + b**2 * (np.cos(asp)**2 * np.cos(rp)**2 - np.sin(rp)**2) + c**2 * (np.sin(asp)**2)
    gamma = -0.5 * np.arctan(top/bot)

    return u,v,np.rad2deg(gamma)

def dmag(a,b,c,aspect):
    th = np.deg2rad(aspect)
    dmag = -2.5*np.log10( (b/a)*( (a**2 * np.cos(th)**2 + c**2 * np.sin(th)**2) / (b**2 * np.cos(th)**2 + c**2 * np.sin(th)**2) )**0.5 )
    return dmag

def probability(params, positive, positive_err, negative, occobj, velocity, runprops):
    # Calculate the priors
    lp = prior(params, runprops)

    # Cancel the likelihood evaluation if priors are -inf
    if not np.isfinite(lp):
        return -np.inf

    # Calculate the likelihood
    pole = SkyCoord(params[3], params[4], unit = "deg", frame = "icrs")
    lh = likelihood(params, positive, positive_err, negative, occobj, velocity, runprops, pole)

    # Add in the light curve prior
    lp2 = lc_prior(params, occobj, runprops, pole)
    #lp2 = 0

    # Return the combined probability
    return (lh + lp + lp2)

def likelihood(params, positive, positive_err, negative, occobj, velocity, runprops, pole):
    # Unpack parameters
    a,b,c = params[:3]
    ra,dec,phi = params[3:6]
    f,g = params[-2:]

    # Loop through each event and calculate chi square
    nevents = runprops.get("nevents")
    #nevents = 2

    chisq = 0
    for i in range(nevents):
        # Exract occultation object
        occ = occobj[i]
        
        # Extract event specific data from data container
        fg_data = np.array(positive[i])
        fg_err = np.array(positive_err[i])
        negevent = negative[i]
        velevent = velocity[i]
        chisq += chisquare(a, b, c, ra, dec, phi, f, g, fg_data, fg_err, negevent, occ, velevent, runprops, pole)

    # Finishing up
    return -0.5*chisq

def chisquare(a, b, c, ra, dec, phi, f, g, fg_data, fg_err, negative, occ, velevent, runprops, pole):
    # Calculate chisquare using sora methods
    # First calculate ellipse values
    # calculate aspect and position angle
    obj = runprops.get("occ_coord")

    position_angle = obj.position_angle(pole).deg
    aperture_angle = np.arcsin( -(np.sin(pole.dec)*np.sin(obj.dec) + np.cos(pole.dec)*np.cos(obj.dec)*np.cos(pole.ra-obj.ra)) )

    pa = position_angle
    op = aperture_angle.to('deg').value
    aspect = (90-np.abs(op))
    #print(pa, aspect)

    # Now calculate ellipse limb based on 3D shape
    u_model, v_model, pa_offset = magnusson(a,b,c,aspect,phi)
    pa += pa_offset
    
    # Filter based on negative chords first
    for i, negative_i in enumerate(negative):
        fgall = negative_i.T
        f_all = fgall[0]
        g_all = fgall[1]
        df_path = f_all - f
        dg_path = g_all - g
        r_path = np.sqrt(df_path**2 + dg_path**2)
        theta_path = np.arctan2(dg_path, df_path)

        r_ellipse = sora.extra.utils.get_ellipse_points(theta_path,
                                       equatorial_radius=u_model,
                                       oblateness=((u_model-v_model)/u_model),
                                       center_f=f,
                                       center_g=g,
                                       position_angle=pa)[2]
        intersects = np.any((r_path - r_ellipse) < 0)
        if intersects:
            return np.inf

    # First calculate the ellipse
    initial = Parameters()
    initial.add(name='equatorial_radius', value=u_model)
    initial.add(name='center_f', value=f)
    initial.add(name='center_g', value=g)
    initial.add(name='oblateness', value= ((u_model-v_model)/u_model))
    initial.add(name='position_angle', value=pa)

    # Now, based on the ellipse shape, scale the f, g errors by the radial velocity.
    # This is important for chords near the limb, where the radial velocity is much lower
    # This is designed based on the SORA implementation of this in sora.occultation.core.check_velocities()
    angs = np.arctan( (-(fg_data[:,0] - f)/(fg_data[:,1] - g)) * np.power(v_model / u_model, 2) ) + (np.pi/2)
    observer_vecs = np.array([np.cos(angs), np.sin(angs)])
    normal_vels = np.abs( np.sum(observer_vecs * np.array(velevent).T, axis = 0) / np.linalg.norm(observer_vecs, axis = 0) )
    
    # Now find the chi square
    chisqs = sora.occultation.fitting.ellipseError(initial, fg_data[:,0], fg_data[:,1], fg_err*normal_vels)
    return np.sum(chisqs)

def prior(params, runprops):
    lp = 0

    # Unpack parameters
    a,b,c = params[:3]
    ra,dec,phi = params[3:6]
    f,g = params[-2:]

    oblateness = (a-b)/a
    deq = 2*np.sqrt(a*b)

    # Check if input params are okay
    if (a > runprops.get("max_a")):
        return -np.inf
    if (a < b) or (a < c):
        return -np.inf
    if (b < c):
        return -np.inf
    if (phi > 180) or (phi < 0):
        return -np.inf
    if (a < 0) or (b < 0) or (c < 0):
        return -np.inf
    if (ra < 0) or (ra > 360) or (dec < -90) or (dec > 90):
        return -np.inf
    if (c/a < 0.3):
        return -np.inf

    # Now implement geometrical priors
    # do here

    # Input other priors here!
    ra_prior = runprops.get("ra_prior", ra)
    dec_prior = runprops.get("dec_prior", dec)
    phi_prior = runprops.get("phi_prior", phi)
    ra_err = runprops.get("ra_error", 10000)
    dec_err = runprops.get("dec_error", 10000)
    phi_err = runprops.get("phi_error", 10000)

    lp += -0.5 * ( (ra - ra_prior)/(ra_err) )**2
    lp += -0.5 * ( (dec - dec_prior)/(dec_err) )**2
    lp += -0.5 * ( (phi - phi_prior)/(phi_err) )**2

    # Return lp
    return lp

def lc_prior(params, occobj, runprops, pole):
    # Unpack parameters
    a,b,c = params[:3]
    ra,dec,phi = params[3:6]
    f,g = params[-2:]
    occ = occobj[0]

    # Calculate the aspect angle
    obj = runprops.get("lc_coord")
    aperture_angle = np.arcsin( -(np.sin(pole.dec)*np.sin(obj.dec) + np.cos(pole.dec)*np.cos(obj.dec)*np.cos(pole.ra-obj.ra)) )
    op = aperture_angle.to('deg').value
    aspect = (90-np.abs(op))
    #print(aspect)

    # Calculate the light curve amplitude
    dm = dmag(a,b,c,aspect)

    dm_prior = runprops.get("dmag_prior")
    dm_err = runprops.get("dmag_error")

    # Now calculate the prior
    lp = -0.5 * ( (dm - dm_prior)/(dm_err) )**2
    return lp

def clustering(sampler, state, paramnames, probability, backend, allpositive, allpositive_err, allnegative, occobj, allvelocity,
               runprops, const = 50, lag = 10, max_prune_frac = 0.8, moves = None):
    nwalkers = runprops.get("nwalkers")
    reburnin = runprops.get("clustering_burnin")
    if reburnin == 0:
        return sampler, state

    # Getting important values from the chain
    avllhood = np.mean(sampler.get_log_prob()[-lag:,:], axis = 0)
    lastparams = sampler.get_chain()[-1,:,:]
    ngens = sampler.get_chain().shape[0]

    if ngens < lag:
        print("Chain too short for clustering algorithm, clustering not performed")
        return sampler, state

    # Sorting the walkers by likelihood values
    llhoodparam = pd.DataFrame(columns = ['llhood'] + paramnames)
    for i in range(nwalkers):
        llhoodparam.loc[i] = np.concatenate([np.array([avllhood[i]]),lastparams[i,:]])
    llhoodparam = llhoodparam.sort_values(by=['llhood'], ascending = False)
    llhoodparam = llhoodparam.values

    # Performing rejection tests
    reject = np.zeros(nwalkers)
    for i in range(1,nwalkers-1):
        term1 = -llhoodparam[i+1,0] + llhoodparam[i,0]
        term2 = const*(-llhoodparam[i,0] + llhoodparam[0,0])/(i)
        print(term1, term2)
        if term1 > term2:
            reject[(i+1):] = 1
            break
    freject = reject.sum()/nwalkers
    print(freject)
    ndim = np.shape(lastparams)[1]
    # Pruning walkers based on the clusters found,
    # replacing them with random linear combinations of walkers within the cluster
    # Skipping if cluster is not big enough
    if freject < max_prune_frac:
        params = llhoodparam[:,1:]
        for i in range(len(reject)):
            if reject[i] == 1:
                p = np.random.uniform()
                c1 = np.random.randint(i)
                c2 = np.random.randint(i)
                while c1 == c2:
                    c2 = np.random.randint(i)
                params[i,:] = (p*params[c1,:] + (1-p)*params[c2,:])
        #sampler.reset()
        sampler = emcee.EnsembleSampler(nwalkers, ndim, probability, backend=backend,
                                        args = [allpositive, allpositive_err, allnegative, occobj, allvelocity, runprops], 
                                        moves = moves)
        state = sampler.run_mcmc(params, reburnin, progress = True, skip_initial_state_check = True)
        return sampler, state
    else:
        print("Cluster not big enough, clustering not performed")
        return sampler, state
    
