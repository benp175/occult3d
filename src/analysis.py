#
#     analysis
#     
#     Analysis utilities for occult3d
#
#     Benjamin Proudfoot
#     04/17/24
#

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import corner
import sora
import os
import pandas as pd
import sys
import commentjson as json
from likelihood import magnusson, dmag
from astropy.coordinates import SkyCoord
from scipy.linalg import expm


# define rotation functions
def rotate(f,g,pa,op):
    # Rotate f,g into a on-sky coordinate system aligned with the rings
    x = f*np.cos(pa) + g*np.sin(pa)
    y = -f*np.sin(pa) + g*np.cos(pa)

    # Now rotate it into the ring plane
    # x remains the same
    yp = y/np.abs(np.sin(op))

    return x, yp

def derotate(x,yp,pa,op):
    # Rotate back into the sky plane
    y = yp*np.abs(np.sin(op))

    # Rotate back into N-S orientation
    f = x*np.cos(pa) - y*np.sin(pa)
    g = x*np.sin(pa) + y*np.cos(pa)

    return f,g

# Define function to get position angle of the rings and opening angle
def get_angles(t, obj, ra, dec, observer = "geocenter", subobserver = False):
    # Define the ring pole
    alpha = (ra/360)*24
    delta = dec
    obj.pole = '{} {}'.format(alpha, delta)
    #obj.pole = SkyCoord(ra,dec, frame = "icrs", unit = "deg")

    # Get the angles
    angs = obj.get_pole_position_angle(t, observer = observer)
    pa = angs[0].to('deg').value
    op = angs[1].to('deg').value

    if subobserver:
        frame = sora.body.frame.PlanetocentricFrame(epoch = t, pole = '{} {}'.format(alpha, delta), rotation_velocity = 0, prime_angle = 0)
        obj.frame = frame
        return obj.get_orientation(t, observer = observer)
    else:
        return pa, op

def analysis(sampler, allpositive, allpositive_err, allnegative, occobj, paramnames, runprops):
    # Load in the full chains
    burnin = int(runprops.get('nburnin'))
    clusterburn = int(runprops.get('clustering_burnin'))
    if not runprops.get("use_clustering"):
        clusterburn = 0
    chain = sampler.get_chain()
    
    # Make walker trace plots
    print("Making walker plots")
    plot_trace(chain, paramnames, runprops)
    del chain
    
    # Get flat chains
    thin = 1
    flatchain = sampler.get_chain(flat = True, discard = (burnin+clusterburn), thin = thin)
    flatlhood = sampler.get_log_prob(flat = True, discard = (burnin+clusterburn), thin = thin)
    best = np.argmax(flatlhood)

    # Getting nice latex labels
    latexlabels = [r"$a$ (km)", r"$b$ (km)", r"$c$ (km)", r"$\alpha$ ($^{\circ}$)", r"$\delta$ ($^{\circ}$)", r"$\phi$ ($^{\circ}$)", r"$f$ (km)", "$g$ (km)"]

    # Calculate derived parameters
    dfchain = flatchain.copy()

    # Add volumetric radius
    r_vol = (flatchain[:,0]*flatchain[:,1]*flatchain[:,2])**(1/3)
    dfchain = np.concatenate((dfchain, np.array([r_vol]).T), axis = 1)
    paramnames.append("volumetric radius")
    latexlabels.append(r"$r_{vol}$")

    # Add density
    mass = np.random.normal(runprops.get("mass"), runprops.get("mass_err"), r_vol.shape)
    #r_sat = np.random.normal(runprops.get("r_sat"), runprops.get("r_sat_err"), r_vol.shape)
    
    size_ratio = np.random.normal(runprops.get("r_sat"), runprops.get("r_sat_err"), r_vol.shape)
    r_sat = r_vol/size_ratio
    
    density = mass/((4/3)*np.pi*((r_vol*1000)**3 + ((r_sat*1000)**3)))
    dfchain = np.concatenate((dfchain, np.array([density]).T), axis = 1)
    paramnames.append("density")
    latexlabels.append(r"$\rho$ (kg/m$^3$)")

    # Add light curve magnitude
    obj = runprops.get("lc_coord")
    pole = SkyCoord(flatchain[:,3], flatchain[:,4], unit = "deg", frame = "icrs")
    aperture_angle = np.arcsin( -(np.sin(pole.dec)*np.sin(obj.dec) + np.cos(pole.dec)*np.cos(obj.dec)*np.cos(pole.ra-obj.ra)) )
    op = aperture_angle.to('deg').value
    aspect = (90-np.abs(op))
    dm = dmag(flatchain[:,0],flatchain[:,1],flatchain[:,2],aspect)
    dfchain = np.concatenate((dfchain, np.array([dm]).T), axis = 1)
    paramnames.append("delta mag")
    latexlabels.append(r"$\Delta m$ (mag)")

    # Add u, v, pa
    obj = runprops.get("occ_coord")
    position_angle = obj.position_angle(pole).deg
    aperture_angle = np.arcsin( -(np.sin(pole.dec)*np.sin(obj.dec) + np.cos(pole.dec)*np.cos(obj.dec)*np.cos(pole.ra-obj.ra)) )
    pa = position_angle
    op = aperture_angle.to('deg').value
    aspect = (90-np.abs(op))
    u, v, pa_offset = magnusson(flatchain[:,0],flatchain[:,1],flatchain[:,2],aspect,flatchain[:,5])
    pa += pa_offset

    dfchain = np.concatenate((dfchain, np.array([u]).T), axis = 1)
    paramnames.append("u")
    latexlabels.append(r"$u$ (km)")

    dfchain = np.concatenate((dfchain, np.array([v]).T), axis = 1)
    paramnames.append("v")
    latexlabels.append(r"$v$ (km)")

    # Add axes ratios
    aa = flatchain[:,0]
    bb = flatchain[:,1]
    cc = flatchain[:,2]

    ca = cc/aa
    ba = bb/aa

    dfchain = np.concatenate((dfchain, np.array([ca]).T), axis = 1)
    dfchain = np.concatenate((dfchain, np.array([ba]).T), axis = 1)

    paramnames.append("c/a")
    paramnames.append("b/a")

    latexlabels.append(r"$c/a$")
    latexlabels.append(r"$b/a$")
    
        # add more if wanted

    del flatchain
    flatchain = dfchain

    # Make sigs
    print("Making sigsdf")
    make_sigs(flatchain, flatchain[best,:], paramnames, runprops)
    
    # Make likelihood plots
    print("Making likelihood plots")
    plot_likelihoods(flatchain, flatlhood, paramnames, runprops)

    # Make corner plots
    print("Making corner plots")
    plot_corner(flatchain[:,:8], flatchain[best,:8], latexlabels[:8], runprops)
    plot_corner(flatchain[:,:], flatchain[best,:], latexlabels[:], runprops, fname = "corner+derived.pdf")
    
    # Make limb plots
    print("Making limb plots")
    plot_limbs(flatchain, flatlhood, occobj, runprops, ndraws = 100)

    # Make 3d limb plots
    print("Making 3d limb plots")
    plot_limbs_3d(flatchain, flatlhood, occobj, runprops, lim = np.median(r_vol))

def plot_trace(chain, names, runprops):
    # Make the walker plot PDFs
    walkerpdf = PdfPages(runprops.get("results path") + "/walkers.pdf")
    burnin = int(runprops.get('nburnin'))
    clusterburn = int(runprops.get('clustering_burnin'))
    nwalkers = runprops.get("nwalkers")
    numgens = chain.shape[0]
    for i in range(len(names)):
        plt.figure(dpi = 50)
        for j in range(nwalkers):
            plt.plot(np.reshape(chain[0:numgens,j,i], numgens), alpha=0.2, rasterized=True)
        plt.ylabel(names[i])
        plt.xlabel("Generation")
        walkerpdf.savefig(bbox_inches='tight')

    walkerpdf.close()
    plt.close("all")

def plot_likelihoods(flatchain, flatlhood, names, runprops):
    nwalkers = runprops.get("nwalkers")
    likelihoodspdf = PdfPages(runprops.get("results path") + "/likelihoods.pdf")
    ylimmin = np.percentile(flatlhood, 1)
    ylimmax = flatlhood.max() + 1
    for i in range(len(names)):
        plt.figure(figsize = (9,9))
        plt.subplot(221)
        plt.hist(flatchain[:,i].flatten(), bins = 40, histtype = "step", color = "black")
        plt.subplot(223)
        plt.scatter(flatchain[:,i].flatten(), flatlhood,
                    c = np.mod(np.linspace(0,flatlhood.size - 1, flatlhood.size), nwalkers),
                    cmap = "nipy_spectral", edgecolors = "none", rasterized=True, alpha=0.1)
        plt.xlabel(names[i])
        plt.ylabel("Log(L)")
        plt.ylim(ylimmin, ylimmax)
        plt.subplot(224)
        llflat = flatlhood.flatten()
        plt.hist(llflat[np.isfinite(llflat)], bins = 40, orientation = "horizontal",
                 histtype = "step", color = "black")
        plt.ylim(ylimmin, ylimmax)
        likelihoodspdf.savefig(bbox_inches='tight')
    likelihoodspdf.close()
    plt.close("all")

def plot_corner(flatchain, best, names, runprops, fname = "corner.pdf"):
    fig = corner.corner(flatchain, labels = names, plot_datapoints = True, color = "#404788FF", 
                        fill_contours = True, show_titles = True, bins = 40, title_fmt = ".3f", label_kwargs=dict(fontsize=20))
    fig.tight_layout(pad = 1.08, h_pad = -0.4, w_pad = -0.4)
    for ax in fig.get_axes():
        ax.tick_params(axis = "both", labelsize = 12, pad = 0.0)
    fig.savefig(runprops.get("results path") + "/" + fname, bbox_inches='tight')
    plt.close("all")

def plot_limbs(flatchain, flatlhood, occobj, runprops, ndraws = 100):
    # Get the draws from the flatchains
    drawsindex = np.random.randint(flatchain.shape[0], size = ndraws)
    draws = flatchain[drawsindex,:]

    best = flatchain[np.argmax(flatlhood),:]

    # Now loop over the events and plot the chords and limbs from the draws
    nevents = runprops.get("nevents")
    limbspdf = PdfPages(runprops.get("results path") + "/limbs.pdf")
    
    for i in range(nevents):
        obj = runprops.get("occ_coord")
        pole = SkyCoord(best[3], best[4], unit = "deg", frame = "icrs")

        best_pa = obj.position_angle(pole).deg
        aperture_angle = np.arcsin( -(np.sin(pole.dec)*np.sin(obj.dec) + np.cos(pole.dec)*np.cos(obj.dec)*np.cos(pole.ra-obj.ra)) )
        op = aperture_angle.to('deg').value
        best_aspect = (90-np.abs(op))
        
        best_u, best_v, best_offset = magnusson(best[0], best[1], best[2], best_aspect, best[5])
        best_pa += best_offset

        best_f = best[6]
        best_g = best[7]
        best_ob = (best_u - best_v)/best_u

        # Plot best ellipse
        plt.figure()
        sora.extra.plots.draw_ellipse(best_u, oblateness = best_ob, center_f = best_f, center_g = best_g, position_angle = best_pa,
                                     alpha = 1.0, color = "black", zorder = 2.5, lw = 2.0)
        plt.scatter(best_f,best_g, marker = "*", zorder = 2.5, alpha = 1.0, color = "black")

        # Plot ellipse draws
        maxlhood = np.max(flatlhood)
        for j in range(ndraws):
            while (maxlhood - 0.5) > flatlhood[drawsindex[j]]:
                drawsindex[j] = np.random.randint(flatchain.shape[0])

            draw = drawsindex[j]

            pole = SkyCoord(flatchain[draw,3], flatchain[draw,4], unit = "deg", frame = "icrs")
            pa = obj.position_angle(pole).deg
            aperture_angle = np.arcsin( -(np.sin(pole.dec)*np.sin(obj.dec) + np.cos(pole.dec)*np.cos(obj.dec)*np.cos(pole.ra-obj.ra)) )
            op = aperture_angle.to('deg').value
            aspect = (90-np.abs(op))

            u, v, offset = magnusson(flatchain[draw,0], flatchain[draw,1], flatchain[draw,2], aspect, flatchain[draw,5])
            pa += offset

            f = flatchain[draw,6]
            g = flatchain[draw,7]
            ob = (u - v)/u
            
            sora.extra.plots.draw_ellipse(u, oblateness = ob, 
                                         center_f = f, center_g = g, position_angle = pa,
                                         alpha = 0.02, color = "black", zorder = 2.6, lw = 1.0)
            plt.scatter(f,g, marker = ".", zorder = 2.6, alpha = 0.02, color = "black", edgecolors = "none")

        # Get axes now
        xlims = plt.xlim()
        ylims = plt.ylim()
        

        # Color cycle
        import matplotlib as mpl
        #plt.rcParams['axes.prop_cycle'] = plt.cycler(color=["#12007b","#d2a641","#b62d81","#458832","#418a8c","#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"])
        #plt.gca().set_prop_cycle(plt.cycler(color=["#12007b","#d2a641","#b62d81","#458832","#418a8c","#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]))

        # Plot chords
        if "results" in os.getcwd():
            os.chdir("../../runs/" + runprops.get("name") )
        else:
            os.chdir("../runs/" + runprops.get("name") )
        occ = occobj[i]     # This should already have all the chords inside it(?)
        
        # Add the offset of the one chord
        #if i == 1:
        #    occ.chords[0].lightcurve.dt = best_toff
        
        occ.chords.plot_chords(zorder = 3, lw = 1.5)
        occ.chords.plot_chords(segment = "error", color = "red", zorder = 4, lw = 0.9)
        if "results" in os.getcwd():
            os.chdir(runprops.get("results path"))
        else:
            os.chdir("../../src")
        
        # Edit plot axes
        plt.gca().set_aspect('equal', adjustable='datalim')
        dx = xlims[1] - xlims[0]
        #plt.xlim(xlims[0]-0.45*dx, xlims[1]+0.15*dx)
        plt.xlabel("f (km)", fontsize = 14)
        plt.ylabel("g (km)", fontsize = 14)
        #plt.legend(loc = "upper right", fontsize = 8)
        plt.gca().invert_xaxis()
        limbspdf.savefig(bbox_inches='tight')
    limbspdf.close()
    plt.close("all")


def plot_limbs_3d(flatchain, flatlhood, occobj, runprops, lim = 500):
    # Get the best parameter set
    best = flatchain[np.argmax(flatlhood),:]

    # Now loop over the events and plot the chords and limbs from the draws
    nevents = runprops.get("nevents")
    limbspdf = PdfPages(runprops.get("results path") + "/limbs_3d.pdf")
    
    for i in range(nevents):
        obj = runprops.get("body_obj")
        
        # Get the shape model out of the best array
        rx = best[0]
        ry = best[1]
        rz = best[2]
        ra = best[3]
        dec = best[4]
        phi = best[5]
        f = best[6]
        g = best[7]

        # Create the ellipsoid
        u = np.deg2rad(np.linspace(0, 360, 361, endpoint = True))
        v = np.deg2rad(np.linspace(0, 180, 181, endpoint = True))
        x = rx * np.outer(np.cos(u), np.sin(v))
        y = ry * np.outer(np.sin(u), np.sin(v))
        z = rz * np.outer(np.ones_like(u), np.cos(v))

        # Rotate the ellipsoid around its rotation axis
        alpha = 0
        beta = 0
        gamma = np.radians(-phi)  # Rotation around z-axis, negative when viewing the Northern(?), positive when viewing southern(?)?????

        Rx = expm(np.array([[0, 0, 0], [0, 0, -alpha], [0, alpha, 0]]))
        Ry = expm(np.array([[0, 0, beta], [0, 0, 0], [-beta, 0, 0]]))
        Rz = expm(np.array([[0, -gamma, 0], [gamma, 0, 0], [0, 0, 0]]))

        rotated_ellipsoid = np.array([x, y, z])
        for ii in range(x.shape[0]):
            for jj in range(x.shape[1]):
                rotated_ellipsoid[:, ii, jj] = Rz @ Ry @ Rx @ rotated_ellipsoid[:, ii, jj]

        x, y, z = rotated_ellipsoid

        # Get the orientation of the ellipsoid on the sky        
        t = occobj[i].tca
        pa, op = get_angles(t, obj, ra, dec, observer = "geocenter")
        pa_rad = np.deg2rad(360-pa)
        op_rad = np.deg2rad(op)
        subobs = get_angles(t, obj, ra, dec, observer = "geocenter", subobserver = True)["sub_observer"]
        lon,lat = [float(x) for x in subobs.split()]

        # Plot the ellipsoid
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(x, y, z,  rstride=10, cstride=10, color='gainsboro', edgecolor = "black", linewidth = 0.25, shade = False)

        # Now get the chords, tranform them, and plot them
        for j in range(len(occobj[i].chords)):
            # Get the positive portion of the chord
            try:
                fsky, gsky = occobj[i].chords[j].path(segment = "positive")
                fchord = fsky - f
                gchord = gsky - g
                xchord, ychord = rotate(fchord, gchord, pa_rad, op_rad)
                plt.plot(xchord, ychord, zorder = 3, lw = 2.5)
            
                # Get the error portions of the chord
                ferr_sky1, gerr_sky1, ferr_sky2, gerr_sky2 = occobj[i].chords[j].path(segment = "error")
                ferr1 = ferr_sky1 - f
                gerr1 = gerr_sky1 - g
                ferr2 = ferr_sky2 - f
                gerr2 = gerr_sky2 - g
                xerr1, yerr1 = rotate(ferr1, gerr1, pa_rad, op_rad)
                xerr2, yerr2 = rotate(ferr2, gerr2, pa_rad, op_rad)
                plt.plot(xerr1, yerr1, zorder = 4, lw = 2, color = "red")
                plt.plot(xerr2, yerr2, zorder = 4, lw = 2, color = "red")
            except ValueError:
                continue

        # Mess with plot settings for correct size and viewing geometry
        ax.set_xlim(-lim/1.25,lim/1.25)
        ax.set_ylim(-lim/1.25,lim/1.25)
        ax.set_zlim(-lim/1.25,lim/1.25)
        ax.axis("off")
        ax.view_init(elev=lat, azim=90, roll = pa)              # elevation is subsolar latitude, azimuth does nothing, roll is position angle
        ax.set_proj_type('ortho')
        ax.set_aspect("equal")

        #ax.legend(loc = (0.8,0.6), fontsize = 8)

        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

        #plt.show()

        # Now add another subplot on top of the original to hold all the other shit
        ax = fig.add_subplot(111)
        ax.axis("off")

        # Make compass markings
        ax.plot([0.12,0.12],[0.0,0.12], linewidth = 1, color = "black")
        ax.plot([0.12,0.0],[0.0,0.0], linewidth = 1, color = "black")
        ax.text(0.12, 0.135, "N", fontsize = 14, ha = "center")
        ax.text(-0.02, -0.015, "E", fontsize = 14, ha = "center")
        #ax.text(0.06, 0.25, "1000 km", fontsize = 10, ha = "center")

        #ax.arrow(0.9, 0.25, -0.1*np.cos(pastar/57.2958), -0.1*np.sin(pastar/57.2958), color = "red", head_width = 0.01)

        ax.set_xlim(0,1)
        ax.set_ylim(0,1)
        ax.set_aspect("equal")
        ax.axis('off')
        
        limbspdf.savefig(bbox_inches='tight')
    limbspdf.close()
    plt.close("all")

def make_sigs(flatchain, best, names, runprops):
    sigsdf = pd.DataFrame(columns = ['-3sigma','-2sigma','-1sigma','median','1sigma','2sigma','3sigma', 'best fit'], index = names)
    for i in range(len(flatchain[0])):
        num = flatchain[:,i]
        median = np.percentile(num,50, axis = None)
        neg3sig= np.percentile(num,0.37, axis = None)
        neg2sig = np.percentile(num,2.275, axis = None)
        neg1sig = np.percentile(num,15.866, axis = None)
        pos1sig = np.percentile(num,84.134, axis = None)
        pos2sig = np.percentile(num,97.724, axis = None)
        pos3sig = np.percentile(num,99.63, axis = None)
        bestfit = best[i]
        sigsdf.loc[names[i],'-3sigma'] = neg3sig-median
        sigsdf.loc[names[i],'-2sigma'] = neg2sig-median
        sigsdf.loc[names[i],'-1sigma'] = neg1sig-median
        sigsdf.loc[names[i],'median'] = median
        sigsdf.loc[names[i],'1sigma'] = pos1sig-median
        sigsdf.loc[names[i],'2sigma'] = pos2sig-median
        sigsdf.loc[names[i],'3sigma'] = pos3sig-median
        sigsdf.loc[names[i],'best fit'] = bestfit
    #print(sigsdf)
    filename = runprops.get("results path") + '/sigsdf.csv'    
    sigsdf.to_csv(filename)

class ReadJson(object):
    def __init__(self, filename):
        print('Read the runprops.txt file')
        self.data = json.load(open(filename))
    def outProps(self):
        return self.data

if __name__ == "__main__":
    # Esnure this is running from a results directory
    if not "results" in os.getcwd():
        print("analysis.py must be run from a results directory.")
        sys.exit()

    # Import necessary packages
    import emcee
    import data

    # Load in the runprops
    getData = ReadJson('runprops.txt')
    runprops = getData.outProps()
    runprops["results path"] = os.getcwd()

    # Recondition the data
    allpositive, allpositive_err, allnegative, occobj, junk = data.condition_data(runprops)

    # Provide names for the parameters
    paramnames = ["a","b","c","ra","dec","phi","f","g"]

    # Load in chain file
    backend = emcee.backends.HDFBackend('chain.h5')

    # Make plots and stuff
    analysis(backend, allpositive, allpositive_err, allnegative, occobj, paramnames, runprops)
