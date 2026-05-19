# occult3d

A Python tool to fit a triaxial ellipsoid shape model to stellar occultation chords using Bayesian MCMC sampling. The tool is designed fto determine triaxial shape models of trans-Neptunian objects, dwarf planets, and other solar system bodies observed via stellar occultation.

## Overview

`occult3d` takes multi-chord stellar occultation observations and fits a triaxial body (semi-axes *a*, *b*, *c*) along with its spin pole orientation and sky-plane offset. It uses the [`emcee`](https://emcee.readthedocs.io/) ensemble sampler and leverages [`SORA`](https://sora.readthedocs.io/) for occultation geometry. Both positive chords (detections) and negative chords (non-detections) are used as constraints.

## Features

- Fits a full triaxial ellipsoid (semi-axes *a*, *b*, *c*) to occultation chords
- Constrains spin pole orientation (RA, Dec) based on priors and sky-plane offset (*f*, *g*)
- Light curve magnitude prior on spin pole orientation
- Optional Maclaurin spheroid mode (forces *a* = *b*)
- Optional model uncertainty to account for topography
- Saves full MCMC chain to HDF5 for full reproducibility
- Produces diagnostic and publication-quality visualiztions of shape models

## Citation

If you use `occult3d` in your research, please cite the associated paper:

> Citation to be added upon publication.

A BibTeX entry for use in LaTeX:

```bibtex
% Add BibTeX entry here upon publication
```

## Dependencies

Install the following before running:

```
numpy
pandas
scipy
matplotlib
emcee
corner
tqdm
shapely
astropy
sora
commentjson
h5py
```

A SPICE ephemeris kernel for your target body (`.bsp` file) is also required, along with a solar system ephemeris (e.g., `de438s.bsp`, provided in `runs/ephem/`).

## Quick start

A few examples of working examples are shown in `runs/`. An example output is also shown in `results/`. 

### 1. Set up a run directory

Create a subdirectory under `runs/` for your target (or copy an existing one as a template):

```
runs/
└── MyTarget/
    ├── runprops.txt          # Run configuration (see below)
    ├── chords.csv            # Occultation chord data
    ├── occultations.csv      # Event metadata
    └── MyTarget_init_guess.csv   # Initial parameter guess
```

### 2. Prepare your input files

**`chords.csv`** — one row per chord, positive or negative:

| Column | Description |
|---|---|
| `Date` | Event date/time matching `occultations.csv` (e.g. `"2014-11-15 10:19"`) |
| `Location` | Station name |
| `Longitude` | Observer longitude (degrees) |
| `Latitude` | Observer latitude (degrees) |
| `Height` | Observer elevation (metres) |
| `Start` | Light curve start time (UTC string) |
| `Stop` | Light curve end time (UTC string) |
| `Immersion` | Immersion time — leave blank for negative chords |
| `Emersion` | Emersion time — leave blank for negative chords |
| `Immersion error` | Immersion timing uncertainty (seconds) — leave blank for negative chords |
| `Emersion error` | Emersion timing uncertainty (seconds) — leave blank for negative chords |

**`occultations.csv`** — one row per event:

| Column | Description |
|---|---|
| `Date` | Event date/time (must match `chords.csv`) |
| `Star coordinates` | Occulted star ICRS coordinates (e.g. `"04 29 30.61 -00 28 20.908"`) |

**`<target>_init_guess.csv`** — initial MCMC walker positions, one row per parameter:

| Parameter | Description |
|---|---|
| `a` | Semi-major axis (km) |
| `b` | Intermediate axis (km) |
| `c` | Minor axis (km) |
| `ra` | Spin pole right ascension (degrees) |
| `dec` | Spin pole declination (degrees) |
| `phi` | Rotation phase (degrees) |
| `f` | Sky-plane offset in RA direction (km) |
| `g` | Sky-plane offset in Dec direction (km) |

Each parameter has a `mean` and `stddev` column used to initialise walkers via a Gaussian ball.

### 3. Configure `runprops.txt`

`runprops.txt` is a JSON file (comments allowed) controlling all run settings. Key fields:

```jsonc
{
    "nevents": 1,                       // Number of occultation events
    "name": "MyTarget",                 // Target name (used for output folder naming)
    "spkid": "20229762",                // SPICE ID for the target body

    "nwalkers": 100,                    // Number of MCMC walkers
    "nsteps": 1000,                     // Production steps per walker
    "nburnin": 1000,                    // Burn-in steps per walker
    "clustering_burnin": 200,           // Additional burn-in after clustering

    "use_clustering": true,             // Enable walker clustering
    "multi_sample": false,              // Use DE/Snooker moves (vs default stretch)

    "occultation data": "occultations.csv",
    "chord data": "chords.csv",
    "init guess": "MyTarget_init_guess.csv",
    "ephem": "MyTarget_ephem.bsp",      // Target ephemeris kernel (placed in runs/ephem/)

    "maclaurin": false,                 // Force a = b (Maclaurin spheroid)

    "max_a": 2000,                      // Prior: maximum semi-major axis (km)
    "ra_prior": 0.0,                    // Prior: spin pole RA centre (degrees)
    "ra_error": 360.0,                  // Prior: spin pole RA uncertainty (degrees)
    "dec_prior": 0.0,                   // Prior: spin pole Dec centre (degrees)
    "dec_error": 90.0,                  // Prior: spin pole Dec uncertainty (degrees)
    "lc_time": "2014-10-30 12:00",      // Reference time for light curve constraint

    "H_mag": 4.476,                     // Absolute magnitude (used to calculate albedo in some cases)
    "H_mag_err": 0.013,
    "dmag_prior": 0.0,                  // Light curve amplitude prior
    "dmag_error": 1.0,
    "mass": 1.36e20,                    // Body mass (kg) for density calculation
    "mass_err": 3.3e18,

    "verbose": false
}
```

### 4. Run

Navigate into your run directory and execute `run.py` from the `src/` directory:

```bash
cd runs/MyTarget
python ../../src/run.py
```

Results are written automatically to a timestamped subdirectory under `results/`, e.g. `results/MyTarget_2025-06-01_14.32/`.

## Outputs

Each completed run produces:

| File | Description |
|---|---|
| `chain.h5` | Full MCMC chain (HDF5, readable with `emcee`) |
| `corner.pdf` | Corner plot of sampled parameters |
| `corner+derived.pdf` | Corner plot including derived quantities (volumetric radius, density) |
| `walkers.pdf` | Walker trace plots for all parameters |
| `limbs.pdf` | Best-fit limb overlaid on chord endpoints (sky plane) |
| `limbs_3d.pdf` | 3D shape visualisation |
| `likelihoods.pdf` | Log-likelihood as a function of a single variable (different colors = different walker) |
| `sigsdf.csv` | Posterior medians and uncertainties for all parameters |
| `runprops.txt` | Copy of the run configuration |
| `<target>_init_guess.csv` | Copy of the initial guess |
| `chords.csv` | Copy of the chord data |
| `occultations.csv` | Copy of the event metadata |

## Example

A fully worked example for 2007 UK₁₂₆ (using a single occultation event from 2014) is provided in `runs/uk126/` and `results/UK126_example/`. This is a good starting point for understanding the expected input format and typical output.

## License

MIT — see [LICENSE](LICENSE) for details.
