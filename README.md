# TMT IRIS ETC

Simple TMT IRIS ETC based on IRIS simulator (see ).  It calculates the (1)
exposure time to reach a specific S/N and (2) S/N for a given exposure time.

*Authors: Greg Walth (UC San Diego)*
*Contributors: Andrey Vayner (UC San Diego)*

## Requirements:
* Python 2.7
* Numpy
* Scipy
* Astropy
* Photutils 0.4
* Matplotlib

### Additional requirements:
IRIS_ancillary_files.tar.gz which is contains the following directories; psfs, model_spectra and skyspectra.  The
directories contain binary files for the IRIS ETC calculation.

## Installation and setup:
Set the simdir in the config.ini to directory that contains the PSFs and
ancillary data.

The expected directory structure within the simdir is the following:

psfs

model_spectra

skyspectra

info


## Examples:
`iris_snr_sim.py -mag 20.0 -filter K -scale 0.004 -mode imager -calc snr -nframes 2`

`iris_snr_sim.py -mag 20.0 -filter K -scale 0.004 -mode imager -calc exptime -snr 10`

`iris_snr_sim.py -mag 20.0 -filter K -scale 0.004 -mode IFS -calc snr -snr 50.0`

`iris_snr_sim.py -mag 0.0 -filter K -scale 0.004 -mode IFS -calc snr -nframes 1 -spectrum vega_all.fits`

`iris_snr_sim.py -mag 0.0 -filter K -scale 0.004 -mode IFS -calc exptime -snr 10 -spectrum vega_all.fits`


## Usage examples:
Imager mode
`iris_snr_sim.py -mag 20.0 -filter K -scale 0.004 -mode imager -calc snr -nframes 2`
`iris_snr_sim.py -mag 20.0 -filter K -scale 0.004 -mode imager -calc exptime -snr 10`
IFS mode
Case 1 (Vega or Flat spectrum)
`iris_snr_sim.py -mag 20.0 -filter K -scale 0.004 -mode IFS -calc exptime -snr 50.0`
`iris_snr_sim.py -mag 20.0 -filter K -scale 0.004 -mode IFS -calc snr -nframes 1 -spectrum Vega`
`iris_snr_sim.py -mag 20.0 -filter K -scale 0.004 -mode IFS -calc exptime -snr 10 -spectrum Vega`
Case 2 (Emission line spectrum)
`iris_snr_sim.py -mag 20.0 -filter K -scale 0.004 -mode IFS -calc snr -nframes 1 -spectrum Emission -line-width 100 -wavelength 2.15`
`iris_snr_sim.py -mag 20.0 -filter K -scale 0.004 -mode IFS -calc exptime -snr 10 -spectrum Emission -line-width 100 -wavelength 2.15`

PSFs
Imager mode
`iris_snr_sim.py -mag 20.0 -filter K -scale 0.004 -mode imager -calc snr -nframes 2 -zenith-angle 45 -atm-cond 75 -psf-loc 0.6 12.`
IFS mode
`iris_snr_sim.py -mag 20.0 -filter K -scale 0.004 -mode IFS -calc exptime -snr 50.0 -zenith-angle 30 -atm-cond 25 -psf-loc 0. 0.`
