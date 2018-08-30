# TMT IRIS ETC

Simple TMT IRIS ETC based on IRIS simulator.  It calculates the (1)
exposure time to reach a specific S/N and (2) S/N for a given exposure time.

## Requirements:
* Python 2.7
* Numpy
* Scipy
* Astropy
* Photutils
* Matplotlib

## Installation and setup:
Set the simdir in the config.ini to directory that contains the PSFs and
ancillary data.

The expected directory structure within the simdir is the following:

psfs

spectra

info

## Examples:
`iris_snr_sim.py -mag 20.0 -filter K -scale 0.004 -mode imager -calc snr -nframes 2`

`iris_snr_sim.py -mag 20.0 -filter K -scale 0.004 -mode imager -calc exptime -snr 10`

`iris_snr_sim.py -mag 20.0 -filter K -scale 0.004 -mode IFS -calc snr -snr 50.0`

`iris_snr_sim.py -mag 0.0 -filter K -scale 0.004 -mode IFS -calc snr -nframes 1 -spectrum vega_all.fits`

`iris_snr_sim.py -mag 0.0 -filter K -scale 0.004 -mode IFS -calc exptime -snr 10 -spectrum vega_all.fits`
