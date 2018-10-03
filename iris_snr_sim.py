#!/usr/bin/env python

# SNR equation:
# S/N = S*sqrt(T)/sqrt(S + npix(B + D + R^2/t))
# t = itime per frame
# T = sqrt(Nframes*itime)
# S, B, D -> electrons per second
# R -> read noise (electrons)


import argparse, os, sys
from math import log10,ceil,sqrt,log
import ConfigParser   # Python 2.7?
#import configparser
import json

import numpy as np
from scipy import integrate,interpolate

from astropy.io import fits
from astropy.modeling import models

from photutils import aperture_photometry
from photutils import CircularAperture, SkyCircularAperture
from photutils.background import Background2D

import photutils
#print photutils.__version__

import matplotlib.pyplot as plt

# constants
c_km = 2.9979E5      # km/s

c = 2.9979E10       # cm/s
h = 6.626068E-27    # cm^2*g/s
k = 1.3806503E-16   # cm^2*g/(s^2*K)

Ang = 1E-8          # cm
mu = 1E-4           # cm


# IRIS interal packages
from get_filterdat import get_filterdat
#from background_specs import background_specs2
from background_specs import background_specs3
from get_psf import get_psf

def extrap1d(interpolator):
    xs = interpolator.x
    ys = interpolator.y

    def pointwise(x):
        if x < xs[0]:
            return ys[0]+(x-xs[0])*(ys[1]-ys[0])/(xs[1]-xs[0])
        elif x > xs[-1]:
            return ys[-1]+(x-xs[-1])*(ys[-1]-ys[-2])/(xs[-1]-xs[-2])
        else:
            return interpolator(x)

    def ufunclike(xs):
        return np.array(map(pointwise, np.array(xs)))

    return ufunclike


def IRIS_ETC(filter = "K", mag = 21.0, flambda=1.62e-19, itime = 1.0,
             nframes = 1, snr = 10.0, radius = 0.024, gain = 1.0, 
             readnoise = 5., darkcurrent = 0.002, scale = 0.004,
             resolution = 4000, collarea = 630.0, positions = [0, 0],
             bgmag = None, efftot = None, mode = "imager", calc = "snr",
             spectrum = "Vega", lam_obs = 2.22, line_width = 200.,
             png_output = None, zenith_angle = 30. , atm_cond = 50.,
             psf_loc = [8.8, 8.8], psf_time = 1.4, verb = 1, psf_old = 0, 
             simdir='~/data/iris/sim/', psfdir='~/data/iris/sim/'):

    # KEYWORDS: filter - broadband filter to use (default: 'K')
    #           mag - magnitude of the point source
    #           itime - integration time per frame (default: 900 s) 
    #           nframes - number of observations (default: 1)
    #           snr - signal-to-noise of source
    #           radius - aperture radius in arcsec
    #           gain - gain in e-/DN
    #           readnoise - read noise in e-
    #           darkcurrent - dark current noise in e-/s
    #           scale - pixel scale (default: 0.004"), sets the IFS mode: slicer or lenslet
    #           imager - calculate the SNR if it's the imager
    #           collarea - collecting area (m^2) (TMT 630, Keck 76)
    #           positions - position of point source
    #           bgmag  - the background magnitude (default: sky
    #                    background corresponding to input filter)
    #           efftot - total throughput 
    #           verb - verbosity level

    #           mode - either "imager" or "ifs"
    #           calc - either "snr" or "exptime"
    
    radius /= scale
  
    if spectrum.lower() == "vega": 
       spectrum = "vega_all.fits"
       #spectrum = "spec_vega.fits"

    
    ##### READ IN FILTER INFORMATION
    filterdat = get_filterdat(filter, simdir)
    lambdamin = filterdat["lambdamin"]
    lambdamax = filterdat["lambdamax"]
    lambdac = filterdat["lambdac"]
    bw = filterdat["bw"]
    filterfile = os.path.expanduser(simdir + filterdat["filterfiles"][0])
    #print filterfile
    
    #print lambdamin
    #print lambdamax
    #print lambdac
    #print bw
    
    ## Determine the length of the cube, dependent on filter 
    ## (supporting only BB for now)
    dxspectrum = 0
    wi = lambdamin # Ang
    wf = lambdamax # Ang
    #resolution = resolution*2.  
    dxspectrum = int(ceil( log10(wf/wi)/log10(1.0+1.0/(resolution*2.0)) ))
    ## resolution times 2 to get nyquist sampled 
    crval1 = wi/10.                      # nm
    cdelt1 = ((wf-wi) / dxspectrum)/10.  # nm/channel
    #print dxspectrum


    # Throughput calculation    
    if efftot is None: 

        # Throughput number from Ryuji (is there wavelength dependence?)
        #teltot = 0.91 # TMT total throughput
        #aotot = 0.80  # NFIRAOS AO total throughput
        teltot = 1.0
        aotot = 1.0 

        wav  = [830,900,2000,2200,2300,2412] # nm
        
        if mode == "imager":
           tput = [0.631,0.772,0.772,0.813,0.763,0.728] # imager
           if verb > 1: print 'IRIS imager selected!!!'
        
        else:
           if (scale == 0.004) or (scale == 0.009):
              tput = [0.340,0.420,0.420,0.490,0.440,0.400] # IFS lenslet
              if verb > 1: print 'IRIS lenslet selected!!!'
           else:
              tput = [0.343,0.465,0.465,0.514,0.482,0.451] # IFS slicer
              if verb > 1: print 'IRIS slicer selected!!!'
        
        #print tput
        w = (np.arange(dxspectrum)+1)*cdelt1 + crval1  # compute wavelength
        #print,lambda
        #####################################################################
        # Interpolating the IRIS throughputs from the PDR-1 Design Description
        # Document (Table 7, page 54)
        #####################################################################
        R = interpolate.interp1d(wav,tput)
        eff_lambda = [R(w) for w0 in w]
        #print eff_lambda
        
        ###############################################################
        # MEAN OF THE INSTRUMENT THROUGHPUT BETWEEN THE FILTER BANDPASS
        ###############################################################
        instot = np.mean(eff_lambda)
        #efftot = instot

        efftot = instot*teltot*aotot
    
    if verb > 1: print  ' '
    #print  'IRIS efficiency ', efftot
    if verb > 1: print  'Total throughput (TMT+NFIRAOS+IRIS) = %.3f' % efftot
    
    
    if bgmag:
       backmag = bgmag
    else:
       backmag = filterdat["backmag"] #background between OH lines
       imagmag = filterdat["imagmag"] #integrated BB background
       if mode == "imager": backmag = imagmag ## use the integrated background if specified
    zp = filterdat["zp"]

    # test case
    # PSFs
    #print lambdac

    if psf_old:
    
         psf_dict = { 928:"psf_x0_y0_wvl928nm_implKOLMO117nm_bin4mas_sm.fits",
                     1092:"psf_x0_y0_wvl1092nm_implKOLMO117nm_bin4mas_sm.fits",
                     1270:"psf_x0_y0_wvl1270nm_implKOLMO117nm_bin4mas_sm.fits",
                     1629:"psf_x0_y0_wvl1629nm_implKOLMO117nm_bin4mas_sm.fits",
                     2182:"psf_x0_y0_wvl2182nm_implKOLMO117nm_bin4mas_sm.fits"}
         
         psf_wvls = psf_dict.keys()
         
         psf_ind = np.argmin(np.abs(lambdac/10. - psf_wvls))
         psf_wvl =  psf_wvls[psf_ind]
         #psf_file = os.path.expanduser(simdir + "/psfs/" + psf_dict[psf_wvl])
         psf_file = os.path.expanduser(psfdir + "/psfs/results_central/" + psf_dict[psf_wvl])
         ext = 0

         #print psf_ind
         #print psf_wvl
         #print psf_file
        
    else:
        if mode.lower() == "ifs":
            psf_wvls = [840, 928, 1026, 988, 1092, 1206, 1149, 1270, 1403, 1474,
                        1629, 1810, 1975, 2182, 2412] # nm
        if mode.lower() == "imager":
            psf_wvls = [830, 876, 925, 970, 1019, 1070, 1166, 1245, 1330, 1485,
                        1626, 1781, 2000, 2191, 2400] # nm

        #print psf_loc
        #print zenith_angle
        #print atm_cond
        psf_file = get_psf(zenith_angle, atm_cond, mode, psf_time, psf_loc, scale)
        #print psf_file
        psf_file = os.path.expanduser(psfdir + "/psfs/" + psf_file)

        #print psf_file
        #print os.path.isfile(psf_file) 

        psf_ind = np.argmin(np.abs(lambdac/10. - psf_wvls))
        psf_wvl =  psf_wvls[psf_ind]
        ext = psf_ind

        #print lambdac
        #print psf_ind
        #print psf_wvl

        # FITS header comments contain wavelength information
        # for i in xrange(len(pf)): print pf[i].header["COMMENT"]

        # truncate PSF with iamge size goes down to 1e-6 
        # check in log scale
        # 350 - 1150



    pf = fits.open(psf_file)
    #print len(pf)
    image = pf[ext].data
    head = pf[ext].header
    #print head['COMMENT']

    # position of center of PSF
    x_im_size,y_im_size = image.shape
    hw_x = x_im_size/2
    hw_y = y_im_size/2

    #print hw_x,hw_y

    #sys.exit()
    
    #print image.sum()
    #image /= image.sum()

    flux_phot = zp*10**(-0.4*mag) # photons/s/m^2

    #########################################################################
    #########################################################################
    # comparison:
    #########################################################################
    # http://ssc.spitzer.caltech.edu/warmmission/propkit/pet/magtojy/
    # input: Johnson
    #        20 K-banda
    #        2.22 micron
    # output:
    #        6.67e-29 erg/s/cm^2/Hz
    #        4.06e-19 erg/s/cm^2/Ang
    #########################################################################

    #  mag = ABmag - 0.91 ; Vega magnitude

    # convert AB to Vega
             # band  eff     mAB - mVega
    ABconv = [["i",  0.7472, 0.37 ],
              ["z",  0.8917, 0.54 ],
              ["Y",  1.0305, 0.634],
              ["J",  1.2355, 0.91 ],
              ["H",  1.6458, 1.39 ],
              ["Ks", 2.1603, 1.85 ]]

    ABwave  = [i[1] for i in ABconv]
    ABdelta = [i[2] for i in ABconv]
    #print

    #print filter

    if verb > 1:
        fig = plt.figure()
        p = fig.add_subplot(111)
        p.plot(ABwave, ABdelta)
        p.set_xlabel("Wavelength ($\mu$m)")
        p.set_ylabel("m$_{\\rm AB}$ - m$_{\\rm Vega}$")
        plt.show()


    R_i = interpolate.interp1d(ABwave,ABdelta)
    R_x = extrap1d(R_i)
    delta = R_x(lambdac/1e4)

    #print delta

    # delta = mAB - mVega
    ABmag = mag + delta
    #print ABmag
    #################################################################

    fnu = 10**(-0.4*(ABmag + 48.60))                 # erg/s/cm^2/Hz
    #print "Calculated from magnitude"
    #print fnu,"erg/s/cm^2/Hz"
    #flambda = fnu*Ang/((lam_obs*mu)**2/c)
    flambda = fnu*Ang/((lambdac*Ang)**2/c)
    #print flambda,"erg/s/cm^2/Ang"
    #nu = c/(lam_obs*mu)
    #print nu,"Hz"
    #dnu = nu/(2*resolution)
    #dlambda = lam_obs*1e4/(2*resolution)
    #dlambda = (wf-wi) # Ang
    #print dlambda, "Ang"
    #print dnu, "Hz"
    #print fnu*dnu,"erg/s/cm^2"
    #print flambda*lambdac,"erg/s/cm^2"
    E_phot = (h*c)/(lambdac*Ang) # erg
    #print flambda*lambdac/E_phot,"photons/s/cm^2"
    #print flambda/E_phot,"photons/s/cm^2/Ang"
    # above is correct!!
    ########################################################################
    if verb > 1:
        # Vega test spectrum
        if spectrum == "vega_all.fits":
            ext = 0 
            spec_file = os.path.expanduser(simdir + "/model_spectra/" + spectrum)
            pf = fits.open(spec_file)
            spec = pf[ext].data
            head = pf[ext].header
            cdelt1 = head["cdelt1"]
            crval1 = head["crval1"]
        
            nelem = spec.shape[0]
            specwave = (np.arange(nelem))*cdelt1 + crval1  # Angstrom
            #spec /= 206265.**2

        elif spectrum == "spec_vega.fits":
            ext = 0 
            spec_file = os.path.expanduser(simdir + "/model_spectra/" + spectrum)
            pf = fits.open(spec_file)
            specwave = pf[ext].data[0,:] # Angstrom
            spec = pf[ext].data[1,:]     # erg/s/cm^2/Ang
            nelem = spec.shape[0]
            
        E_phot = (h*c)/(specwave*Ang) # erg
        #print specwave
        #print E_phot
 
        fig = plt.figure()
        p = fig.add_subplot(111)
        p.plot(specwave, spec/E_phot) # photons/s/cm^2/Ang
        p.set_xlim(3000,11000)
        #p.set_xlim(20000,24000)
        #p.set_ylim(0,2000)
        p.set_xlabel("Wavelength ($\AA$)")
        p.set_ylabel("Flux (photons cm$^{-2}$ s$^{-1}$ $\AA^{-1}$)")
        p.set_title("Vega photon spectrum")


        # ABnu = STlamb @ 5492.9 Ang
        STlamb = 3.63E-9*np.ones((nelem)) # erg/cm^2/s/Ang   STlamb = 0
        ABnu = 3.63E-20*np.ones((nelem))  # erg/cm^2/s/Hz    ABnu = 0

        p.plot(specwave,STlamb/E_phot)
        p.plot(specwave,ABnu/E_phot*(c/(specwave*Ang)**2)*Ang)

        plt.show()


    #print
    #print
    #print "Calculated from IRIS zeropoints"
    E_phot = (h*c)/(lambdac*Ang) # erg
    flux = flux_phot*E_phot*(1./(100*100)) # erg/s/cm^2
    # convert from m**2 to cm**2
    #print flux,"erg/s/cm^2"
    #print flux_phot*(1./(100*100)),"photons/s/cm^2"

    ##print flux_phot/dnu,"photons/s/cm^2/Hz"
    #print flux_phot*(1./(100*100))/lambdac,"photons/s/cm^2/Ang"
    #print flux_phot,"photons/s/m^2"
    #print flux_phot*collarea,"photons/s"
    #print flux_phot*collarea*efftot,"photons/s"

    #print flux/dnu,"erg/s/cm^2/Hz"
    #print flux/dlambda,"erg/s/cm^2/Ang"

    if verb > 1: print
    #########################################################################
    #########################################################################

    #ymax,xmax = image.shape
    #print xmax,ymax
    if mode.lower() == "ifs": 
        #hwbox = 25
        hwbox = 10
    elif mode == "imager":
        #hwbox = 239
        hwbox = 100
    # write check for hwbox boundary




    if verb > 1: print image.shape
    if 0: 
        # original code
        xc,yc = [hw_x,hw_y]
        xs = xc
        ys = yc
        subimage = image

    else:

        # center coordinates
        #xp,yp = positions
        xc,yc = positions
       
        # image coordinates 
        xp = xc + hw_x
        yp = yc + hw_y
        # 239 x 239 is the shape of the PSF image

        # subimage coordinates
        xs = xc + hwbox
        ys = yc + hwbox

        subimage = image[yp-hwbox:yp+hwbox+1,xp-hwbox:xp+hwbox+1]
        #print xc,yc
        #print xp,yp
        #print xs,ys

    
    # normalize by the full PSF image
    subimage /= image.sum()

    # to define apertures used throughout the calculations
    radii = np.arange(1,50,1) # pixels
    apertures = [CircularAperture([xs,ys], r=r) for r in radii]
    aperture = CircularAperture([xs,ys], r=radius)

    masks = aperture.to_mask(method='center')
    mask = masks[0]

    ###########################################################################
    ###########################################################################
    # IFS MODE
    ###########################################################################
    ###########################################################################
    if mode.lower() == "ifs":

        #bkgd = background_specs2(resolution*2.0, filter, convolve=True, simdir = simdir)
        bkgd = background_specs3(resolution*2.0, filter, convolve=True, simdir = simdir,
                                 filteronly=True)

        ohspec = bkgd.backspecs[0,:]
        cospec = bkgd.backspecs[1,:]
        bbspec = bkgd.backspecs[2,:]

        ohspectrum = ohspec*scale**2.0  ## photons/um/s/m^2
        contspectrum = cospec*scale**2.0
        bbspectrum = bbspec*scale**2.0
      
        backtot = ohspectrum + contspectrum + bbspectrum
      
        if verb >1:
           print 'mean OH: ', np.mean(ohspectrum)
           print 'mean continuum: ', np.mean(contspectrum)
           print 'mean bb: ', np.mean(bbspectrum)
           print 'mean background: ', np.mean(backtot)

        backwave    = bkgd.waves/1e4

        wave = np.linspace(wi/1e4,wf/1e4,dxspectrum)
        #print dxspectrum
        #print len(wave)
        #print len(backwave)
        #print wave
        #print backwave

        backtot_func = interpolate.interp1d(backwave,backtot)
        backtot = backtot_func(wave)

        #print
        #print "Flux = %.2e photons/s/m^2" % flux_phot

        #print "Image sum = %.1f" % subimage.sum()
        #print subimage.shape
        #print subimage.size

        if spectrum.lower() == "flat":
            spec_temp = np.ones(dxspectrum)
            intFlux = integrate.trapz(spec_temp,wave)
            intNorm = flux_phot/intFlux
            #print "Spec integration = %.1f" % intFlux
            #print "Spec normalization = %.4e" % intNorm

        elif spectrum.lower() == "emission":
            specwave = wave
            lam_width=lam_obs/c_km*line_width
            instwidth = (lam_obs/resolution)
            width = np.sqrt(instwidth**2+lam_width**2)
            A = flux_phot/(width*np.sqrt(2*np.pi))  #  photons/s/m^2/micron
            #print flux_phot
            #print width*1e4
            #print A
            spec_temp = A*np.exp(-0.5*((specwave - lam_obs)/width)**2.)
            intFlux = integrate.trapz(spec_temp,specwave)
            intNorm = flux_phot/intFlux
            #print "Spec integration = %.1f" % intFlux
            #print "Spec normalization = %.4e" % intNorm

        else:
            if spectrum == "vega_all.fits":
                ext = 0 
                spec_file = os.path.expanduser(simdir + "/model_spectra/" + spectrum)
                pf = fits.open(spec_file)
                spec = pf[ext].data
                head = pf[ext].header
                cdelt1 = head["cdelt1"]
                crval1 = head["crval1"]
            
                nelem = spec.shape[0]
                specwave = (np.arange(nelem))*cdelt1 + crval1  # Angstrom
                specwave /= 1e4 # -> microns

            elif spectrum == "spec_vega.fits":
                ext = 0 
                spec_file = os.path.expanduser(simdir + "/model_spectra/" + spectrum)
                pf = fits.open(spec_file)
                specwave = pf[ext].data[0,:] # Angstrom
                spec = pf[ext].data[1,:]     # erg/s/cm^2/Ang
                nelem = spec.shape[0]
                
                E_phot = (h*c)/(specwave*Ang) # erg
                spec *= 100*100*1e4/E_phot # -> photons/s/m^2/um
                specwave /= 1e4 # -> microns


            #intFlux = integrate.trapz(spec,specwave,dx=dx)
            #intFlux = integrate.simps(spec,specwave)
            #intNorm = flux_phot/intFlux
            #print
            #print "Spec integration = %.2e" % intFlux
            #print "Spec normalization = %.4e" % intNorm
            #print

            if verb > 1:
                fig = plt.figure()
                p = fig.add_subplot(111)
                p.plot(specwave, spec)
                #p.set_xscale("log")
                #p.set_yscale("log")
                plt.show()

            ################################################
            # convolve with the resolution of the instrument
            ################################################
            delt = 2.0*(wave[1]-wave[0])/(specwave[1]-specwave[0])
            #delt = 0
            if delt > 1:
            
               stddev = delt/2*sqrt(2*log(2))
               psf_func = models.Gaussian1D(amplitude=1.0, stddev=stddev)
               x = np.arange(4*int(delt)+1)-2*int(delt)
               psf = psf_func(x)
               psf /= psf.sum() # normalize
            
               spec = np.convolve(spec, psf,mode='same')

            spec_func = interpolate.interp1d(specwave,spec)
            spec_temp = spec_func(wave)

            intFlux = integrate.trapz(spec_temp,wave)
            #intFlux = integrate.simps(spec_temp,wave)
            intNorm = flux_phot/intFlux
            #print
            #print "Spec integration = %.2e" % intFlux
            #print "Spec normalization = %.4e" % intNorm
            #print

            #spec_norm = np.mean(spec_temp)
            #spec_temp /= spec_norm


        # essentially the output of mkpointsourcecube
        cube = (subimage[np.newaxis]*spec_temp[:,np.newaxis,np.newaxis]).astype(np.float32)
        # photons/s/m^2/um
        cube = intNorm*cube
        #print "Cube sum = %.2e photons/s/m^2/um" % cube.sum()
        #print "Cube mean = %.2e photons/s/m^2/um" % cube.mean()

        if verb > 1: print

        newcube = np.ones(cube.shape)
        #print newcube.sum()
        #print cube.shape
        #print cube.size

        if verb > 2:
            # [electrons]
            hdu = fits.PrimaryHDU(cube)
            hdul = fits.HDUList([hdu])
            hdul.writeto('cube.fits',clobber=True)
      

        # convert the signal and the background into photons/s observed
        # with TMT
        observedCube = cube*collarea*efftot    # photons/s/um
        backtot = backtot*collarea*efftot       # photons/s/um

        # get photons/s per spectral channel, since each spectral
        # channel has the same bandwidth
        if verb > 1: print "Observed cube sum = %.2e photons/s/um" % observedCube.sum()
        if verb > 1: print "Background cube sum = %.2e photons/s/um" % backtot.sum()
        #print "Observed cube mean = %.2e photons/s/um" % observedCube.mean()

        if verb > 1: print
        observedCube = observedCube*(wave[1]-wave[0])
        backtot = backtot*(wave[1]-wave[0])
        if verb > 1: print "dL = %f micron" % (wave[1]-wave[0])
        if verb > 1: print "Observed cube sum = %.2e photons/s" % observedCube.sum()
        if verb > 1: print "Background cube sum = %.2e photons/s" % backtot.sum()
        #print "Observed cube mean = %.2e photons/s" % observedCube.mean()

        if verb > 1: print


        ##############
        # filter curve
        ##############
        # not needed until actual filters are known
        #filterdata = np.loadtxt(filterfile)
        #filterwav = filterdata[:,0]       # micron
        #filtertran = filterdata[:,1]      # transmission [fraction]
        #filter_norm = np.max(filtertran)
        #print filter_norm
        #filtertran /= filter_norm
        #filter_func = interpolate.interp1d(filterwav,filtertran)
        #filter_tput = filter_func(wave)

        if verb > 1:
            fig = plt.figure()
            p = fig.add_subplot(111)
            #p.plot(wave, filter_tput*cube[:,ys,xs])
            p.plot(wave, cube[:,ys,xs],c="k")
            p.plot(wave, np.sum(cube,axis=(1,2)),c="b")
            plt.show()

        if verb > 1: 
            print 'n wavelength channels: ', len(wave)
            print 'channel width (micron): ', wave[1]-wave[0]
            print 'mean flux input cube center (phot/s/m^2/micron): %.2e' % np.mean(cube[:, ys, xs])
            print 'mean counts/spectral channel input cube center (phot/s): %.2e' % np.mean(observedCube[:, ys, xs])
            print 'mean background (phot/s): ', np.mean(backtot)
        #print "CORRECT ABOVE"
        
        backgroundCube = np.broadcast_to(backtot[:,np.newaxis,np.newaxis],cube.shape)
        #print backgroundCube
        if verb > 1: print backgroundCube.shape



        ### Calculate total noise number of photons from detector
        #darknoise = (sqrt(darkcurrent*itime)) * (1/sqrt(coadds))
        #readnoise = sqrt(coadds)*readnoise
        #darknoise = (sqrt(darkcurrent*itime)) 
        darknoise = darkcurrent       ## electrons/s
        readnoise = readnoise**2.0/itime  ## scale read noise
        
                                        # total noise per pixel
        # noisetot = sqrt((readnoise*readnoise) + (darknoise*darknoise))
        noisetot = darknoise + readnoise
        noise = noisetot
        ### Combine detector noise and background (sky+tel+AO)
        #noisetotal = SQRT(noise*noise + background*background)
        noisetotal = noise + backgroundCube



        ####################################################
        # Case 1: find s/n for a given exposure time and mag
        ####################################################
        if calc == "snr":
            if verb > 1: print "Case 1: find S/N for a given exposure time and mag"
        
            signal = observedCube*np.sqrt(itime*nframes)  # photons/s
            # make a background cube and add noise
            # noise = sqrt(S + B + R^2/t)
            #noiseCube = np.sqrt(observedCube+backgroundCube+darkcurrent+readnoise**2.0/itime)
            noiseCube = np.sqrt(observedCube+noisetotal)
            
            # SNR cube  = S*sqrt(itime*nframes)/sqrt(S + B+ R^2/t)
            ##snrCube = observedCube*nframes*itime/rmsNoiseCube
            #snrCube = observedCube*sqrt(itime*nframes)/noiseCube
            #snrCube = float(snrCube)
            snrCube = signal/noiseCube

            if verb > 2:
                hdu = fits.PrimaryHDU(snrCube)
                hdul = fits.HDUList([hdu])
                hdul.writeto('snrCube.fits',clobber=True)


          
            data_cutout = []
            data_cutout_aper = []
            for n in xrange(dxspectrum): 
                data_cutout.append(mask.cutout(snrCube[n,:,:]))
                if photutils.__version__ == "0.4":
                    data_cutout_tmp = mask.multiply(snrCube[n,:,:]) # in version 0.4 of photutils
                else:
                    data_cutout_tmp = mask.apply(snrCube[n,:,:])
                data_cutout_aper.append(data_cutout_tmp)

                #if verb > 0 and n == 0:
                #    fig = plt.figure()
                #    p = fig.add_subplot(111)
                #    p.imshow(snrCube[n,:,:],interpolation='none')
                #    plt.show()

            data_cutout = np.array(data_cutout)
            data_cutout_aper = np.array(data_cutout_aper)
            if verb > 1: print data_cutout.shape
            if verb > 1: print data_cutout_aper.shape
	    peakSNR=""
            medianSNR=""
            meanSNR=""
	    minexptime=""
	    medianexptime=""
	    meanexptime=""	
            ###############
            # Main S/N plot
            ###############
            if verb > 0:
                fig = plt.figure()
                p = fig.add_subplot(111)
                #p.plot(wave, filter_tput*cube[:,ys,xs])
                p.plot(wave, snrCube[:,ys,xs],c="k",label="Peak flux")
                p.plot(wave, np.mean(data_cutout_aper,axis=(1,2)),label="Mean flux")
                p.plot(wave, np.median(data_cutout_aper,axis=(1,2)),label="Median flux")
                p.set_xlabel("Wavelength ($\mu$m)")
                p.set_ylabel("S/N")
                leg = p.legend(loc=1,numpoints=1)
                if png_output:
                    fig.savefig(png_output)
                else:
                    plt.show()
            
            #print data_cutout.shape
            #print data_cutout_aper.shape
            
            #;; save the SNR cube if given
            #if n_elements(outcube) ne 0 then begin
            #   mkosiriscube, wave, transpose(snrCube, [2, 1, 0]), outcube, /micron, scale = scale, units = 'SNR', params = fitsParams, values = fitsValues
            #endif
            #return, snrCube

            ## total noise in photons
            ## rmsNoiseCube = sqrt(observedCube*itime*nframes + noiseCube*itime*nframes + darkcurrent*itime*nframes + readnoise^2.0*nframes)
            
            ### original code from Tuan, probably not correct 
            #rmsNoiseCube = (observedCube*itime*nframes + backgroundCube*itime*nframes + darkcurrent*itime*nframes + readnoise**2.0*nframes)
            #simNoiseCube = np.random.poisson(lam=rmsNoiseCube, size=rmsNoiseCube.shape).astype("float64")
            #totalObservedCube = observedCube*nframes*itime + simNoiseCube
            
            #for ii = 0, s[3] - 1 do begin
            #   for jj = 0, s[2]-1 do begin
            #      for kk = 0, s[1]-1 do begin
            #         simNoiseCube[kk, jj, ii] = randomu(seed2, poisson = rmsNoiseCube[kk, jj, ii], /double)
            #      endfor
            #   endfor
            #endfor
            
            totalObservedCube = observedCube*itime*nframes + backgroundCube*itime*nframes + darkcurrent*itime*nframes + readnoise**2.0*nframes
            # model + background + noise
            # [electrons]
            simCube_tot = np.random.poisson(lam=totalObservedCube, size=totalObservedCube.shape).astype("float64")
            # divide back by total integration time to get the simulated image
            simCube = simCube_tot/(itime*nframes) # [electrons/s]
            simCube_DN = simCube_tot/gain # [DNs]
            
            if verb > 2:
                # [electrons]
                hdu = fits.PrimaryHDU(simCube_tot)
                hdul = fits.HDUList([hdu])
                hdul.writeto('simCube_tot.fits',clobber=True)
            
                # [electrons/s]
                hdu = fits.PrimaryHDU(simCube)
                hdul = fits.HDUList([hdu])
                hdul.writeto('simCube.fits',clobber=True)
            
                # [DNs]
                hdu = fits.PrimaryHDU(simCube_DN)
                hdul = fits.HDUList([hdu])
                hdul.writeto('simCube_DN.fits',clobber=True)
            
            
            #totalObservedCube = float(totalObservedCube)
            #;; save the file
            #if not(keyword_set(quiet)) then begin
            #   print, '% IRIS_SIM_SNR: '
            #   print, 'saving simulated observed cube: ', simcube
            #endif
            
            #mkosiriscube, wave, transpose(totalObservedCube, [2, 1, 0]), simcube, /micron, scale = scale, units = 'phot', params = fitsParams, values = fitsValues
            
            #if n_elements(savesky) ne 0  then begin
            #   mkosiriscube, wave, transpose(simNoiseCube, [2, 1, 0]), savesky, /micron, scale = scale, units = 'phot', params = fitsParams, values = fitsValues
            #endif

            
        #######################################################
        # Case 2: find integration time for a given s/n and mag
        #######################################################
        elif calc == "exptime":
            if verb > 1: print "Case 2: find integration time for a given S/N and mag"
            
            # snr = observedCube*np.sqrt(itime*nframes)/np.sqrt(observedCube+noisetotal)
            # itime * nframes =  (snr * np.sqrt(observedCube+noisetotal)/observedCube)**2
            
            totime =  (snr * np.sqrt(observedCube+noisetotal)/observedCube)**2
            # totime = itime * nframes
            
            data_cutout = []
            data_cutout_aper = []
            for n in xrange(dxspectrum): 
                data_cutout.append(mask.cutout(totime[n,:,:]))

                if photutils.__version__ == "0.4":
                    data_cutout_tmp = mask.multiply(totime[n,:,:]) # in version 0.4 of photutils
                else:
                    data_cutout_tmp = mask.apply(totime[n,:,:])
                data_cutout_aper.append(data_cutout_tmp)

                #if verb > 0 and n == 0:
                #    fig = plt.figure()
                #    p = fig.add_subplot(111)
                #    p.imshow(totime[n,:,:],interpolation='none')
                #    plt.show()

            data_cutout = np.array(data_cutout)
            data_cutout_aper = np.array(data_cutout_aper)
            #print data_cutout.shape
            #print data_cutout_aper.shape
	    peakSNR=""
            medianSNR=""
            meanSNR=""
	    minexptime=""
	    medianexptime=""
	    meanexptime=""	
            ####################
            # Main exposure plot
            ####################
            if verb > 0:
                fig = plt.figure()
                p = fig.add_subplot(111)
                p.plot(wave, totime[:,ys,xs],c="k",label="Peak flux")
                p.plot(wave, np.mean(data_cutout_aper,axis=(1,2)),label="Mean flux")
                p.plot(wave, np.median(data_cutout_aper,axis=(1,2)),label="Median flux")
                p.set_xlabel("Wavelength ($\mu$m)")
                p.set_ylabel("Total Exposure Time (seconds)")
                leg = p.legend(loc=1,numpoints=1)

                if png_output:
                    fig.savefig(png_output)
                else:
                    plt.show()

            #print totime
            #print np.max(totime)
            #data_cutout = mask.cutout(totime)
            #data_cutout_aper = mask.apply(totime)
            if verb > 1: print "Min time (peak flux) = %.4f seconds" % np.min(totime)
            if verb > 1: print "Median time (median aperture flux) = %.4f seconds" % np.median(data_cutout_aper)
            if verb > 1: print "Mean time (mean aperture flux) = %.4f seconds" % np.mean(data_cutout_aper)

            if verb > 1:
                fig = plt.figure()
                p = fig.add_subplot(111)
                p.imshow(totime)
                plt.show()

            # exposure time for aperture 
            data_cutout = []
            data_cutout_aper = []
            noise_cutout = []
            noise_cutout_aper = []
            for n in xrange(dxspectrum): 
                data_cutout.append(mask.cutout(observedCube[n,:,:]))

                if photutils.__version__ == "0.4":
                    data_cutout_tmp = mask.multiply(observedCube[n,:,:]) # in version 0.4 of photutils
                    noise_cutout_tmp = mask.multiply(noisetotal[n,:,:])
                else:
                    data_cutout_tmp = mask.apply(observedCube[n,:,:])
                    noise_cutout_tmp = mask.apply(noisetotal[n,:,:])
                data_cutout_aper.append(data_cutout_tmp)

            data_cutout = np.array(data_cutout)
            data_cutout_aper = np.array(data_cutout_aper)
            noise_cutout = np.array(noise_cutout)
            noise_cutout_aper = np.array(noise_cutout_aper)

            aper_sum = data_cutout_aper.sum()
            noise_sum = np.sqrt((noise_cutout_aper**2).sum())
            totime =  (snr * np.sqrt(aper_sum+noise_sum)/aper_sum)**2
            if verb > 1: print 'Time (aperture = %.4f") = %.4f' % (2*radius*scale, totime)

    ###########################################################################
    ###########################################################################
    # IMAGER MODE
    ###########################################################################
    ###########################################################################
    else:

        # Scale by the zeropoint flux 
        subimage *= flux_phot
    
        ############################# NOISE ####################################
        # Calculate total background number of photons for whole tel aperture
        #efftot = effao*efftel*effiris #total efficiency 
        if verb > 1: print 'background magnitude: ', backmag
        phots_m2 = (10**(-0.4*backmag)) * zp # phots per sec per m2
        #print phots_m2
        
        # divide by the number of spectral channels if it's not an image
        phots_tel = phots_m2 * collarea     # phots per sec for TMT
        #phots_int = phots_tel               # phots per sec
        #phots_chan = phots_int 
        # phots from background per square arcsecond through the telescope
        phots_back = efftot*phots_tel
        #background = sqrt(phots_back*scale*scale) #photons from background per spaxial^2
        background = phots_back*scale*scale #photons from background per spaxial^2

        ###################################################################################################
        ### Calculate total noise number of photons from detector
        #darknoise = (sqrt(darkcurrent*itime)) * (1/sqrt(coadds))
        #readnoise = sqrt(coadds)*readnoise
        #darknoise = (sqrt(darkcurrent*itime)) 
        darknoise = darkcurrent       ## electrons/s
        readnoise = readnoise**2.0/itime  ## scale read noise
        
                                        # total noise per pixel
        # noisetot = sqrt((readnoise*readnoise) + (darknoise*darknoise))
        noisetot = darknoise + readnoise
        noise = noisetot
        ### Combine detector noise and background (sky+tel+AO)
        #noisetotal = SQRT(noise*noise + background*background)
        noisetotal = noise + background
        if verb > 1: print 'detector noise (e-/s): ', noise
        if verb > 1: print 'total background noise (phot/s):', background
        if verb > 1: print '  '
        if verb > 1: print 'Total Noise (photons per pixel^2)= ', noisetotal
        ###################################################################################################
        
        ## put in the TMT collecting area and efficiency
        tmtImage = subimage*collarea*efftot
        
        if verb > 1: print
        if verb > 1: print
        if verb > 1: print
   


        ####################################################
        # Case 1: find s/n for a given exposure time and mag
        ####################################################
        if calc == "snr":

            if verb > 1: print "Case 1: find S/N for a given exposure time and mag"
        
            signal = tmtImage*np.sqrt(itime*nframes)  # photons/s
            noisemap = np.sqrt(tmtImage+noisetotal)
            
            snrMap = signal/noisemap
            #print np.max(snrMap)
            #print np.mean(snrMap)
            
            if verb > 1:
                fig = plt.figure()
                p = fig.add_subplot(111)
                #p.hist(snrMap)
            
                X = snrMap.flatten()
                x0 = np.min(X) 
                x1 = np.max(X)
                bins = 50
                n,bins,patches = p.hist(X,bins,range=(x0,x1),histtype='stepfilled',
                                        color="y",alpha=0.3)
                #n,bins,patches = p.hist(X,bins,range=(x0,x1),histtype='stepfilled',cumulative=-1, normed=1,
                #                        color="y",alpha=0.3)
                p.set_xlabel("Signal/Noise")
                p.set_ylabel("Number of pixels")
                p.set_yscale("log")
                plt.show()

            if verb > 1:
                fig = plt.figure()
                p = fig.add_subplot(111)
                p.imshow(snrMap,interpolation='none')
                plt.show()
            
            if verb > 2:
                hdu = fits.PrimaryHDU(snrMap)
                hdul = fits.HDUList([hdu])
                hdul.writeto('snrImage.fits',clobber=True)
            
            # model + background
            totalObserved = tmtImage*itime*nframes + background*itime*nframes + darkcurrent*itime*nframes + readnoise*itime*nframes
            #print totalObserved.shape
            #print totalObserved.dtype
            
            if verb > 2:
                hdu = fits.PrimaryHDU(totalObserved)
                hdul = fits.HDUList([hdu])
                hdul.writeto('new2.fits',clobber=True)
            
            # model + background + noise
            # [electrons]
            simImage_tot = np.random.poisson(lam=totalObserved, size=totalObserved.shape).astype("float64")
            #print simImage_tot.dtype
            
            # divide back by total integration time to get the simulated image
            simImage = simImage_tot/(itime*nframes) # [electrons/s]
            simImage_DN = simImage_tot/gain # [DNs]
            
            if verb > 2:
                # [electrons]
                hdu = fits.PrimaryHDU(simImage_tot)
                hdul = fits.HDUList([hdu])
                hdul.writeto('simImage_tot.fits',clobber=True)
            
                # [electrons/s]
                hdu = fits.PrimaryHDU(simImage)
                hdul = fits.HDUList([hdu])
                hdul.writeto('simImage.fits',clobber=True)
            
                # [DNs]
                hdu = fits.PrimaryHDU(simImage_DN)
                hdul = fits.HDUList([hdu])
                hdul.writeto('simImage_DN.fits',clobber=True)
            
            # Sky background counts
            #bkg_func = Background2D(simImage_DN,simImage_DN.shape)     # constant
            #print bkg_func.background
            #print bkg_func.background_median
            #print bkg_func.background_rms_median
            
            #image = mask.to_image(shape=((200, 200)))
            data_cutout = mask.cutout(snrMap)
            if photutils.__version__ == "0.4":
                data_cutout_aper = mask.multiply(snrMap) # in version 0.4 of photutils
            else:
                data_cutout_aper = mask.apply(snrMap)
            peakSNR=str("%0.4f" %np.max(snrMap))
            medianSNR=str("%0.4f" %np.median(data_cutout_aper))
            meanSNR=str("%0.4f" %np.mean(data_cutout_aper))
	    minexptime=""
	    medianexptime=""
	    meanexptime=""
            #print np.min(snrMap)
            if verb > 1: print "Peak S/N = %.4f" % np.max(snrMap)
            if verb > 1: print "Median S/N = %.4f" % np.median(data_cutout_aper)
            if verb > 1: print "Mean S/N = %.4f" % np.mean(data_cutout_aper)
            
            if verb > 1:
                fig = plt.figure()
                p = fig.add_subplot(111)
                p.imshow(data_cutout_aper,interpolation='none')
                plt.show()
            
            phot_table = aperture_photometry(signal, aperture, error=noisemap)
            #print phot_table
            snr_int = phot_table["aperture_sum"].quantity/phot_table["aperture_sum_err"].quantity
            if verb > 1: print 'S/N (aperture = %.4f") = %.4f' % (2*radius*scale, snr_int[0])
            
            if verb > 1:
                phot_table = aperture_photometry(signal, apertures, error=noisemap)
                dn     = np.array([phot_table["aperture_sum_%i" % i] for i in range(len(radii))])
                dn_err = np.array([phot_table["aperture_sum_err_%i" % i] for i in range(len(radii))])
                #print phot_table
           
                fig = plt.figure()
                p = fig.add_subplot(111)
                p.errorbar(radii,dn,yerr=dn_err)
                #p.scatter(radii,dn)
                p.set_xlabel("Aperture radius [pixels]")
                p.set_ylabel("Counts [photons/s/aperture]")
                plt.show()
            
            #simImage = dblarr(s[1], s[2])
            #for i = 0, s[1]-1 do begin
            #   for j = 0, s[2]-1 do begin
            #      simImage[i, j] = randomn(seed, poisson = totalObserved[i, j], /double)
            #   endfor
            #endfor
            
            if verb > 1: print
            if verb > 1: print
            if verb > 1: print
    
        #######################################################
        # Case 2: find integration time for a given s/n and mag
        #######################################################
        elif calc == "exptime":

            if verb > 1: print "Case 2: find integration time for a given S/N and mag"
            
            # snr = tmtImage*np.sqrt(itime*nframes)/np.sqrt(tmtImage+noisetotal)
            # itime * nframes =  (snr * np.sqrt(tmtImage+noisetotal)/tmtImage)**2
            
            totime =  (snr * np.sqrt(tmtImage+noisetotal)/tmtImage)**2
            # totime = itime * nframes
            

            #print totime
            #print np.max(totime)
            data_cutout = mask.cutout(totime)
            if photutils.__version__ == "0.4":
                data_cutout_aper = mask.multiply(totime) # in version 0.4 of photutils
            else:
                data_cutout_aper = mask.apply(totime)
            minexptime= str("%0.4f" %np.min(totime))
            medianexptime= str("%0.4f" %np.median(data_cutout_aper))
            meanexptime=  str("%0.4f" %np.mean(data_cutout_aper))
	    peakSNR=""
            medianSNR=""
            meanSNR=""                
            if verb > 1: print "Min time (peak flux) = %.4f seconds" % np.min(totime)
            if verb > 1: print "Median time (median aperture flux) = %.4f seconds" % np.median(data_cutout_aper)
            if verb > 1: print "Mean time (mean aperture flux) = %.4f seconds" % np.mean(data_cutout_aper)

            if verb > 1:
                fig = plt.figure()
                p = fig.add_subplot(111)
                p.imshow(totime)
                plt.show()

  
            # exposure time for aperture 
            data_cutout = mask.cutout(tmtImage)
            #data_cutout_aper = mask.apply(tmtImage)
            if photutils.__version__ == "0.4":
                data_cutout_aper = mask.multiply(tmtImage) # in version 0.4 of photutils
            else:
                data_cutout_aper = mask.apply(tmtImage)
            aper_sum = data_cutout_aper.sum()
            totime =  (snr * np.sqrt(aper_sum+noisetotal)/aper_sum)**2
            if verb > 1: print 'Time (aperture = %.4f") = %.4f' % (2*radius*scale, totime[0])
            
    jsondict={"mag":mag,"peakSNR":peakSNR,"medianSNR":medianSNR,"meanSNR":meanSNR,"minexptime":minexptime,"medianexptime":medianexptime,"meanexptime":meanexptime,"fluxdensity":str("%0.4e" %flambda[0])}
    print(json.dumps(jsondict))  
        
        #tmtImage_aper = aperture_photometry(tmtImage, aperture)
        #tmtImage_sum = tmtImage_aper["aperture_sum"]
        #tmtImage_err = tmtImage_aper["aperture_err"]
        #print tmtImage_aper
        
        #totime =  (snr * np.sqrt(tmtImage_sum+noisetotal)/tmtImage_sum)**2
        #print totime
        
        #bkg_totalObserved_func = Background2D(totalObserved,totalObserved.shape)     # constant
        #bkg_totalObserved = bkg_totalObserved_func.background
        
        #totalObserved_aper = aperture_photometry(totalObserved-bkg_totalObserved, aperture)
        #totalObserved_sum = totalObserved_aper["aperture_sum"]
        #totalObserved_err = totalObserved_aper["aperture_err"]
        #print totalObserved_aper
        
        
        #bkg_simImage_func = Background2D(simImage,simImage.shape)     # constant
        #bkg_simImage = bkg_simImage_func.background
        
        #simImage_aper = aperture_photometry(simImage-bkg_simImage, aperture)
        #simImage_sum = simImage_aper["aperture_sum"]
        #simImage_err = simImage_aper["aperture_err"]
        #print simImage_sum


# ~/python.linux/dev/iris/snr/iris_snr_sim.py
    
# Usage:
#  Imager mode
#    iris_snr_sim.py -mag 20.0 -filter K -scale 0.004 -mode imager -calc snr -nframes 2
#    iris_snr_sim.py -mag 20.0 -filter K -scale 0.004 -mode imager -calc exptime -snr 10
#  IFS mode
#    Case 1 (Vega or Flat spectrum)
#      iris_snr_sim.py -mag 20.0 -filter K -scale 0.004 -mode IFS -calc exptime -snr 50.0
#      iris_snr_sim.py -mag 20.0 -filter K -scale 0.004 -mode IFS -calc snr -nframes 1 -spectrum Vega
#      iris_snr_sim.py -mag 20.0 -filter K -scale 0.004 -mode IFS -calc exptime -snr 10 -spectrum Vega
#    Case 2 (Emission line spectrum)
#      iris_snr_sim.py -mag 20.0 -filter K -scale 0.004 -mode IFS -calc snr -nframes 1 -spectrum Emission -line-width 100 -wavelength 2.15
#      iris_snr_sim.py -mag 20.0 -filter K -scale 0.004 -mode IFS -calc exptime -snr 10 -spectrum Emission -line-width 100 -wavelength 2.15
#
#  PSFs
#    Imager mode
#      iris_snr_sim.py -mag 20.0 -filter K -scale 0.004 -mode imager -calc snr -nframes 2 -zenith-angle 45 -atm-cond 75 -psf-loc 0.6 12.
#    IFS mode
#      iris_snr_sim.py -mag 20.0 -filter K -scale 0.004 -mode IFS -calc exptime -snr 50.0 -zenith-angle 30 -atm-cond 25 -psf-loc 0. 0.
#
#
#  Vega test cases
#    iris_snr_sim.py -mag 0.0 -filter K -scale 0.004 -mode IFS -calc snr -nframes 1 -spectrum Vega
#    iris_snr_sim.py -mag 0.0 -filter K -scale 0.004 -mode IFS -calc exptime -snr 10 -spectrum Vega



parser = argparse.ArgumentParser(description='TMT IRIS S/N exposure calculator')


parser.add_argument('-filter', metavar='value', type=str, nargs='?',
                    default="K", help='filter name')
parser.add_argument('-scale', metavar='value', type=float, nargs='?',
                    default=0.004, help='detector scale [arcsec]')
parser.add_argument('-itime', metavar='value', type=float, nargs='?',
                    default=1.0, help='integration time [seconds]')
parser.add_argument('-resolution', metavar='value', type=int, nargs='?',
                    default=4000, help='resolution of the instrument')
parser.add_argument('-spectrum',  choices=['Vega','Flat','Emission'],
                    default="Vega", help='input spectrum')
parser.add_argument('-wavelength', metavar='value', type=float, nargs='?',
                    default="2.22", help='emission line wavelength [microns]')
parser.add_argument('-line-width', metavar='value', type=float, nargs='?',
                    default="200.", help='emission line width in velocity [km/s]')
parser.add_argument('-nframes', metavar='value', type=int, nargs='?',
                    default=1, help='number of frames')
parser.add_argument('-snr', metavar='value', type=float, nargs='?',
                    default=10.0, help='signal-to-noise ratio')
parser.add_argument('-calc', choices=['snr','exptime'], required=True,
                    help='calculation performed')
parser.add_argument('-mode', choices=['imager','IFS'], required=True,
                    help='instrumental mode')


parser.add_argument('-zenith-angle', type=float, metavar='value', nargs='?',
                    default=30., help='zenith angle of simulated PSF [degrees]')
parser.add_argument('-atm-cond', type=float, metavar='value', nargs='?',
                    default=50., help='atmosphere conditions of simulated PSF')
parser.add_argument('-psf-loc', nargs=2, type=float, metavar='value',
                    default=[8.8, 8.8], help='location of PSF on the focal plane of the instrument [arcsec]')

parser.add_argument('-psf-old', action='store_true',
                     help='use old PSFs')

parser.add_argument('-o', nargs='?', metavar='value', default=None,
                    help='Output file name, else display to screen')


group1 = parser.add_mutually_exclusive_group(required=True)
group1.add_argument('-mag', metavar='value', type=float, nargs='?',
                    default=21.0, help='magnitude of source [Vega]')
group1.add_argument('-flambda', metavar='value', type=float, nargs='?',
                    default=1.62e-19, help='flux density of source [erg/s/cm^2/Ang]')


if not os.path.exists('config.ini'):
    print "Missing config.ini file!"
    sys.exit()

try:
    #config = configparser.ConfigParser()
    config = ConfigParser.ConfigParser()
    config.read('config.ini')
    simdir = config.get('CONFIG','simdir')
    #simdir = config['CONFIG']['simdir']
    psfdir = config.get('CONFIG','psfdir')
    #psfdir = config['CONFIG']['psfdir']
except:
    print "Problem with config.ini file!"
    print "Missing parameter?"
    sys.exit()




args = parser.parse_args()

mag = args.mag
flambda = args.flambda

filter = args.filter
scale = args.scale
itime = args.itime
resolution = args.resolution
spectrum = args.spectrum
wavelength = args.wavelength
line_width = args.line_width


zenith_angle = args.zenith_angle  
atm_cond = args.atm_cond
psf_loc = args.psf_loc

psf_old = args.psf_old

nframes = args.nframes
snr = args.snr

mode = args.mode
calc = args.calc

png_output = args.o

#print mag
#print flambda
#sys.exit()

###############################################################
# verb = 0    No output
# verb = 1    Normal verbosity (including basic plotting)
# verb = 2    Diagnostic verbosity (all plots)
# verb = 3    Additional diagnostics (writes all fits files)
###############################################################

IRIS_ETC(mode=mode,calc=calc, nframes=nframes, snr=snr, itime=itime, mag=mag,
         resolution=resolution, filter=filter, scale=scale, simdir=simdir,
         spectrum=spectrum, lam_obs = wavelength, line_width = line_width, 
         zenith_angle=zenith_angle, atm_cond=atm_cond, psf_loc=psf_loc,
         png_output=png_output, psfdir=psfdir, psf_old=psf_old, verb=1)






    
# Test 1    
#IRIS_ETC(mode="imager",calc="snr")
# Test 2    
#IRIS_ETC(mode="imager",calc="itime")
#IRIS_ETC(mode="ifs",calc="snr", verb=2, mag=10)
#IRIS_ETC(mode="ifs",calc="snr", verb=2, mag=10, spectrum="Flat")
#IRIS_ETC(mode="ifs",calc="itime")
    
    
    
    
