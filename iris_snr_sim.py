#!/usr/bin/env python

# SNR equation:
# S/N = S*sqrt(T)/sqrt(S + npix(B + D + R^2/t))
# t = itime per frame
# T = sqrt(Nframes*itime)
# S, B, D -> electrons per second
# R -> read noise (electrons)


import argparse, os
from math import log10,ceil,sqrt,log

import numpy as np
from scipy import integrate,interpolate

from astropy.io import fits
from astropy.modeling import models

from photutils import aperture_photometry
from photutils import CircularAperture, SkyCircularAperture
from photutils.background import Background2D

import matplotlib.pyplot as plt

from get_filterdat import get_filterdat
#from background_specs import background_specs2
from background_specs import background_specs3

def IRIS_ETC(filter = "K", mag = 21.0, itime = 1.0, nframes = 1, snr = 10.0,
             radius = 0.024, gain = 1.0, readnoise = 5., darkcurrent = 0.002,
             scale = 0.004, resolution = 4000, collarea = 630.0, 
             positions = [239, 239], bgmag = None, efftot = None,
             mode = "imager", calc = "snr", spectrum = "vega_all.fits",
             verb = 0, simdir='~/data/iris/sim/'):

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
    #           calc - either "snr" or "itime"
    
    radius /= scale
    
    xc,yc = positions
    
    ##### READ IN FILTER INFORMATION
    filterdat = get_filterdat(filter, simdir)
    lambdamin = filterdat["lambdamin"]
    lambdamax = filterdat["lambdamax"]
    lambdac = filterdat["lambdac"]
    bw = filterdat["bw"]
    filterfile = os.path.expanduser(simdir + filterdat["filterfiles"][0])
    print filterfile
    
    #print lambdamin
    #print lambdamax
    #print lambdac
    #print bw
    
    ## Determine the length of the cube, dependent on filter 
    ## (supporting only BB for now)
    dxspectrum = 0
    wi = lambdamin
    wf = lambdamax
    #resolution = resolution*2.  
    dxspectrum = int(ceil( log10(wf/wi)/log10(1.0+1.0/(resolution*2.0)) ))
    ## resolution times 2 to get nyquist sampled 
    crval1 = wi/10.                      #nm
    cdelt1 = ((wf-wi) / dxspectrum)/10.  #nm/channel


    # Throughput calculation    
    if efftot is None: 

        # Throughput number from Ryuji (is there wavelength dependence?)
        teltot = 0.91 # TMT total throughput
        aotot = 0.80  # NFIRAOS AO total throughput
        #teltot = 1.0
        #aotot = 1.0 

        wav  = [840,900,2000,2200,2300,2400] # nm
        
        if mode == "imager":
           tput = [0.631,0.772,0.772,0.813,0.763,0.728] # imager
           print 'IRIS imager selected!!!'
        
        else:
           if (scale == 0.004) or (scale == 0.009):
              tput = [0.340,0.420,0.420,0.490,0.440,0.400] # IFS lenslet
              print 'IRIS lenslet selected!!!'
           else:
              tput = [0.343,0.465,0.465,0.514,0.482,0.451] # IFS slicer
              print 'IRIS slicer selected!!!'
        
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
    
    print  ' '
    #print  'IRIS efficiency ', efftot
    print  'Total throughput (TMT+NFIRAOS+IRIS) = %.3f' % efftot
    
    
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
    
    psf_dict = { 928:"psf_x0_y0_wvl928nm_implKOLMO117nm_bin4mas_sm.fits",
                1092:"psf_x0_y0_wvl1092nm_implKOLMO117nm_bin4mas_sm.fits",
                1270:"psf_x0_y0_wvl1270nm_implKOLMO117nm_bin4mas_sm.fits",
                1629:"psf_x0_y0_wvl1629nm_implKOLMO117nm_bin4mas_sm.fits",
                2182:"psf_x0_y0_wvl2182nm_implKOLMO117nm_bin4mas_sm.fits"}
    
    psf_wvls = psf_dict.keys()
    
    psf_ind = np.argmin(np.abs(lambdac/10. - psf_wvls))
    psf_wvl =  psf_wvls[psf_ind]
    psf_file = os.path.expanduser(simdir + "/psfs/" + psf_dict[psf_wvl])
    #print psf_ind
    #print psf_wvl
    #print psf_file
   
    ext = 0 
    pf = fits.open(psf_file)

    image = pf[ext].data
    #print image.sum()
    flux = zp*10**(-0.4*mag) # photons/s/m^2



    if mode == "ifs":

        hwbox = 10
        xp = hwbox
        yp = hwbox
        subimage = image[yc-hwbox:yc+hwbox+1,xc-hwbox:xc+hwbox+1]

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
      
        if verb:
           print 'mean OH: ', np.mean(ohspectrum)
           print 'mean continuum: ', np.mean(contspectrum)
           print 'mean bb: ', np.mean(bbspectrum)
           print 'mean background: ', np.mean(backtot)

        backwave    = bkgd.waves/1e4

        wave = np.linspace(wi/1e4,wf/1e4,dxspectrum)
        print dxspectrum
        print len(wave)
        print len(backwave)
        print wave
        print backwave

        backtot_func = interpolate.interp1d(backwave,backtot)
        backtot = backtot_func(wave)

        print subimage.sum()
        print subimage.shape

        if spectrum == "Flat":
            spec_temp = np.ones(dxspectrum)
            #intFlux = integrate.trapz(spec_temp,wave)
            intFlux = dxspectrum
            print intFlux
            intNorm = flux/intFlux
        else:
            ext = 0 
            pf = fits.open(spectrum)
            spec = pf[ext].data
            head = pf[ext].header
            cdelt1 = head["cdelt1"]
            crval1 = head["crval1"]
            
            nelem = spec.shape[0]
            
            specwave = (np.arange(nelem))*cdelt1 + crval1  # compute wavelength [Angstroms]
            specwave /= 1e4 # -> microns

            intFlux = integrate.trapz(spec,specwave)
            intNorm = flux/intFlux
            print intFlux

            #fig = plt.figure()
            #p = fig.add_subplot(111)
            #p.plot(specwave, spec)
            ##p.set_xscale("log")
            ##p.set_yscale("log")
            #plt.show()

            ################################################
            # convolve with the resolution of the instrument
            ################################################
            delt = 2.0*(wave[1]-wave[0])/(specwave[1]-specwave[0])
            if delt > 1:
            
               stddev = delt/2*sqrt(2*log(2))
               psf_func = models.Gaussian1D(amplitude=1.0, stddev=stddev)
               x = np.arange(4*int(delt)+1)-2*int(delt)
               psf = psf_func(x)
               psf /= psf.sum() # normaliza
            
               spec = np.convolve(spec, psf,mode='same')

            spec_func = interpolate.interp1d(specwave,spec)
            spec_temp = spec_func(wave)

            #spec_norm = np.mean(spec_temp)
            #spec_temp /= spec_norm


        cube = intNorm*(subimage[np.newaxis]*spec_temp[:,np.newaxis,np.newaxis]).astype(np.float32)
        print cube.sum()
        print cube.shape

        # convert the signal and the background into photons/s observed
        # with TMT
        observedCube = cube*collarea*efftot    # photons/s/um
        backtot = backtot*collarea*efftot       # photons/s/um

        # get photons/s per spectral channel, since each spectral
        # channel has the same bandwidth
        observedCube = observedCube*(wave[1]-wave[0])
        backtot = backtot*(wave[1]-wave[0])

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

        fig = plt.figure()
        p = fig.add_subplot(111)
        #p.plot(wave, filter_tput*cube[:,yc,xc])
        p.plot(wave, cube[:,yp,xp])
        plt.show()

        if verb:
            print 'n wavelength channels: ', len(wave)
            print 'channel width (micron): ', wave[1]-wave[0]
            print 'mean flux input cube center (phot/s/m^2/micron): ', np.mean(cube[:, yp, xp])
            print 'mean counts/spectral channel input cube center (phot/s): ', np.mean(observedCube[:, yp, xp])
            print 'mean background (phot/s): ', np.mean(backtot)
        
        backgroundCube = np.broadcast_to(backtot[:,np.newaxis,np.newaxis],cube.shape)
        #print backgroundCube
        print backgroundCube.shape

        # make a background cube and add noise
        # noise = sqrt(S + B + R^2/t)
        noiseCube = np.sqrt(observedCube+backgroundCube+darkcurrent+readnoise**2.0/itime)

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
        
        if verb > 1:
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
        #;snrCube = observedCube*nframes*itime/rmsNoiseCube
        
        # SNR cube  = S*sqrt(itime*nframes)/sqrt(S + B+ R^2/t)
        snrCube = observedCube*sqrt(itime*nframes)/noiseCube
        #snrCube = float(snrCube)
        if verb > 1:
            hdu = fits.PrimaryHDU(snrCube)
            hdul = fits.HDUList([hdu])
            hdul.writeto('snrCube.fits',clobber=True)
        
        
        #;; save the SNR cube if given
        #if n_elements(outcube) ne 0 then begin
        #   mkosiriscube, wave, transpose(snrCube, [2, 1, 0]), outcube, /micron, scale = scale, units = 'SNR', params = fitsParams, values = fitsValues
        #endif
        #return, snrCube

    else:
        image *= flux/image.sum()
    
        ############################# NOISE ####################################
        # Calculate total background number of photons for whole tel aperture
        #efftot = effao*efftel*effiris #total efficiency 
        print 'background magnitude: ', backmag
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
        print 'detector noise (e-/s): ', noise
        print 'total background noise (phot/s):', background
        print '  '
        print 'Total Noise (photons per pixel^2)= ', noisetotal
        ###################################################################################################
        
        ## put in the TMT collecting area and efficiency
        tmtImage = image*collarea*efftot
        
        print
        print
        print
   
        # to define apertures used throughout the calculations
        radii = np.arange(1,50,1) # pixels
        apertures = [CircularAperture(positions, r=r) for r in radii]
        aperture = CircularAperture(positions, r=radius)
        masks = aperture.to_mask(method='center')
        mask = masks[0]

        ####################################################
        # Case 1: find s/n for a given exposure time and mag
        ####################################################
        if calc == "snr":

            print "Case 1: find S/N for a given exposure time and mag"
        
            signal = tmtImage*np.sqrt(itime*nframes)  # photons/s
            noisemap = np.sqrt(tmtImage+noisetotal)
            
            snrMap = signal/noisemap
            #print np.max(snrMap)
            #print np.mean(snrMap)
            
            if verb > 0:
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
                hdu = fits.PrimaryHDU(snrMap)
                hdul = fits.HDUList([hdu])
                hdul.writeto('snrImage.fits',clobber=True)
            
            # model + background
            totalObserved = tmtImage*itime*nframes + background*itime*nframes + darkcurrent*itime*nframes + readnoise*itime*nframes
            #print totalObserved.shape
            #print totalObserved.dtype
            
            if verb > 1:
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
            
            if verb > 1:
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
            data_cutout_aper = mask.apply(snrMap)
            #data_cutout_aper = mask.multiply(snrMap) # in version 0.4 of photutils
            #print np.min(snrMap)
            print "Peak S/N = %.4f" % np.max(snrMap)
            print "Median S/N = %.4f" % np.median(data_cutout_aper)
            print "Mean S/N = %.4f" % np.mean(data_cutout_aper)
            
            if verb > 0:
                fig = plt.figure()
                p = fig.add_subplot(111)
                p.imshow(data_cutout_aper,interpolation='none')
                plt.show()
            
            phot_table = aperture_photometry(signal, aperture, error=noisemap)
            #print phot_table
            snr_int = phot_table["aperture_sum"].quantity/phot_table["aperture_sum_err"].quantity
            print 'S/N (aperture = %.4f") = %.4f' % (2*radius*scale, snr_int[0])
            
            if verb > 0:
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
            
            print
            print
            print
    
        #######################################################
        # Case 2: find integration time for a given s/n and mag
        #######################################################
        elif calc == "itime":

            print "Case 2: find integration time for a given S/N and mag"
            
            # snr = tmtImage*np.sqrt(itime*nframes)/np.sqrt(tmtImage+noisetotal)
            # itime * nframes =  (snr * np.sqrt(tmtImage+noisetotal)/tmtImage)**2
            
            totime =  (snr * np.sqrt(tmtImage+noisetotal)/tmtImage)**2
            # totime = itime * nframes
            
            #print totime
            #print np.max(totime)
            data_cutout = mask.cutout(totime)
            data_cutout_aper = mask.apply(totime)
            print "Min time (peak flux) = %.4f seconds" % np.min(totime)
            print "Median time (median aperture flux) = %.4f seconds" % np.median(data_cutout_aper)
            print "Mean time (mean aperture flux) = %.4f seconds" % np.mean(data_cutout_aper)

  
            # exposure time for aperture 
            data_cutout = mask.cutout(tmtImage)
            data_cutout_aper = mask.apply(tmtImage)
            aper_sum = data_cutout_aper.sum()
            totime =  (snr * np.sqrt(aper_sum+noisetotal)/aper_sum)**2
            print 'Time (aperture = %.4f") = %.4f' % (2*radius*scale, totime[0])
            
            if verb > 0:
                fig = plt.figure()
                p = fig.add_subplot(111)
                p.imshow(totime)
                plt.show()
        
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


    
# usage:
#   iris_snr_sim.py -mag 20.0 -filter K -scale 0.004 -imager -snr 60.0 
#   iris_snr_sim.py -mag 20.0 -filter K -scale 0.004 -imager -frames 2

parser = argparse.ArgumentParser(description='TMT IRIS S/N exposure calculator')

parser.add_argument('-mag', metavar='value', type=float, nargs='?',
                    default=21.0, help='magnitude of source')
parser.add_argument('-filter', metavar='value', type=str, nargs='?',
                    default="K", help='filter name')
parser.add_argument('-scale', metavar='value', type=float, nargs='?',
                    default=0.004, help='detector scale')
parser.add_argument('-itime', metavar='value', type=float, nargs='?',
                    default=1.0, help='integration time')

group1 = parser.add_mutually_exclusive_group(required=True)
group1.add_argument('-IFS', action="store_true",
                    help='row to use')
group1.add_argument('-imager', action="store_true",
                    help='column to use')

group2 = parser.add_mutually_exclusive_group(required=True)
group2.add_argument('-nframes', metavar='value', type=int, nargs='?',
                    help='number of frames')
group2.add_argument('-snr', metavar='value', type=float, nargs='?',
                    help='signal-to-noise')



#simdir = "~/IRIS_snr_sim/"
simdir = "~/python.linux/packages/IRIS_snr_sim/"
#simdir = "~/data/iris/sim/"

args = parser.parse_args()

mag  = args.mag
filter = args.filter
scale = args.scale
itime = args.itime

nframes = args.nframes
snr = args.snr

ifs = args.IFS
imager = args.imager


if snr: calc="itime"
elif nframes: calc="snr"
print calc

if ifs: mode="ifs"
elif imager: mode="imager"
print mode

IRIS_ETC(mode=mode,calc=calc, nframes=nframes, snr=snr, itime=itime, mag=mag,
         filter=filter, scale=scale, simdir=simdir)
    
# Test 1    
#IRIS_ETC(mode="imager",calc="snr")
# Test 2    
#IRIS_ETC(mode="imager",calc="itime")
#IRIS_ETC(mode="ifs",calc="snr", verb=2, mag=10)
#IRIS_ETC(mode="ifs",calc="snr", verb=2, mag=10, spectrum="Flat")
#IRIS_ETC(mode="ifs",calc="itime")
    
    
    
    
