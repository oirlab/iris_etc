#!/usr/bin/env python


#NAME:
#	BACKGROUND_SPECS3.PRO
#
#PURPOSE:
#	Produce background spectra (OH, continuum sky, and thermal) 
#	given a resolving power and temperature 
#	Typically ran outside of simulator for testing
#	
#INPUT:
#	RES: resolving power
#	FILTER: Filter to scale by
#
#KEYWORDS:
#	T_tel: Temperature of telescope (K)
#       T_atm: Temperature of atmosphere (K)
#	T_aos: Temperature of adaptive optic system (K)
#	T_zod: Temperature of zodical emission (K)#
#	Em_tel: Emissivity of telescope 
#       Em_atm: Emissivity of atmosphere 
#	Em_aos: Emissivity of adaptive optic system 
#	Em_zod: Emissivity of zodical emission#
#
#       convolve - convolve the OH spectrum to the correct spectral
#                  resolution (default is set)
#       noconvolve - do not convolve the OH spectrum
#       collarea - collecting area, by default set to TMT
#       nobbnorm - don't normalize the BB to zero mag (use absolute flux)
#       ohsim    - use simulated OH lines
#
#OUTPUT: Individual spectrum of each background component (tel,atm,ao)
#	and one final spectrum with all background combined with OH emission
#	(fits images) NOTE: will be in units of photons/s/m^2/arcsec/micron
#
#REVISION HISTORY:
# December 2008	 - Shelley Wright: Created.
# Feb 2010 - Modified to use better units and conversions for a given filter
# June 2010 - Added in Zodical emission BB
# 2011-02-08 - T. Do - slight modification to use the integrated flux
#              instead of the total flux, so start out with the zero
#              point in photons/s/m^2
# 2011-10-03 - T. Do - fullbb, fullcont, and fulloh spectra will now
# be returned scaled by the zeropoint of the filter used. This will be
# useful to get the background magnitude scaled by one of the filters
# 2012-04-11 - T. Do - added 'fullback' keyword to return the full
# back ground. Fixed bug where full background is returned instead of
# just within the filter range
# 2012-11-06 - Changed the procedure to use the sky background from
#              Gemini and with blackbody components for the telescope
#              and AO system. Also, by default with convolve the OH
#              lines unless specifically set not to with
#              'noconvolve'. Old 'convolve' keyword is left in place
#              for convenience. 
# 2012-11-10 - T. Do - Changed the default TMT telescope and AO system
#              temperature and emissivity to the values in
#              David Anderson's TMT report "SCIENTIFIC IMPACT
#              OF CHANGING THE NFIRAOS OPERATING TEMPERATURE"
# 2015-07-17 - D. Marshall - Fixed bug with 'geminifile' path slashes
#
#-

import os

from math import log, log10, ceil, floor, exp, sqrt

import numpy as np
from scipy import integrate,interpolate
import matplotlib.pyplot as plt

from astropy.modeling import models
from astropy.io import fits

from get_filterdat import get_filterdat

def extrap1d(interpolator):
    xs = interpolator.x
    ys = interpolator.y

    def pointwise(x):
        if x < xs[0]:
            #return ys[0]+(x-xs[0])*(ys[1]-ys[0])/(xs[1]-xs[0])
            return np.median(ys[0:100])
        elif x > xs[-1]:
            #return ys[-1]+(x-xs[-1])*(ys[-1]-ys[-2])/(xs[-1]-xs[-2])
            return np.median(ys[-100:])
        else:
            return interpolator(x)

    def ufunclike(xs):
        return np.array(map(pointwise, np.array(xs)))

    return ufunclike

class background_specs3():

    def __init__(self, resolution, filter, T_tel=275, T_atm=258.0, T_aos=243.0,
                    T_zod=5800.0, Em_tel=0.09, Em_atm=0.2, Em_aos=0.01,
                    Em_zod=3e-14*49.0,
                   
                          #waves = waves,
                          fullbb = False, fullcont = False, 
                          fulloh = False, #fullwaves = fullwaves, 
                          fullback = False,
                          convolve = True,# nobbnorm = nobbnorm, 
                          noconvolve = False, ohsim = False,
                   
                    verb = 0, simdir='~/data/iris/sim/', filteronly=False):


    #def background_specs2(resolution, filter, T_tel=275, T_atm=258.0, T_aos=243.0,
    #                      T_zod=5800.0, Em_tel=0.09, Em_atm=0.2, Em_aos=0.01,
    #                      Em_zod=3e-14*49.0,
    #                     
    #                            #waves = waves,
    #                            fullbb = False, fullcont = False, 
    #                            fulloh = False, #fullwaves = fullwaves, 
    #                            fullback = False,
    #                            convolve = True,# nobbnorm = nobbnorm, 
    #                            noconvolve = False, ohsim = False,
    #                     
    #                          verb = 0, simdir='~/data/iris/sim/'):

    # ignore the atmospheric BB and Zodiacal light for now since
    #we're going to be using Gemini
    #
    #if n_elements(collarea) eq 0 then collarea=630.0	#m^2, area of TMT primary
    ##some parameters
    #h=6.626e-34 #J*s
    #c=3.0e8     #m/s^2
    #k=1.38e-23  #J/K
    #hc=1.986e-16# Planck constant times speed of light (erg cm)
    
        h = 6.626e-27 #erg*s
        c = 3.0e10    #cm s^-2
        k = 1.38e-16  #boltman's constant (erg/K)

        ## Need to define zeropoint ###
        ### READ IN DATA FOR GIVEN FILTER FOR PSF (BROADBAND) 
        filterdat = get_filterdat(filter,simdir)
        zp = filterdat["zp"]         # units of phot/s/m2
        lambdamin = filterdat["lambdamin"]
        lambdamax = filterdat["lambdamax"]

        dxspectrum = 0
        if filteronly:
            wi = lambdamin
            wf = lambdamax
        else:
            wi = 8000.  #Angstroms
            wf = 25000. #Angstroms

        sterrad = 2.35e-11 # sterradians per square arcsecond
        
        ## CREATE THE SIZE OF THE SPECTRUM
        #determine length in pixels of complete spectrum(9000 to 25000), dxspectrum
        dxspectrum = int(ceil(log10(wf/wi)/log10(1.0+1.0/resolution) ) )
        print dxspectrum
        wave = np.linspace(wi,wf,dxspectrum)
        
        ## READ IN OH Lines ( the resolving power of this OH sky lines is around R=2600)
        #openr,ohlines_file,/get_lun,simdir+"/info/ohlineslist.dat"
        #woh=0.0 #initialize woh for search in ohlineslist.dat
        #	    #woh is last wavelength read in .dat
        
        # Generate arrays for loop below
        bbtel = np.zeros(dxspectrum)	#telescope blackbody spectrum
        bbaos = np.zeros(dxspectrum)	#AO blackbody spectrum
        bbatm = np.zeros(dxspectrum)	#ATM blackbody spectrum
        bbspec = np.zeros(dxspectrum)	#TOTAL blackbody spectrum
        bbzod = np.zeros(dxspectrum)
        ohspec = np.zeros(dxspectrum)	#OH lines
        contspec = np.zeros(dxspectrum)	#continuum of sky 
        wavelength = np.zeros(dxspectrum)
        
        ## Loop over the first and last pixel of the complete spectrum
        ## i (pixel along complete spectrum)
        #for i=0L,dxspectrum-1 do begin 
        print dxspectrum
        for i in xrange(dxspectrum):
        
            if filteronly:
                 wa=wave[i]	#wavelength in angstroms corresponding to pixel x
            else:        
                 wa=wi*(1.0+1.0/resolution)**i	#wavelength in angstroms corresponding to pixel x
            wamin=wa*(1.0-1.0/(2.0*resolution))	#min wavelength falling in this pixel
            wamax=wa*(1.0+1.0/(2.0*resolution))	#max pixel falling in this pixel
            wavelength[i] = wa
            
            ## Generate thermal Blackbodies (Tel, AO system, atmosphere)
            # erg s^-1 cm^-2 cm^-1 sr^-1
            bbtel[i] = (2*h*c*c/(wa*1e-8)**5) / (exp(h*c/(wa*1e-8*k*T_tel))-1.0)
            bbaos[i] = (2*h*c*c/(wa*1e-8)**5) / (exp(h*c/(wa*1e-8*k*T_aos))-1.0) 
            bbatm[i] = (2*h*c*c/(wa*1e-8)**5) / (exp(h*c/(wa*1e-8*k*T_atm))-1.0)
            bbzod[i] = (2*h*c*c/(wa*1e-8)**5) / (exp(h*c/(wa*1e-8*k*T_zod))-1.0) 
            # convert to photons s^-1 cm^-2 A^-1 sr-1 (conversion from
            # table) 	
            # 5.03e7*lambda photons/erg (where labmda is in angstrom)
            bbtel[i] = (bbtel[i] * 5.03e7 * wa) / 1e8  
            bbaos[i] = (bbaos[i] * 5.03e7 * wa) / 1e8  
            bbatm[i] = (bbatm[i] * 5.03e7 * wa) / 1e8 
            bbzod[i] = (bbzod[i] * 5.03e7 * wa) / 1e8  
            # now convert to photon s^-1 m^-2 um^-1 sr-2 (switching to meters and microns
            # for vega conversion down below, yes it cancels above but better to see the step)
            bbtel[i] = bbtel[i] * 1e4 * 1e4  
            bbaos[i] = bbaos[i] * 1e4 * 1e4    
            bbatm[i] = bbatm[i] * 1e4 * 1e4  
            bbzod[i] = bbzod[i] * 1e4 * 1e4  
            
            ## Total BB together with emissivities from each component
            ## photons s^-1 m^-2 um^-1 arcsecond^-2
            #bbspec[i] = sterrad*(bbatm[i]*Em_atm + bbtel[i]*Em_tel + bbaos[i]*Em_aos) 
        
            # only use the BB for the AO system and the telescope since
            # the Gemini observations already includes the atmosphere
            bbspec[i] = sterrad*(bbtel[i]*Em_tel + bbaos[i]*Em_aos) 
            #bbspec[i] = bbzod[i]*Em_Zod*sterrad
        
        
        if ohsim:
           # use the OH line simulator instead of loading the Gemini file
           ohspec = sim_ohlines(wavelength/1e4, simdir = simdir)
        
           delt = 2.0 # convolve with a Gaussian of 2 pix fwhm
        
        
           #psf = psf_gaussian(fwhm = delt, npixel = 4*fix(delt)+1, ndimen = 1, /normal)
        
           stddev = delt/2*sqrt(2*log(2))
           psf_func = models.Gaussian1D(amplitude=1.0, stddev=stddev)
           x = np.arange(4*int(delt)+1)-2*int(delt)
           psf = psf_func(x)
           psf /= psf.sum() # normaliza
        
           #ohspec = convol(ohspec, psf, /edge_truncate)
           ohspec = np.convolve(ohspec, psf)
        
        else:
           # read in Gemini data
           geminifile = os.path.expanduser(simdir+'skyspectra/mk_skybg_zm_16_15_ph.fits')
        
           #geminiSpec = readspec(geminifile, wave = backwave)*1e3
        
           ext = 0 
           pf = fits.open(geminifile)
           # load gemini file and convert from  ph/sec/arcsec^2/nm/m^2 to  ph/sec/arcsec^2/micron/m^2
           geminiSpec = pf[ext].data*1e3
           head = pf[ext].header
           cdelt1 = head["cdelt1"]
           crval1 = head["crval1"]
        
           nelem = geminiSpec.shape[0]
           #print nelem
        
           backwave = (np.arange(nelem))*cdelt1 + crval1  # compute wavelength
           backwave /= 1e4 # nm
           geminiWave = backwave*10.0 # nm -> Angstroms
        
        
        
           delt = 2.0*(wavelength[1]-wavelength[0])/(geminiWave[1]-geminiWave[0])
           if delt > 1:
              #psf = psf_gaussian(fwhm = delt, npixel = 4*fix(delt)+1, ndimen = 1, /normal)
        
              stddev = delt/2*sqrt(2*log(2))
              psf_func = models.Gaussian1D(amplitude=1.0, stddev=stddev)
              x = np.arange(4*int(delt)+1)-2*int(delt)
              psf = psf_func(x)
              psf /= psf.sum() # normaliza
        
              geminiSpec = np.convolve(geminiSpec, psf,mode='same')
        
           #print geminiWave
           #print wavelength
           # interpolate Gemini data to that of IRIS
           geminiOrig = geminiSpec
           #geminiSpec = interpol(geminiSpec, geminiWave, wavelength)
           #print geminiWave.shape
           #print geminiSpec.shape
           print geminiWave[0:10]
           print geminiWave[-40:]
           #print geminiSpec[0:40]
           #print geminiSpec[-40:]
           R_i = interpolate.interp1d(geminiWave,geminiSpec)
           R_x = extrap1d(R_i)
           geminiSpec = R_x(wavelength)
        
        
        
           # make sure that we have valid values beyond the edge of the Gemini
           # spectrum
           #bad = where((wavelength <= geminiWave[0]) or (geminiSpec <= 0.0), nbad)
        
           #if nbad > 0:
           #   geminiSpec[bad] = np.median(geminiSpec[max(bad)+1:max(bad)+100]) # replace the uninterpolated points to the median of the next 100 points to get some sort of continuum
        
           # place gemini spectrum as the OH spectrum
           ohspec = geminiSpec
        
        ### Get information on the filter selection used 
        ### Using zeropoints only from broadband filters (for right now)
        
        

        
        ## normalize the spectra to mag/sq arcsec and get total flux for range of desired filter 
        imin = int(ceil( log10(lambdamin/wi)/log10(1.0+1.0/(resolution)) ))
        #pixel of spectrum corresponding to min
        imax = int(floor( log10(lambdamax/wi)/log10(1.0+1.0/(resolution)) ))
        #same for max
        print lambdamin,lambdamax
        print 'min wave',imin
        print 'max wave',imax
        
        ## Define background spectra to region of desired filter
        ohspec_filt = ohspec[imin:imax+1]
        contspec_filt = contspec[imin:imax+1]
        bbspec_filt = bbspec[imin:imax+1]
        self.waves = wavelength[imin:imax+1] # (feeding plottiing of background)
        
        
        # convolve the OH lines to this resolution
        if not noconvolve:
           delt = 2.0*self.waves[1]/(resolution*(self.waves[1]-self.waves[0]))
           #print, 'delt: ', delt
           #psf = psf_gaussian(fwhm = delt, npixel = 4*fix(delt)+1, ndimen = 1, /normal)
           stddev = delt/2*sqrt(2*log(2))
           psf_func = models.Gaussian1D(amplitude=1.0, stddev=stddev)
           x = np.arange(4*int(delt)+1)-2*int(delt)
           psf = psf_func(x)
           psf /= psf.sum() # normaliza
           
           #ohspec_filt = convol(ohspec_filt, psf, /edge_truncate)
           ohspec_filt = np.convolve(ohspec_filt, psf, mode='same')
        
        # tot_oh = total(ohspec_filt)   #total integrated relative photon flux for OH spectrum
        # tot_cont = total(contspec_filt) #same for continuum spectrum
        # tot_bb = total(bbspec_filt)
        
        # tot_oh = int_tabulated(waves/1e4, ohspec_filt)
        # tot_cont = int_tabulated(waves/1e4, contspec_filt)
        # tot_bb = int_tabulated(waves/1e4, bbspec_filt)
        
        # ## *total #photons/s/um/sq arcsec collected by TMT for 0 mag/sq arcsec is
        # ## normalize to mag/sq.arcsec=0
        # # ohspectrum = (ohspec_filt/tot_oh) * zp * collarea
        # # contspectrum = (contspec_filt/tot_cont) *  zp * collarea
        # # bbspectrum = (bbspec_filt/tot_bb) *  zp * collarea
        
        
        # # scale by zeropoint to get #photons/s/m^2/um/arcec^2 for 0 mag/arcsec^2
        # ohspectrum = (ohspec_filt/tot_oh) * zp
        # contspectrum = (contspec_filt/tot_cont) *  zp
        # if keyword_set(nobbnorm) then bbspectrum = bbspec_filt else bbspectrum = (bbspec_filt/tot_bb) *  zp
        
        
        # return the background in the specific filter
        self.backspecs = np.zeros((3, ohspec_filt.shape[0]))
        self.backspecs[0, :] = ohspec_filt
        self.backspecs[1, :] = contspec_filt
        self.backspecs[2, :] = bbspec_filt
        
        ## return the entire background array, scaled by the zero points for
        ## the givien filter
        #fullbb = (bbspec/tot_bb)*zp
        
        
        
        # don't normalize
        if fullback or fulloh or fullcont or fullbb:
           fullbb = bbspec
           print fullbb.shape
        
           fullcont = contspec
           fulloh = ohspec
        
           self.fullwaves = wavelength
        
           self.fullback = np.zeros((3, fulloh.shape[0]))
           self.fullback[0, :] = fulloh
           self.fullback[1, :] = fullcont
           self.fullback[2, :] = fullbb
        
        ## write results
        #writefits,simdir+'info/tmt_oh_spectrum_m0_'+filter+'.fits',ohspectrum
        #writefits,simdir+'info/tmt_cont_spectrum_m0_'+filter+'.fits',contspectrum
        #writefits,simdir+'info/tmt_bb_spectrum_m0_'+filter+'.fits',bbspectrum
        
        ## output for iris_sim.pro

class background_specs2():

    def __init__(self, resolution, filter, T_tel=275, T_atm=258.0, T_aos=243.0,
                    T_zod=5800.0, Em_tel=0.09, Em_atm=0.2, Em_aos=0.01,
                    Em_zod=3e-14*49.0,
                   
                          #waves = waves,
                          fullbb = False, fullcont = False, 
                          fulloh = False, #fullwaves = fullwaves, 
                          fullback = False,
                          convolve = True,# nobbnorm = nobbnorm, 
                          noconvolve = False, ohsim = False,
                   
                    verb = 0, simdir='~/data/iris/sim/'):


    #def background_specs2(resolution, filter, T_tel=275, T_atm=258.0, T_aos=243.0,
    #                      T_zod=5800.0, Em_tel=0.09, Em_atm=0.2, Em_aos=0.01,
    #                      Em_zod=3e-14*49.0,
    #                     
    #                            #waves = waves,
    #                            fullbb = False, fullcont = False, 
    #                            fulloh = False, #fullwaves = fullwaves, 
    #                            fullback = False,
    #                            convolve = True,# nobbnorm = nobbnorm, 
    #                            noconvolve = False, ohsim = False,
    #                     
    #                          verb = 0, simdir='~/data/iris/sim/'):

    # ignore the atmospheric BB and Zodiacal light for now since
    #we're going to be using Gemini
    #
    #if n_elements(collarea) eq 0 then collarea=630.0	#m^2, area of TMT primary
    ##some parameters
    #h=6.626e-34 #J*s
    #c=3.0e8     #m/s^2
    #k=1.38e-23  #J/K
    #hc=1.986e-16# Planck constant times speed of light (erg cm)
    
        h = 6.626e-27 #erg*s
        c = 3.0e10    #cm s^-2
        k = 1.38e-16  #boltman's constant (erg/K)
        
        wf = 25000. #Angstroms
        wi = 8000.  #Angstroms
        sterrad = 2.35e-11 # sterradians per square arcsecond
        
        ## CREATE THE SIZE OF THE SPECTRUM
        #determine length in pixels of complete spectrum(9000 to 25000), dxspectrum
        dxspectrum = int(ceil(log10(wf/wi)/log10(1.0+1.0/resolution) ) )
        print dxspectrum
        
        ## READ IN OH Lines ( the resolving power of this OH sky lines is around R=2600)
        #openr,ohlines_file,/get_lun,simdir+"/info/ohlineslist.dat"
        #woh=0.0 #initialize woh for search in ohlineslist.dat
        #	    #woh is last wavelength read in .dat
        
        # Generate arrays for loop below
        bbtel = np.zeros(dxspectrum)	#telescope blackbody spectrum
        bbaos = np.zeros(dxspectrum)	#AO blackbody spectrum
        bbatm = np.zeros(dxspectrum)	#ATM blackbody spectrum
        bbspec = np.zeros(dxspectrum)	#TOTAL blackbody spectrum
        bbzod = np.zeros(dxspectrum)
        ohspec = np.zeros(dxspectrum)	#OH lines
        contspec = np.zeros(dxspectrum)	#continuum of sky 
        wavelength = np.zeros(dxspectrum)
        
        ## Loop over the first and last pixel of the complete spectrum
        ## i (pixel along complete spectrum)
        #for i=0L,dxspectrum-1 do begin 
        print dxspectrum-1
        for i in xrange(dxspectrum-1):
        
        
            wa=wi*(1.0+1.0/resolution)**i	#wavelength in angstroms corresponding to pixel x
            wamin=wa*(1.0-1.0/(2.0*resolution))	#min wavelength falling in this pixel
            wamax=wa*(1.0+1.0/(2.0*resolution))	#max pixel falling in this pixel
            wavelength[i] = wa
            
            ## Generate thermal Blackbodies (Tel, AO system, atmosphere)
            # erg s^-1 cm^-2 cm^-1 sr^-1
            bbtel[i] = (2*h*c*c/(wa*1e-8)**5) / (exp(h*c/(wa*1e-8*k*T_tel))-1.0)
            bbaos[i] = (2*h*c*c/(wa*1e-8)**5) / (exp(h*c/(wa*1e-8*k*T_aos))-1.0) 
            bbatm[i] = (2*h*c*c/(wa*1e-8)**5) / (exp(h*c/(wa*1e-8*k*T_atm))-1.0)
            bbzod[i] = (2*h*c*c/(wa*1e-8)**5) / (exp(h*c/(wa*1e-8*k*T_zod))-1.0) 
            # convert to photons s^-1 cm^-2 A^-1 sr-1 (conversion from
            # table) 	
            # 5.03e7*lambda photons/erg (where labmda is in angstrom)
            bbtel[i] = (bbtel[i] * 5.03e7 * wa) / 1e8  
            bbaos[i] = (bbaos[i] * 5.03e7 * wa) / 1e8  
            bbatm[i] = (bbatm[i] * 5.03e7 * wa) / 1e8 
            bbzod[i] = (bbzod[i] * 5.03e7 * wa) / 1e8  
            # now convert to photon s^-1 m^-2 um^-1 sr-2 (switching to meters and microns
            # for vega conversion down below, yes it cancels above but better to see the step)
            bbtel[i] = bbtel[i] * 1e4 * 1e4  
            bbaos[i] = bbaos[i] * 1e4 * 1e4    
            bbatm[i] = bbatm[i] * 1e4 * 1e4  
            bbzod[i] = bbzod[i] * 1e4 * 1e4  
            
            ## Total BB together with emissivities from each component
            ## photons s^-1 m^-2 um^-1 arcsecond^-2
            #bbspec[i] = sterrad*(bbatm[i]*Em_atm + bbtel[i]*Em_tel + bbaos[i]*Em_aos) 
        
            # only use the BB for the AO system and the telescope since
            # the Gemini observations already includes the atmosphere
            bbspec[i] = sterrad*(bbtel[i]*Em_tel + bbaos[i]*Em_aos) 
            #bbspec[i] = bbzod[i]*Em_Zod*sterrad
        
        
        if ohsim:
           # use the OH line simulator instead of loading the Gemini file
           ohspec = sim_ohlines(wavelength/1e4, simdir = simdir)
        
           delt = 2.0 # convolve with a Gaussian of 2 pix fwhm
        
        
           #psf = psf_gaussian(fwhm = delt, npixel = 4*fix(delt)+1, ndimen = 1, /normal)
        
           stddev = delt/2*sqrt(2*log(2))
           psf_func = models.Gaussian1D(amplitude=1.0, stddev=stddev)
           x = np.arange(4*int(delt)+1)-2*int(delt)
           psf = psf_func(x)
           psf /= psf.sum() # normaliza
        
           #ohspec = convol(ohspec, psf, /edge_truncate)
           ohspec = np.convolve(ohspec, psf)
        
        else:
           # read in Gemini data
           geminifile = os.path.expanduser(simdir+'skyspectra/mk_skybg_zm_16_15_ph.fits')
        
           #geminiSpec = readspec(geminifile, wave = backwave)*1e3
        
           ext = 0 
           pf = fits.open(geminifile)
           # load gemini file and convert from  ph/sec/arcsec^2/nm/m^2 to  ph/sec/arcsec^2/micron/m^2
           geminiSpec = pf[ext].data*1e3
           head = pf[ext].header
           cdelt1 = head["cdelt1"]
           crval1 = head["crval1"]
        
           nelem = geminiSpec.shape[0]
           #print nelem
        
           backwave = (np.arange(nelem))*cdelt1 + crval1  # compute wavelength
           backwave /= 1e4 # nm
           geminiWave = backwave*10.0 # nm -> Angstroms
        
        
        
           delt = 2.0*(wavelength[1]-wavelength[0])/(geminiWave[1]-geminiWave[0])
           if delt > 1:
              #psf = psf_gaussian(fwhm = delt, npixel = 4*fix(delt)+1, ndimen = 1, /normal)
        
              stddev = delt/2*sqrt(2*log(2))
              psf_func = models.Gaussian1D(amplitude=1.0, stddev=stddev)
              x = np.arange(4*int(delt)+1)-2*int(delt)
              psf = psf_func(x)
              psf /= psf.sum() # normaliza
        
              geminiSpec = np.convolve(geminiSpec, psf,mode='same')
        
           #print geminiWave
           #print wavelength
           # interpolate Gemini data to that of IRIS
           geminiOrig = geminiSpec
           #geminiSpec = interpol(geminiSpec, geminiWave, wavelength)
           #print geminiWave.shape
           #print geminiSpec.shape
           print geminiWave[0:10]
           print geminiWave[-40:]
           #print geminiSpec[0:40]
           #print geminiSpec[-40:]
           R_i = interpolate.interp1d(geminiWave,geminiSpec)
           R_x = extrap1d(R_i)
           geminiSpec = R_x(wavelength)
        
        
        
           # make sure that we have valid values beyond the edge of the Gemini
           # spectrum
           #bad = where((wavelength <= geminiWave[0]) or (geminiSpec <= 0.0), nbad)
        
           #if nbad > 0:
           #   geminiSpec[bad] = np.median(geminiSpec[max(bad)+1:max(bad)+100]) # replace the uninterpolated points to the median of the next 100 points to get some sort of continuum
        
           # place gemini spectrum as the OH spectrum
           ohspec = geminiSpec
        
        ### Get information on the filter selection used 
        ### Using zeropoints only from broadband filters (for right now)
        
        
        ## Need to define zeropoint ###
        ### READ IN DATA FOR GIVEN FILTER FOR PSF (BROADBAND) 
        filterdat = get_filterdat(filter,simdir)
        backmag = filterdat["backmag"]
        zp = filterdat["zp"]         # units of phot/s/m2
        zpphot = filterdat["zpphot"] # units to phot/s/m2/um for vega 
        psfname = filterdat["psfname"]
        psfsamp = filterdat["psfsamp"]
        psfsize = filterdat["psfsize"]
        lambdamin = filterdat["lambdamin"]
        lambdamax = filterdat["lambdamax"]
        lambdac = filterdat["lambdac"]
        bw = filterdat["bw"]
        
        ## normalize the spectra to mag/sq arcsec and get total flux for range of desired filter 
        imin = int(ceil( log10(lambdamin/wi)/log10(1.0+1.0/(resolution)) ))
        #pixel of spectrum corresponding to min
        imax = int(floor( log10(lambdamax/wi)/log10(1.0+1.0/(resolution)) ))
        #same for max
        print lambdamin,lambdamax
        print 'min wave',imin
        print 'max wave',imax
        
        ## Define background spectra to region of desired filter
        ohspec_filt = ohspec[imin:imax]
        contspec_filt = contspec[imin:imax]
        bbspec_filt = bbspec[imin:imax]
        self.waves = wavelength[imin:imax] # (feeding plottiing of background)
        
        
        # convolve the OH lines to this resolution
        if not noconvolve:
           delt = 2.0*self.waves[1]/(resolution*(self.waves[1]-self.waves[0]))
           #print, 'delt: ', delt
           #psf = psf_gaussian(fwhm = delt, npixel = 4*fix(delt)+1, ndimen = 1, /normal)
           stddev = delt/2*sqrt(2*log(2))
           psf_func = models.Gaussian1D(amplitude=1.0, stddev=stddev)
           x = np.arange(4*int(delt)+1)-2*int(delt)
           psf = psf_func(x)
           psf /= psf.sum() # normaliza
           
           #ohspec_filt = convol(ohspec_filt, psf, /edge_truncate)
           ohspec_filt = np.convolve(ohspec_filt, psf, mode='same')
        
        # tot_oh = total(ohspec_filt)   #total integrated relative photon flux for OH spectrum
        # tot_cont = total(contspec_filt) #same for continuum spectrum
        # tot_bb = total(bbspec_filt)
        
        # tot_oh = int_tabulated(waves/1e4, ohspec_filt)
        # tot_cont = int_tabulated(waves/1e4, contspec_filt)
        # tot_bb = int_tabulated(waves/1e4, bbspec_filt)
        
        # ## *total #photons/s/um/sq arcsec collected by TMT for 0 mag/sq arcsec is
        # ## normalize to mag/sq.arcsec=0
        # # ohspectrum = (ohspec_filt/tot_oh) * zp * collarea
        # # contspectrum = (contspec_filt/tot_cont) *  zp * collarea
        # # bbspectrum = (bbspec_filt/tot_bb) *  zp * collarea
        
        
        # # scale by zeropoint to get #photons/s/m^2/um/arcec^2 for 0 mag/arcsec^2
        # ohspectrum = (ohspec_filt/tot_oh) * zp
        # contspectrum = (contspec_filt/tot_cont) *  zp
        # if keyword_set(nobbnorm) then bbspectrum = bbspec_filt else bbspectrum = (bbspec_filt/tot_bb) *  zp
        
        
        # return the background in the specific filter
        self.backspecs = np.zeros((3, ohspec_filt.shape[0]))
        self.backspecs[0, :] = ohspec_filt
        self.backspecs[1, :] = contspec_filt
        self.backspecs[2, :] = bbspec_filt
        
        ## return the entire background array, scaled by the zero points for
        ## the givien filter
        #fullbb = (bbspec/tot_bb)*zp
        
        
        
        # don't normalize
        if fullback or fulloh or fullcont or fullbb:
           fullbb = bbspec
           print fullbb.shape
        
           fullcont = contspec
           fulloh = ohspec
        
           self.fullwaves = wavelength
        
           self.fullback = np.zeros((3, fulloh.shape[0]))
           self.fullback[0, :] = fulloh
           self.fullback[1, :] = fullcont
           self.fullback[2, :] = fullbb
        
        ## write results
        #writefits,simdir+'info/tmt_oh_spectrum_m0_'+filter+'.fits',ohspectrum
        #writefits,simdir+'info/tmt_cont_spectrum_m0_'+filter+'.fits',contspectrum
        #writefits,simdir+'info/tmt_bb_spectrum_m0_'+filter+'.fits',bbspectrum
        
        ## output for iris_sim.pro

def test_background_specs2(simdir='~/data/iris/sim/'):

    resolution = 4000
    filter = "Kbb"
    bkgd1 = background_specs2(resolution*2.0, filter, Em_tel=0.02, fulloh=True)
    bkgd2 = background_specs2(resolution*2.0, filter, Em_tel=0.09, fulloh=True)

    fullwaves   = bkgd1.fullwaves/1e4
    fullback1   = bkgd1.fullback
    fullback2   = bkgd2.fullback
    ohspec1     = fullback1[0,:]
    ohspec2     = fullback2[0,:]

    backwave    = bkgd1.waves/1e4
    background1 = bkgd1.backspecs
    background2 = bkgd2.backspecs

    # read in Gemini data
    geminifile = os.path.expanduser(simdir+'skyspectra/mk_skybg_zm_16_15_ph.fits')
    ext = 0 
    pf = fits.open(geminifile)
    # load gemini file and convert from  ph/sec/arcsec^2/nm/m^2 to  ph/sec/arcsec^2/micron/m^2
    geminiSpec = pf[ext].data*1e3
    head = pf[ext].header
    cdelt1 = head["cdelt1"]
    crval1 = head["crval1"]

    nelem = geminiSpec.shape[0]
    #print nelem
    geminiWave = (np.arange(nelem))*cdelt1 + crval1  # compute wavelength
    #print geminiWave.shape
    geminiWave /= 1e4 # nm
    geminiWave /= 1e3 # nm -> microns

    #geminiWave = backwave*10.0 # go from nm to angstroms

    fig = plt.figure()
    p = fig.add_subplot(211)

    p.plot(fullwaves,ohspec1,c="k",zorder=3,label="Full OH spectrum")
    p.plot(backwave,np.sum(background2,axis=0),c="r",zorder=2,label="Convolved")
    p.plot(geminiWave,geminiSpec,c="b",zorder=1,label="Gemini")
    #p.plot(backwave,np.sum(background1,axis=0)-np.sum(background2,axis=0),c="b")
    p.set_yscale("log")
    p.set_xlim(2.0,2.1)

    p.legend()

    p = fig.add_subplot(212)
    #p.plot(fullwaves,ohspec1,c="k",zorder=3,label="Full OH spectrum")
    p.plot(fullwaves,np.sum(fullback1,axis=0),c="k")
    
    p.set_yscale("log")
    p.set_xlim(0.8,0.9)

    plt.show()

#test_background_specs2()
    
    

