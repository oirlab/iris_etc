
import numpy as np

def get_psf(za,cond,mode,time,psf_loc,scale): 

    x,y = psf_loc

    za_arr = [ 0, 30, 45] # zenith angle [degrees]
    cond_arr = [25, 50, 75] # observing conditions
    time_arr = [1.4, 300] # integration time [seconds]

    if mode.lower() == "ifs":
        scale_arr = [2, 4, 9, 25, 50] # plate scale [mas]
        x_arr = y_arr = [0] # position on focal plane [arcsec]
        ins = "ifu"


    elif mode.lower() == "imager":
        scale_arr = [2] # plate scale [mas]
        x_arr = y_arr = [0.6, 4.7, 8.8, 12.9, 17] # position on focal plane [arcsec]
        ins = "im"


    # select the closest file, in case the user guesses wrong
    za_ind = np.argmin(np.abs(np.array(za_arr) - za))
    cond_ind = np.argmin(np.abs(np.array(cond_arr) - cond))
    time_ind = np.argmin(np.abs(np.array(time_arr) - time))
    scale_ind = np.argmin(np.abs(np.array(scale_arr) - scale*1e3))
    x_ind = np.argmin(np.abs(np.array(x_arr) - x))
    y_ind = np.argmin(np.abs(np.array(y_arr) - y))

    za_s = za_arr[za_ind]
    cond_s = cond_arr[cond_ind]
    time_s = time_arr[time_ind]
    scale_s = scale_arr[scale_ind]
    x_s = x_arr[x_ind]
    y_s = y_arr[y_ind]

    return "za%i_%ip_%s_%ss/evlpsfcl_1_x%s_y%s_%imas.fits" % (za_s,cond_s,ins,time_s,x_s,y_s,scale_s)

#print get_psf(30,75,"ifu",1.4,0,0,50)
#print get_psf(45,25,"im",300,0.6,0.6,2)

#print get_psf(10,45,"ifu",2,0.6,4.7,10)
