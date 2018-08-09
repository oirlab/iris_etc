
import os
import numpy as np

def get_filterdat(filter,simdir='~/iris/sensitivity/'):
    filterfile = os.path.expanduser(simdir + "info/filter_info.dat")

    filterall = np.genfromtxt(filterfile,dtype=None,
                   names = ["filterread", "lambdamin", "lambdamax", "lambdac",
                            "bw", "backmag", "imagmag", "zp", "zpphot",
                            "psfname", "psfsamp", "psfsize", "filterfiles"])
   
    index = np.where(filterall['filterread'] == filter)
    filterdat=filterall[index]
    
    return filterdat

