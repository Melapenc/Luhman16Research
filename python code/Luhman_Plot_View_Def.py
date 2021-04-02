from astropy.io import ascii
from astropy.table import Table, Column, MaskedColumn
import sys
import os
import math
# import matplotlib as plt
import os.path
import matplotlib.pyplot as plt
import operator
import numpy as np
from astropy.stats import sigma_clip
from scipy.optimize import curve_fit
from scipy.optimize import leastsq
import astropy.io.fits as fits
import matplotlib.lines as lines
sigma_num=5

def median_flux(ap_ph):
    ap_ph_med = ap_ph/np.nanmedian(ap_ph)
    return ap_ph_med
def receive_ascii_data(dir_lightcurv,part_str):
    col_1='AperturePhot'
    col_2='bmjd'
    col_3='x_com_cen'
    col_4='y_com_cen'
    data = ascii.read(dir_lightcurv)
    ap_phot = data[col_1]
    bmjd = data[col_2]
    xcen=data[col_3]
    ycen=data[col_4]
#     ap_phot = median_flux(ap_phot)
    plt.plot(bmjd,ap_phot,'.',label=part_str)
    return(ap_phot,bmjd,xcen,ycen)
def clip_of_mask_flux(clip_flux):
    clip = sigma_clip(clip_flux, sigma=sigma_num, sigma_lower=None, sigma_upper=None, iters=5, cenfunc=np.ma.median, stdfunc=np.std, axis=None, copy=True)
    clipped = clip[np.logical_not(clip.mask)] - 1.
    return clipped

def clip_of_mask (clp_med_flux, x):
    clip = sigma_clip(clp_med_flux, sigma=sigma_num, sigma_lower=None, sigma_upper=None, iters=5, cenfunc=np.ma.median, stdfunc=np.std, axis=None, copy=True)
    clipped_x = x[np.logical_not(clip.mask)]
    return clipped_x  

def clip_arr(data,time,sigma_num,label_nme,colorpnts):
#     Calls previous definition functions and plots them
#     This is to make the code more neater.
    clip_flux = clip_of_mask_flux(data)
    clip_time = clip_of_mask(data,time)
#     plt.plot(clip_time,clip_flux,'.',label=label_nme,color=colorpnts,)
    return clip_flux,clip_time

def bin_funct(data_arr,nbin):
    data= data_arr.data
    bins=len(data)
    binned_data = [np.mean(data[i*nbin:i*nbin+nbin]) for i in range(1,bins//nbin+1)]
    return binned_data
def choose_fourier(nmode):
    def fourier_model(x,*fp):
        fourier = fp[1] ##setting the offset first.
#         print(fourier)
        for i in range(nmode):
            n=i+1
            fourier = fourier + fp[2+i*2]*np.cos(2.*np.pi*n*x/fp[0]) + fp[3+i*2]*np.sin(2.*np.pi*n*x/fp[0])
        return(fourier)
    return(fourier_model)

# def choose_fourier1(nmode):
#     def fourier_model(x,*fp):
#         fourier = fp[1] ##setting the offset first.
# #         print(fourier)
#         for i in range(nmode+1):
#             n=i
#             fourier = fourier + fp[2+i*2]*np.cos(2.*np.pi*n*x/fp[0]) + fp[3+i*2]*np.sin(2.*np.pi*n*x/fp[0])
#         return(fourier)
#     return(fourier_model)

def choose_ipsv(maxorder):
    def ipsv_model(xy_arr,*p):
        x=xy_arr[0,:]
        y=xy_arr[1,:]
        x_bar=np.mean(xy_arr[0,:])
        y_bar=np.mean(xy_arr[1,:])
        
        index_ipsv = 1
        ipsv = p[0]
        for i in range(maxorder):
            order=i+1
            loop_order=i+2
            for j in range(loop_order):
                ipsv=ipsv+p[index_ipsv]*(x-x_bar)**(order-j)*(y-y_bar)**(j)
                index_ipsv=index_ipsv+1
#         print(index_ipsv)
        return(ipsv)
    return(ipsv_model)

def choose_f_ipsv(nmodes,maxorder):
    def f_ipsv_model(xyt_arr,*p):
        x=xyt_arr[0,:]
        y=xyt_arr[1,:]
        t=xyt_arr[2,:]
        xy = np.array([x,y])
#       functions to find the number of parameters for each model.
        n_ipsv=1
        for i in range(maxorder):#parameters for ipsv
            n_ipsv=n_ipsv +(i+2) 
            
        n_f = 2+2*(nmodes) #parameters for fourier
#         print(n_f,n_ipsv)
#       splitting up actual parameter ( p ) according to model (ipsv then fourier parameters)
        p_ipsv = p[0:n_ipsv]
        p_f = p[n_ipsv::]
#         print(p)
        ipsv_model = choose_ipsv(maxorder)(xy,*p_ipsv)
        f_model = choose_fourier(nmodes)(t,*p_f)
        model = ipsv_model *f_model
        return(model)
    return(f_ipsv_model)

def organize_data(lightcurve):
    data_num = ascii.read(lightcurve)
    #saves in all data into a table array
    col_1='AperturePhot';col_2='bmjd';col_3='x_com_cen';col_4='y_com_cen'
    ap_ph = data_num[col_1]
    bmjd = data_num[col_2]
    x_cen=data_num[col_3]
    y_cen=data_num[col_4]
    
#     takes the median of the flux
#     ap_ph=median_flux(ap_ph)


#     Clips out data points from
#      the lightcurve along with its
#      centroids and time.
#     c_f:clipped flux of original aperture photometry array
#     c_h:clipped time of the flux
    c_f =clip_of_mask(ap_ph,ap_ph)
    c_h = clip_of_mask(ap_ph,bmjd)
    c_cenx = clip_of_mask(ap_ph,x_cen)
    c_ceny = clip_of_mask(ap_ph,y_cen)
    
#     c_fbin = bin_funct(c_f,nbin_num);c_f = c_fbin
#     c_hbin = bin_funct(c_h,nbin_num);c_h = c_hbin
#     c_cenxbin = bin_funct(c_cenx,nbin_num);c_cenx = c_cenxbin
#     c_cenybin = bin_funct(c_ceny,nbin_num);c_ceny = c_cenybin
    
    arr=np.array([c_f,c_h,c_cenx,c_ceny])
    return(arr)