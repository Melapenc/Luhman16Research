import sys
from astropy.io import ascii
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
# from astropy.convolution import convolve
from astropy.stats import sigma_clip
from scipy.optimize import curve_fit
import os
import astropy.io.fits as fits
import matplotlib.lines as lines
import string

sigma_num=5

def fourier_sinusoidal_series_twoModes(x,a0,b0,a1,b1,a2,b2,freq,offset):
    result = a0*np.cos(2.*0*np.pi*freq*x) \
             + b0*np.sin(2.*0*np.pi*freq*x) \
             + a1*np.cos(2.*1*np.pi*freq*x) \
             + b1*np.sin(2.*1*np.pi*freq*x) \
             + a2*np.cos(2.*2*np.pi*freq*x) \
             + b2*np.sin(2.*2*np.pi*freq*x) \
             + offset
    return result

def clip_of_mask_flux(clip_flux):
    clip = sigma_clip( clip_flux, sigma=sigma_num, sigma_lower=None, sigma_upper=None, iters=5, cenfunc=np.ma.median, stdfunc=np.std, axis=None, copy=True)
    clipped = clip[np.logical_not(clip.mask)] - 1.
    return clipped
def clip_of_mask (clp_med_flux, x):
    clip = sigma_clip(clp_med_flux, sigma=sigma_num, sigma_lower=None, sigma_upper=None, iters=5, cenfunc=np.ma.median, stdfunc=np.std, axis=None, copy=True)
    clipped_x = x[np.logical_not(clip.mask)]
    return clipped_x
def med_flux (x):
    median = np.median(x)
    x_divide_med = x / median
    med_x = np.array(x_divide_med)
    return med_x
def amp_funct(x):
    min_value = min(x)
    max_value = max(x)
    med_value = np.median(x)
    ans = ((max_value - min_value) / med_value)
    return ans
# ---------------------------------------
def Amp_with_Coeff(Extraction_Coeff, Delta_area, ):
    Delta_A = Delta_area
    Average_A = 0.2
    one = 1.
    Ext_coeff = Extraction_Coeff

    epsilon = (one / (Ext_coeff - one))
    Amplitude = (-1 * Delta_A) / (Average_A + epsilon)
    return Amplitude
    # print "\n -----------------------------------------------------------------"
    # print "\n Amplitude with Extraction Coefficient: \n", Amplitude


def make_Amp_MiePython(Datafile):
    def Amp_x(Data_wavelength, Delta_a):
        Intensity = ascii.read(Datafile)
        wavelength = Intensity['Wavelength']
        extraction = Intensity['Transmission']
        Amp_Ext_Coeff = Amp_with_Coeff(extraction, Delta_a)
        Inter_Amp = np.interp(Data_wavelength, wavelength, Amp_Ext_Coeff)
        return Inter_Amp
    return Amp_x

def make_Amp_x(Datafile):
    def Amp_x(Data_wavelength, Delta_a):
        Intensity = ascii.read(Datafile)
        wavelength = Intensity['col1']
        extraction = Intensity['col2']
        Amp_Ext_Coeff = Amp_with_Coeff(extraction, Delta_a)
        Inter_Amp = np.interp(Data_wavelength, wavelength, Amp_Ext_Coeff)
        return Inter_Amp
    return Amp_x

def chi_squared_GOF(Mi, Yi, sigma_error_i):
    # the sum of the value minus of the expected value divided by the sigma of the expected all of it is squared
    # Chi-Squared (Goodness of Fit)
    # chi_squared = sum((Yi - Mi) ** two) / sum(sigma_error_i**two)
    two = 2
    chi_squared = sum(((Yi - Mi) ** two) / sigma_error_i ** two)
    return chi_squared


