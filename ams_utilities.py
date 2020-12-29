
"""
ams_utilities 
Analysis and data manipulation utilities
Andrew Shelton 2020-02-23 

"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
import pyabf
import seaborn as sns
import scipy
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.image as mpimg
from scipy.stats import beta
import math
from scipy.stats import linregress
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import visual_behavior.utilities as vbu



def GS_to_df(certificate,sheet = None,tab=0):
	"""
	For converting a Google spreadsheet into a pandas-readable csv. Can be any sheet of the spreadsheet but must include the sheet ID (gid=...).
	url = shared Google spreadsheet url
	"""
	import pygsheets

	#authorization
	gc = pygsheets.authorize(service_file=certificate)
	#open GS
	sh = gc.open(sheet)
	wks = sh[tab]
	#get data
	data = wks.get_all_values()
	headers = data.pop(0)
	#make into dataframe
	df = pd.DataFrame(data,columns=headers)

	'this code is depricated due to a parsing error in python'
	# csv_export_url = url.replace('/edit#gid=', '/export?format=csv&gid=')
	# df = pd.read_csv(csv_export_url)
	# export?format=csv'
	return df

def smooth_timeseries(x,y,N=None):
	"""
	Interpolate between points in a timeseries in order to smooth and reduce noise. Mostly for
	display purposes. 
	x,y = orginal xy values of data
	N = number of datapoints to include from the original data. NOTE: too low of values may generate erroneous smoothing unrepresentative of orginal data.
	t,c,k = vectors (arrays) of interpolation knots, B-spline coefficients, and and degree of the spline, respectively.

	returns smoothed_x as all the x-values of the new timeseries and smoothed_y as the BSpline function of those x-values. Call smoothed_y(smoothed_x) to plot.
	"""

	t, c, k = scipy.interpolate.splrep(x, y, s=0, k=4)
	xmin, xmax = x.min(), x.max()
	smoothed_x = np.linspace(xmin, xmax, N)
	smoothed_y = scipy.interpolate.BSpline(t,c,k,extrapolate=False)

	return smoothed_x, smoothed_y

def binomialCI(successes=None,attempts=None,alpha=0.05):
	""" 
	Written 2015. Computes the uppor and lower bounds of a binomial confidence interval about a given set of data.
	"""

	x = successes
	n = attempts    

	# NOTE: the ppf (percent point function) is equivalent to the inverse CDF
	lower = beta.ppf(alpha/2,x,n-x+1)
	if math.isnan(lower):
		lower = 0
	upper = beta.ppf(1-alpha/2,x+1,n-x)
	if math.isnan(upper):
		upper = 1
        
	return lower,upper
    
def CI_plotting(successes,attempts,mean,alpha=0.05):
	"""
	For use with ax.errorbar when plotting errors in matplotlib_pyplot. The function ax.error useses 
	a y-value as the second input to add or subtract a yerr from. Using the binomialCI function above 
	gives only the points where there error bar would end, not the value ax.errorbar computes. 
	This function circumvents having to add/subtract the mean when plotting.

	"""
	lower,upper = binomialCI(successes,attempts)

	lower_error = mean - lower
	upper_error = upper - mean

	return lower_error,upper_error

def calculate_ts_derivative(df,ts):
	""" Calculate the derivative of a time-series """
	
	dt = np.gradient(ts)

	df['dt'] = dt

	return df
	
def make_abf_df(abf):
	""" for making a pandas dataframe out of .abf files, a common format for ephys data produced by Clampfit and related software.
	abf = pyabf.ABF(abf_path)   """

	df = pd.DataFrame()
	df_sweep = []
	df_trigger = []
	df_TTL = []

	for sweepNumber in abf.sweepList:
		for channel in abf.channelList:
			if channel==0:
				abf.setSweep(sweepNumber,channel)
				df_sweep.append(abf.sweepY)
			if channel==1:
				abf.setSweep(sweepNumber,channel)
				df_trigger.append(abf.sweepY)
			if channel==2:
				abf.setSweep(sweepNumber,channel)
				df_TTL.append(abf.sweepY)

	for i,sweep in enumerate(df_sweep):
		df['sweep'+str(i)] = df_sweep[i]
	for ii,trigger in enumerate(df_trigger):
		df['trigger'+str(ii)] = df_trigger[ii]
	for iii,TTL in enumerate(df_TTL):
		df['TTL'+str(iii)] = df_TTL[iii]
	df['time'] = abf.sweepX
	channels = ['sweep','trigger','TTL']
	for u,unit in enumerate(abf.adcUnits):
		df[channels[u]+' units'] = abf.adcUnits[u]
	df['datapath'] = abf.abfFilePath
	df['protocol'] = abf.protocol
	df['NoS'] = abf.sweepNumber

	return df

def add_sweep_avg_to_df(df):
	"""iterrate over every sample in a series of sweeps and compute an average, then add
	a new DataFrame column to make_abf_df"""
	sweeps = [col for col in df.columns if 'sweep' in col]

	dft = df[sweeps]

	df_sweep_avg = [np.mean(trial) for idx,trial in dft.iterrows()]

	df['sweep_avg'] = df_sweep_avg

	return df
	    
def load_image(path):
	""" load an image into python.
	path = path to the image"""
	image = cv2.imread(path)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	return image

def p2f(x):
#a simple function for converting a percentage string to a float decimal
	return float(x.strip('%'))/100

def str_flt(array):
	"""
	create a new array of floats from and array of strings
	"""
	import re, ast
	try:
		new_array = array.astype(np.float)
	except:
		array = re.sub('\s+', ' ', array)
		new_array  = np.array(ast.literal_eval(array)) 
	
	return new_array

def transpose_raw_df(raw_df,cols=None,drop=None):
	"""
	flip a dataframe along its diagonal axis and give the columns names from cols. Allows for pre-determined elements to be dropped
	"""
	raw_df_T=raw_df.transpose()
	raw_df_T.columns=raw_df_T.loc[cols]
	df=raw_df_T.drop(drop)
	return df
	
def save_figure(fig, fname, formats = ['.pdf'],transparent=False,dpi=300,facecolor=None,**kwargs):
	"""save a matplotlib.pyplot figure as any file format (commonly .pdf, .png, .jpeg)
	fig = fig to be saved
	fname = path for figure to be saved to. generally as r'C:/.../.../{}'.format(figure name)
	"""
	import matplotlib as mpl
	mpl.rcParams['pdf.fonttype'] = 42

	if 'size' in kwargs.keys():
		fig.set_size_inches(kwargs['size'])

	elif 'figsize' in kwargs.keys():
		fig.set_size_inches(kwargs['figsize'])
	# else:
	#     fig.set_size_inches(11,8.5)
	for f in formats:
		fig.savefig(fname + f, transparent = transparent,dpi=dpi)

def get_response_windows(stims,df,before=5000,after=20000,start=0,stop=None,step=3):

    '''Get time series of data in a window around an index or list of indicies.
Provide a DataFrame object and window (frame integers) around the stim index to reference into. Returns a list
of values corresponding to the window
''' 
    response_windows = [df['Membrane Voltage (mV)'].iloc[stim-before:stim+after].values for i,stim in enumerate(stims[start:stop:step])]

    return response_windows

def stim_onset(data,distance=None,num_std=None):
	stim_frames = find_peaks(data,height=(np.mean(data)+np.std(data)*num_std),width=1)[0]
	stims = [stim_frames[0]]
	for i,onset in enumerate(stim_frames[:len(stim_frames)-1]):
		if (stim_frames[i]-stim_frames[i-1])>distance:
			stims.append(onset)
	return stims

def find_pulse(data,mean_intvl=5000,std=5,pulse_width=50):
	import more_itertools as mit

	stim_start = np.where(data>np.mean(data.iloc[:mean_intvl])+np.std(data.iloc[:mean_intvl])*std)[0]

	steps = [list(group) for group in mit.consecutive_groups(stim_start)]

	stims = [step for step in steps if len(step)>pulse_width]

	return(stims)


def points_in_circle_np(radius, x0=0, y0=0, ):
	""" 
	For xy points in a coordinate plane, generate a circle of radius = radius (in pixels) around point (x0,y0)
	"""
	x_ = np.arange(x0 - radius - 1, x0 + radius + 1, dtype=int)
	y_ = np.arange(y0 - radius - 1, y0 + radius + 1, dtype=int)
	x, y = np.where((x_[:,np.newaxis] - x0)**2 + (y_ - y0)**2 <= radius**2)
	for x, y in zip(x_[x], y_[y]):
		yield x, y

	
	""" Sample use:
	im_Gnb4 = tf.imread(“image.tif”)
	plt.imshow(im_Gnb4)
	im_Gnb4_shape = np.shape(im_Gnb4)
	coords = data.iloc[:-3,2:4]
	coords *= 8.1097
	# print (im_shape)
	# print (len(cell_area))
	coords = round(coords)
	radius = 15
	avg_intensity = []
	for index, row in coords.iterrows():
	    sum_intensity = 0
	    cell_area = [item for item in points_in_circle_np(radius, x0=row[‘Y’], y0=row[‘X’])]
	    for pixel in cell_area:
		if not (pixel[1]<0 or pixel[1]>im_Gnb4_shape[1] or pixel[0]<0 or pixel[0]>im_Gnb4_shape[0]):
		    sum_intensity += im_Gnb4[pixel]
	    avg_intensity.append(sum_intensity/len(cell_area))
	# print(np.amax(avg_intensity))
	# print(np.amin(avg_intensity))
	Imax = np.amax(avg_intensity)
	Imin = np.amin(avg_intensity)
	num = avg_intensity - Imin
	denom = Imax - Imin
	norm_AI_Gnb4 = num/denom
	#print(norm_AI_Gnb4)
	temp_df_Gnb4 = pd.DataFrame(norm_AI_Gnb4, columns=[‘norm_AI_Gnb4’])
	data_Gnb4 = pd.concat([data, temp_df_Gnb4], axis=1, sort=False)
	data_Gnb4
	"""
