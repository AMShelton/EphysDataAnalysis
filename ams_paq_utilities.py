import os
import sys
import pyabf
import paq2py
import numpy as np
import pandas as pd
import seaborn as sns
# import DoC_tools as dt
import scipy.interpolate
from ams_utilities import *
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
# import visual_behavior.utilities as vbu
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''
Functions for creating single or multiple pandas DataFrames containing experiment-relevant ephys data and experiment labels.
'''

def p2p(datapath,plot=None):
	data = paq2py.paq_read(datapath,plot=plot)

	return data

def get_date(df):
	date = ''.join(df['Recording_Date'].unique())

	return date

def get_voltage(data):
	voltage = data['data'][0]

	return voltage

def get_current(data):
	current = data['data'][1]

	return current

def make_paq_df_multi(path):

	df = pd.DataFrame()

	if 'allparams.paq' in path:
		data = p2p(path,plot=False)
		df['Voltage'] = get_voltage(data)
		df['Current'] = get_current(data)
		df['time'] = df.index/data['rate']
		df['mouseID'] = path.split('_')[3]
		df['Recording_Date'] = path.split('_')[2].split("s\\")[1]
		df['data_path'] = path
	else:
		pass

	if 'Cell' in path:
		df['Cell'] = path.split('_')[10]
	else:
		pass

	return df


def make_paq_df_single(datapath,plot=None):

	df = pd.DataFrame()

	# if 'allparams' in datapath:
	data = p2p(datapath,plot=plot)
	df['Voltage'] = get_voltage(data)
	df['Current'] = get_current(data)
	df['time'] = df.index/data['rate']
	df['mouseID'] = datapath.split('_')[3]
	# else:
	# 	pass

	return df

def raw_df_convert(raw_df,lines_to_drop=None,standardise=True):
    # drop unwanted rows from dataset
    df = raw_df.drop(lines_to_drop, axis=0)
    df.dropna(inplace = True)
    df = df.reset_index(drop=True)
    cols = df['metric'].values
    df = str_flt(df.iloc[:,1:])

    # Convert to float array and standardise data ((x - mean)/std)
    if standardise==True:

        mean,std = np.mean(df) , np.std(df)
        df -= mean
        df /= std

    # Transpose array so variables arranged column-wise 
    data = df.T
    data.columns = cols
    data.columns.names = ['metric']
    data.index.names = ["cell"]
    data
    
    return data

def IV_df(df):
	IV_df = pd.DataFrame(pd.np.r_[
		# df[15000:20000],
		# df[35000:40000],
		df[55000:60000],
		df[75000:80000],
		df[95000:100000],
		df[115000:120000],
		df[135000:140000],
		df[155000:160000],
		df[175000:180000],
		df[195000:200000],
		df[215000:220000]],columns=df.columns)

	return IV_df

def rheobase_df(df):
	rb_df = df[280000:285050]

	# rb_df = df[280020:285100]

	return rb_df

def rheobasex2_df(df):
	R2_df = df[294000:311000]

	return R2_df

def ramp_df(df):
	ramp_df = df[319800:325500]

	return ramp_df

def rb2x10_df(df):
	rb2x10_df = [df[347000:352025],
		df[362000:367025],
		df[377000:382025],
		df[392000:397025],
		df[407000:412025],
		df[422000:427025],
		df[437000:442025],
		df[452000:457025],
		df[467000:472025],
		df[482000:487025]]

	return rb2x10_df

def Ih_df(df):
	Ih = df[505000:560000]

	return Ih

def rheobasex4_df(df):
	R4_df = df[567000:572100]

	return R4_df
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''
Functions for finding element-wise differences. Probably unneccesary considering np.diff() is a thing. Helpfully returns
both differences and indicies of elements
'''
def get_v_diffs(df=None):

	v_diffs = []
	index = []

	for i in range(0, len(df)):
	    if i<(len(df)-1):
	        v_diff = df['Voltage'].iloc[i+1] - df['Voltage'].iloc[i]
	        v_diffs.append(v_diff)
	        index.append(df.index[i])

	return v_diffs, index

def get_c_diffs(df=None):

	c_diffs = []
	index = []

	for i in range(0, len(df)):
	    if i<(len(df)-1):
	        c_diff = df['Current'].iloc[i+1] - df['Current'].iloc[i]
	        c_diffs.append(c_diff)
	        index.append(df.index[i])

	return c_diffs, index

def get_t_diffs(df=None):

    t_diffs = []
    index = []

    for i in range(0, len(df)):
        if i<(len(df)-1):
            t_diff = df['time'].iloc[i+1] - df['time'].iloc[i]
            t_diffs.append(t_diff)
            index.append(df.index[i])
            
    return(t_diffs)

def get_onset_offset_index(df,spike=0,min_rate=1.5):

	'''
	Crucially important function for finding the onset/offset of spikes based on the rate change of Vm. Importantly, 
	this will generally only be useful for the first spike in a series.
	'''
	peaks = find_peaks(df['Voltage'],height=0.0)
	peak = peaks[0][spike] + df.index[0]
	v_diffs, index = get_v_diffs(df=df.loc[peak-40:peak])
	onset_index = [index[ii] for ii,rate in enumerate(v_diffs) if rate>=min_rate][0]
	if len(peaks[0])>1:
		if (peaks[0][1] - peaks[0][0])<200:
			offset_index =  np.where(df['Voltage'].loc[peak:]==min(df['Voltage'].loc[peaks[0][0]+ df.index[0]:peaks[0][1]+ df.index[0]]))[0][0] + peak
		else:
			offset_index =  np.where(df['Voltage'].loc[peak:]<df['Voltage'][onset_index]+1.0)[0][0] + peak
	else:
		offset_index =  np.where(df['Voltage'].loc[peak:]<df['Voltage'][onset_index]+1.0)[0][0] + peak
	
	
	return onset_index,offset_index
	
	# peak = find_peaks(df['Voltage'],height=0.0)[0][spike] + df.index[0]
	# v_diffs, index = get_v_diffs(df=df.loc[peak-40:peak])
	# onset_index = [index[ii] for ii,rate in enumerate(v_diffs) if rate>=min_rate][0]
	# offset_index =  np.where(df['Voltage'].loc[peak:]<df['Voltage'][onset_index]+1.0)[0][0] + peak
	
	return onset_index,offset_index
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''
PARAM_DF FUNCTIONS:
Functions for general cell properites
'''
def Resting_Potential(df):
	Resting_p = np.mean(df['Voltage'][0:5000])

	return Resting_p

def Baselineshift(df):
	start = np.mean(df['Voltage'][0:5000])
	end = np.mean(df['Voltage'][(len(df['Voltage']) - 5000):])
	Baselineshift = end - start

	return Baselineshift

def input_resistance(df):

	IV = IV_df(df)
	hold = np.mean(df['Current'].loc[0:5000].mean())
	y = IV['Voltage'].astype(float).groupby(IV.index//5001.0).mean() 
	x = IV['Current'].astype(float).groupby(IV.index//5001.0).mean() - hold
	input_resistance = linregress(x,y)[0]*1000

	return input_resistance

def tau(df):

	df_tau = df[50000:53000]

	x0 = df_tau['time']
	y0 = df_tau['Voltage']

	def func(t, a, tau, c):
		return a * np.exp(-t / tau) + c

	c_0 = y0.values[1]
	tau_0 = 1
	a_0 = (y0.values[0] - y0.values[-1])

	popt, pcov = curve_fit(func, x0, y0,p0=(a_0, tau_0, c_0),maxfev=100000)
	x = np.linspace(5.0,5.3,5000)
	y = func(x,*popt) 

	if np.std(y)<=1e-2:
		try:
			tau_0 = 4
			popt, pcov = curve_fit(func, x0, y0,p0=(a_0, tau_0, c_0),maxfev=10000000)
			x = np.linspace(5.0,5.3,5000)
			y = func(x,*popt)  
		except:
			pass

	tau = popt[1]*1000

	return tau

'''
Functions called on the rheobase firing period
'''

def Rheobase(df):
	rheobase = np.mean(df['Current'][282000:284000]) - np.mean(df['Current'][0:5000])
	# ramp = ramp_df(df)
	# on,off = get_onset_offset_index(ramp)
	# rheobase = ramp['Current'].loc[on] - np.mean(df['Current'].loc[0:5000])

	return rheobase 


def AP1_amp(df):
	SFR = rheobase_df(df)
	AP1 = find_peaks(SFR['Voltage'],height=0.0)[1]['peak_heights'][0]

	return AP1

def AP_duration(df):

	SFR = rheobase_df(df)
	        
	onset_index, offset_index = get_onset_offset_index(df=SFR)
	onset_time = df['time'].loc[onset_index]
	offset_time = df['time'].loc[offset_index]
	AP_duration = 1000*(offset_time - onset_time) 

	return AP_duration

def AP1_abs_amp(df):

	SFR = rheobase_df(df)
	onset_index, offset_index = get_onset_offset_index(df=SFR)
	AP_amp = AP1_amp(df) + abs(SFR['Voltage'].loc[onset_index])

	return AP_amp

def AP_half_Width(df):
	#Note: this version has been updated to find the indices nearest to the value of the hw to accommodate for a finte sampling rate
	'''Find AP1 absolute amplitude above the onset voltage and AP1 max height. Subtract AP1 height from AP1 amplitude/2 to get hw voltage. Next, find the sample points to that voltage on either side of the peak.
	Finally, find the dt from idx1 to idx2 to get spike half width. This approach isn't perfect due to finite sample rate but does give a fairly good approximation. '''

	rb = rheobase_df(df)

	AP_amp = AP1_abs_amp(df)
	AP1 = AP1_amp(df)
	AP_hw_amp = AP1 - AP_amp/2
	onset_idx, offset_idx = get_onset_offset_index(df=rb)
	peak_idx = find_peaks(rb['Voltage'],height=0.0)[0][0] + rb.index[0]

	pre_spike = rb.loc[onset_idx:peak_idx]
	post_spike = rb.loc[peak_idx:offset_idx]

	hw_idx_1 = pre_spike.iloc[(pre_spike['Voltage']-AP_hw_amp).abs().argsort()[:1]].index[0]
	hw_idx_2 = post_spike.iloc[(post_spike['Voltage']-pre_spike['Voltage'].loc[hw_idx_1]).abs().argsort()[:1]].index[0]

	hw_time_1 = rb['time'].loc[hw_idx_1]
	hw_time_2 = rb['time'].loc[hw_idx_2]
	AP_hw = (hw_time_2 - hw_time_1)*1000.0

	return AP_hw

def Rise_time(df):

	SFR = rheobase_df(df)
	onset_index, offset_index = get_onset_offset_index(df=SFR)
	onset_time = df['time'][onset_index]
	peak_idx = find_peaks(SFR['Voltage'],height=0.0)[0][0]
	v_diffs, index = get_v_diffs(df=SFR)
	peak_time = SFR['time'].loc[index[0]+peak_idx]
	rise_time  = (peak_time - onset_time)*1000

	return rise_time

def Fall_time(df):
	SFR = rheobase_df(df)
	onset_index, offset_index = get_onset_offset_index(df=SFR)
	offset_time = SFR['time'].loc[offset_index]
	peak_idx = find_peaks(SFR['Voltage'],height=0.0)[0][0]
	v_diffs, index = get_v_diffs(df=SFR)
	peak_time = SFR['time'].loc[index[0]+peak_idx]
	fall_time = (offset_time - peak_time)*1000

	return fall_time

def Rise_rate(df):

	AP_amp = AP1_abs_amp(df)
	rise_time = Rise_time(df)
	rise_rate = AP_amp/rise_time

	return rise_rate

def Fall_rate(df):

	AP_amp = AP1_abs_amp(df)
	fall_time = Fall_time(df)
	fall_rate = AP_amp/fall_time

	return fall_rate

def ten_ninty_rise_time(df):

	SFR = rheobase_df(df)
	onset_index, offset_index = get_onset_offset_index(df=SFR)
	onset_voltage = df['Voltage'][onset_index]
	AP_amp =  AP1_abs_amp(df)
	AP1 = AP1_amp(df)

	ten_ninty_idx = np.where(np.logical_and(SFR['Voltage']>=onset_voltage+(0.1*AP_amp), 
                                    SFR['Voltage']<=0.9*AP_amp))

	ninty_time = SFR['time'][SFR['Voltage']>=(AP1)*0.9].index[0]/(SFR['time'].index[0]/np.array(SFR['time'])[0])

	ten_time = SFR['time'][SFR['Voltage']>=(onset_voltage+(AP_amp*0.1))].index[0]/(SFR['time'].index[0]/np.array(SFR['time'])[0])

	ten_ninety_rt = (ninty_time - ten_time)*1000

	return ten_ninety_rt

# def Refractory_period(df):

# 	rb_df = rheobase_df(df)
# 	peaks = find_peaks(rb_df['Voltage'],height=0.0,distance=20)
# 	rb_v_diffs, rb_index = get_v_diffs(df=rb_df)
# 	rb_onset_index, rb_offset_index = get_onset_offset_index(df=rb_df)

# 	rb_onset_voltage = rb_df['Voltage'][rb_onset_index]       
# 	rb_onset_time = rb_df['time'][rb_onset_index]
# 	rb_offset_voltage = rb_df['Voltage'].loc[rb_offset_index]
# 	rb_offset_v = (rb_df['Voltage'].loc[rb_offset_index:])

# 	rebound_array = [i for i in rb_offset_v if i > rb_onset_voltage+2]

# 	if len(rebound_array)>1:

# 	    rb_rebound_index = np.where(rb_offset_v > rb_onset_voltage+2)[0][0] + rb_offset_index
# 	    rb_offset_time = rb_df['time'].loc[rb_offset_index]
# 	    rb_rebound_time = rb_df['time'].loc[rb_rebound_index]
# 	    rb_Refractory_period = 1000*(rb_rebound_time - rb_offset_time)

# 	else:
# 	    rb_rebound_index = np.where(np.diff(rb_df['Current'].loc[rb_offset_index:])==min(np.diff(rb_df['Current'].loc[rb_offset_index:])))[0][0] + rb_offset_index
# 	    rb_Refractory_period = 'N/A'

# 	node = find_peaks(rb_df['Voltage'].loc[int(rb_offset_index):int(rb_offset_index)+300],
# 	                      distance=50,
# 	                      height=min(rb_df['Voltage'].loc[int(rb_offset_index):rb_rebound_index])+0.5,
# 	                      prominence=0.75,
# 	                      # threshold=sAHP-100,
# 	                      width=30)

# 	if len(node[0]>0):
# 		ADP_idx = node[0][0] + rb_offset_index
# 		fAHP_min = min(rb_df['Voltage'].loc[int(rb_offset_index):ADP_idx])
# 		fAHP = rb_onset_voltage - fAHP_min
# 		ADP = node[1]['peak_heights'][0]-fAHP_min
# 		if len(peaks[0])<=1:
# 			sAHP_min = np.min(rb_df['Voltage'].loc[ADP_idx:ADP_idx+1500])
# 			sAHP = rb_onset_voltage - sAHP_min
# 		else:
# 			sAHP_min = np.min(rb_df['Voltage'].loc[ADP_idx:peaks[0][1] + rb_df.index[0]])
# 			sAHP = rb_onset_voltage - sAHP_min
# 	else:#activated only without node
# 		ADP = 0.0
# 		ADP_idx = np.nan
# 		fAHP_min = min(rb_df['Voltage'].loc[int(rb_offset_index):int(rb_offset_index)+50])
# 		fAHP = rb_onset_voltage - fAHP_min
# 		fAHP_idx = np.where(rb_df['Voltage'].loc[int(rb_offset_index):rb_rebound_index]==fAHP_min)[0][0]+rb_offset_index
# 		if len(peaks[0])>1:
# 			sAHP_min = min(rb_df['Voltage'].loc[int(rb_offset_index)+51:peaks[0][-1]+rb_df.index[0]])
# 			sAHP = rb_onset_voltage - sAHP_min  
# 		else:
# 			sAHP_min = min(rb_df['Voltage'].loc[int(rb_offset_index)+51:rb_rebound_index]) 
# 			sAHP = rb_onset_voltage - sAHP_min

# 	if len(peaks[0])>1:
# 		if peaks[0][1]-peaks[0][0]<300:
# 			on2,off2 = get_onset_offset_index(rb_df.loc[rb_offset_index:],min_rate=3.0)	
# 			ADP = rb_df['Voltage'][on2] -fAHP_min
# 			ADP_idx = on2#peaks[0][1] + rb_df.index[0]
# 			sAHP_min = np.min(rb_df['Voltage'].loc[ADP_idx:ADP_idx+1500])
# 			sAHP = rb_onset_voltage - sAHP_min
 
# 	return fAHP_min,fAHP,sAHP_min,sAHP,ADP,ADP_idx

def Refractory_period(df):

	rb_df = rheobase_df(df)
	on,off = get_onset_offset_index(rb_df)
	peaks = find_peaks(rb_df['Voltage'],height=0.0)[0] + rb_df.index[0]

	if len(peaks)>1:
		on2,off2 = get_onset_offset_index(rb_df,spike=1)

		if (peaks[1] - peaks[0])<=300:
			ADP_idx = on2
			fAHP_idx = np.where(rb_df['Voltage'].loc[peaks[0]:ADP_idx]==min(rb_df['Voltage'].loc[peaks[0]:ADP_idx]))[0][0] + peaks[0]
			fAHP_min = min(rb_df['Voltage'].loc[peaks[0]:ADP_idx])
			fAHP = fAHP_min - rb_df['Voltage'].loc[on]
			ADP = rb_df['Voltage'].loc[ADP_idx] - fAHP_min

		else:
			node = find_peaks(rb_df['Voltage'].loc[off:off+500],
				distance=100,
				height=min(rb_df['Voltage'].loc[off:off+500]),
				prominence=0.75,
				# threshold=sAHP-100,
				width=30)

			if len(node[0])>0:
				ADP_idx = node[0][0] + off
				fAHP_idx = np.where(rb_df['Voltage'].loc[peaks[0]:ADP_idx]==min(rb_df['Voltage'].loc[peaks[0]:ADP_idx]))[0][0] + peaks[0]
				fAHP_min = min(rb_df['Voltage'].loc[peaks[0]:ADP_idx])
				fAHP = fAHP_min - rb_df['Voltage'].loc[on]
				ADP = rb_df['Voltage'].loc[ADP_idx] - fAHP_min
			else:
				ADP_idx = np.nan
				ADP = 0.0
				fAHP_idx = np.where(rb_df['Voltage'].loc[peaks[0]:peaks[1]]==min(rb_df['Voltage'].loc[peaks[0]:peaks[1]]))[0][0] + peaks[0]
				fAHP_min = min(rb_df['Voltage'].loc[peaks[0]:peaks[1]])
				fAHP = fAHP_min - rb_df['Voltage'].loc[on]

	########################################################################################################################################                
	else:
		
		node = find_peaks(rb_df['Voltage'].loc[off:off+500],
			distance=100,
			height=min(rb_df['Voltage'].loc[off:off+500])+0.5,
			prominence=0.75,
			# threshold=sAHP-100,
			width=20)
		if len(node[0])>0:
			ADP_idx = node[0][0] + off
			fAHP_idx = np.where(rb_df['Voltage'].loc[peaks[0]:ADP_idx]==min(rb_df['Voltage'].loc[peaks[0]:ADP_idx]))[0][0] + peaks[0]
			fAHP_min = min(rb_df['Voltage'].loc[peaks[0]:ADP_idx])
			fAHP = fAHP_min - rb_df['Voltage'].loc[on]
			ADP = rb_df['Voltage'].loc[ADP_idx] - fAHP_min
	        
		else:
			ADP_idx = np.nan
			ADP = 0.0
			fAHP_idx = np.where(rb_df['Voltage'].loc[peaks[0]:285000]==min(rb_df['Voltage'].loc[peaks[0]:285000]))[0][0] + peaks[0]
			fAHP_min =  min(rb_df['Voltage'].loc[peaks[0]:285000])
			fAHP = fAHP_min - rb_df['Voltage'].loc[on]

	return fAHP_min, fAHP, fAHP_idx, ADP, ADP_idx

def sAHP(df):
	sAHP_min = min(df['Voltage'].loc[310000:320500])
	sAHP = sAHP_min - np.mean(df['Voltage'].loc[275000:280000])
	sAHP_index = np.where(df['Voltage'].loc[310000:320500]==min(df['Voltage'].loc[310000:320500]))[0] + 310000

	return sAHP_min, sAHP, sAHP_index

def delay_to_spike(df):
# time from current pulse onset to first action potential
	rb = rheobase_df(df)
	on,off = get_onset_offset_index(rb)
	delay = (rb['time'][on] - rb['time'].iloc[0])*1000

	return delay

def delay_to_fAHP(df):
	rb=rheobase_df(df)
	peaks = find_peaks(rb['Voltage'],height=0.0)[0]
	peaks += rb.index[0]
	fAHP = Refractory_period(df)[0]
	fAHP_delay_index = np.where(rb['Voltage'].loc[peaks[0]:peaks[0]+2500]==fAHP)[0][0] + peaks[0]
	fAHP_delay = ((rb['time'][fAHP_delay_index]-rb['time'].iloc[0])*1000) - ((rb['time'][peaks[0]] - rb['time'].iloc[0])*1000)

	return fAHP_delay

'''
Functions for 2x rheobase firing
'''
def AP12_drop(df):
	AP_range  = rheobasex2_df(df)
	APs = find_peaks(AP_range['Voltage'],height=0.0)[1]
	AP12_drop = APs['peak_heights'][0] - APs['peak_heights'][1]

	return AP12_drop

def APfl_drop(df):
	AP_range  = rheobasex2_df(df)
	APs = find_peaks(AP_range['Voltage'],height=0.0)[1]
	APfl_drop = APs['peak_heights'][0] - APs['peak_heights'][len(APs['peak_heights'])-1]

	return APfl_drop

def AP2l_drop(df):
	AP_range  = rheobasex2_df(df)
	APs = find_peaks(AP_range['Voltage'],height=0.0)[1]
	AP2l_drop = APs['peak_heights'][1] - APs['peak_heights'][len(APs['peak_heights'])-1]

	return AP2l_drop

def Max_AP_change(df):
	AP_range  = rheobasex2_df(df)
	APs = find_peaks(AP_range['Voltage'],height=0.0)[1]

	diffs = []

	for i,value in enumerate(APs['peak_heights']):
	    diff = abs(APs['peak_heights'][len(APs['peak_heights'])-1] - value)
	    diffs.append(diff)
	    
	max_diff = max(diffs)

	return max_diff

'''
Functions called on ramp depolarization
'''
def Threshold(df):
	ramp = ramp_df(df)
	rb = rheobase_df(df)
	peaks = find_peaks(ramp['Voltage'],height=0.0)[0]
	peaks +=ramp.index[0]
	if len(peaks)>0:
		try:
			on,off = get_onset_offset_index(ramp)
			threshold = ramp['Voltage'].loc[on]
		except:
			on,off = get_onset_offset_index(ramp,spike=0,min_rate=3.0)
			threshold = ramp['Voltage'].loc[on]
	else:
		on,off = get_onset_offset_index(rb)
		threshold = rb['Voltage'].loc[on]

	return(threshold)

'''
Functions called on 2x10 rheobase firing
'''
def AP1_delay(df):
	rb210 = rb2x10_df(df)
	delays = []
	for i,dft in enumerate(rb210):
		try:
			on,off = get_onset_offset_index(dft,spike=0,min_rate=4.0)
		except IndexError:
			try:
				on,off = get_onset_offset_index(dft,spike=0,min_rate=6.0)
			except :
				try:
					on,off = get_onset_offset_index(dft,spike=0,min_rate=8.0)
				except:
					try:
						on,off = get_onset_offset_index(dft,spike=0,min_rate=10.0)
					except:
						on,off = get_onset_offset_index(dft,spike=0,min_rate=14.0)
		delay = (dft['time'].loc[on] - dft['time'].iloc[0])*1000
		delays.append(delay)

	return np.mean(delays),np.std(delays)

def AP2_delay(df):
	rb210 = rb2x10_df(df)
	delays = []
	for i,dft in enumerate(rb210):
		peaks = find_peaks(dft['Voltage'],height=0.0)[0]
		peaks+= dft.index[0]
		if len(peaks)>1:
			delay = (dft['time'][peaks[1]] - dft['time'].iloc[0])*1000
			delays.append(delay)
		else:
			delay = (dft['time'].iloc[-1] - dft['time'][peaks[0]])*1000

	return np.mean(delays),np.std(delays)

def rb2x_ISI_1_3(df):
	rb210 =rb2x10_df(df)
	avgs = []
	for i,dft in enumerate(rb210):
		peaks = find_peaks(dft['Voltage'],height=0.0)[0]
		peaks += dft.index[0]
		ISIs = []
		for i,peak in enumerate(peaks):
			try:
				if i>0:
					ISI = (dft['time'][peaks[i]] - dft['time'][peaks[i-1]])*1000
					ISIs.append(ISI)
			except:
				pass
		avgs.append(np.mean(ISIs[:3]))

	return np.nanmean(avgs),np.nanstd(avgs)


def ISI_accom_ratio(df):
	rb210 = rb2x10_df(df)
	AP12_accom = []
	APlast_accom = []

	for i,dft in enumerate(rb210):
		peaks = find_peaks(dft['Voltage'],height=0.0)[0]
		peaks += dft.index[0]
		if len(peaks)>=3:
			ISI1_2 =  ((dft['time'][peaks[2]]-dft['time'][peaks[1]])/(dft['time'][peaks[1]]-dft['time'][peaks[0]]))
			AP12_accom.append(ISI1_2)
			
			ISI_last = ((dft['time'][peaks[-1]]-dft['time'][peaks[-2]])/(dft['time'][peaks[-2]]-dft['time'][peaks[-3]]))
			APlast_accom.append(ISI_last)
		elif len(peaks)==2:
			ISI1_2 = dft['time'][peaks[1]]/dft['time'][peaks[0]]
			ISI_last =  dft['time'][peaks[-1]]/dft['time'][peaks[-2]]
			AP12_accom.append(ISI1_2)
			APlast_accom.append(ISI_last)
		else:
			AP12_accom.append(dft['time'].iloc[-1]/dft['time'][peaks[0]])
			APlast_accom.append(dft['time'].iloc[-1]/dft['time'][peaks[0]])
	    
	return np.mean(AP12_accom),np.std(AP12_accom),np.mean(APlast_accom),np.std(APlast_accom)

def ISI_accom_diff(df):
	rb210 = rb2x10_df(df)
	AP12_accom = []
	APlast_accom = []

	for i,dft in enumerate(rb210):
		peaks = find_peaks(dft['Voltage'],height=0.0)[0]
		peaks += dft.index[0]
		if len(peaks)>=3:
			ISI1_2 =  ((dft['time'][peaks[2]]-dft['time'][peaks[1]])-(dft['time'][peaks[1]]-dft['time'][peaks[0]]))*1000
			AP12_accom.append(ISI1_2)
			
			ISI_last = ((dft['time'][peaks[-1]]-dft['time'][peaks[-2]])-(dft['time'][peaks[-2]]-dft['time'][peaks[-3]]))*1000
			APlast_accom.append(ISI_last)
		elif len(peaks)==2:
			ISI1_2 = (dft['time'][peaks[1]]-dft['time'][peaks[0]])*1000
			ISI_last =  (dft['time'][peaks[-1]]-dft['time'][peaks[-2]])*1000
			AP12_accom.append(ISI1_2)
			APlast_accom.append(ISI_last)
		else:
			AP12_accom.append((dft['time'].iloc[-1]-dft['time'][peaks[0]])*1000)
			APlast_accom.append((dft['time'].iloc[-1]-dft['time'][peaks[0]])*1000)

	return np.mean(AP12_accom),np.std(AP12_accom),np.mean(APlast_accom),np.std(APlast_accom)

def Avg_spike_rate(df):
	num_spikes = []
	rb210 = rb2x10_df(df)
	for i,dft in enumerate(rb210):
		peaks = find_peaks(dft['Voltage'],height=0.0)
		sps = len(peaks[0])/(dft['time'].iloc[-1]-dft['time'].iloc[0])
		num_spikes.append(sps)

	return np.mean(num_spikes)

def CV(df):
	rb210 = rb2x10_df(df)
	ISIs = []
	for i,dft in enumerate(rb210):
		peaks =find_peaks(dft['Voltage'],height=0.0)[0]
		peaks+= dft.index[0]
		try:
			for i,peak in enumerate(peaks):
				if i>0:
					ISI = (dft['time'][peaks[i]]-dft['time'][peaks[i-1]])*1000  
					ISIs.append(ISI)
		except:
			pass
	if len(ISIs)>0:
		CV = np.std(ISIs)/np.mean(ISIs)
	else:
		CV = 0.0
	return CV

'''
Functions called on strong hyperolarization peroid.
'''
def Ih(df):
	dft = Ih_df(df)
	sag_idx = find_peaks(-dft['Voltage'].loc[:515000],distance = 10000,width=100,prominence=3.0)[0]
	sag_idx += dft.index[0]
	sag = dft['Voltage'].loc[sag_idx[0]]
	ss_v = np.mean(dft['Voltage'].loc[510000:511000])
	Ih = sag - ss_v

	return Ih

'''
Functions called on the 4x rheobase firing period.
'''
def ISIs(df):

	R4 = rheobasex4_df(df)
	peaks = find_peaks(R4['Voltage'],height=2.0,distance=20.0,width=5.0,prominence=10.0)
	pps = len(peaks[0])/(R4['time'].iloc[-1]-R4['time'].iloc[0])

	if len(peaks[0])<3:
		R2 = rheobasex2_df(df)
		peaks = find_peaks(R2['Voltage'],height=2.0,distance=20.0,width=5.0,prominence=10.0)
		pps = len(peaks[0])/(R2['time'].iloc[-1]-R2['time'].iloc[0])
	ISIs = []

	for i,peak in enumerate(peaks[0]):
		if i<(len(peaks[0])-1):
			ISI = (peaks[0][i+1] - peaks[0][i])/10
			ISIs.append(ISI)

	max_ISI = max(ISIs)
	min_ISI = min(ISIs)
	mean_ISI = np.mean(ISIs)
	median_ISI = np.median(ISIs)

	return ISIs,max_ISI,min_ISI,mean_ISI,median_ISI,pps

def ISI_stats(df):
	ISI = ISIs(df)[0]
	avg_ISI_1_3 = np.mean(ISI[:3])
	std_ISI_1_3 = np.std(ISI[:3])

	return avg_ISI_1_3, std_ISI_1_3

def ISI_1_2(df):
	try:
		diff12 = ISIs(df)[0][1] - ISIs(df)[0][0]
	except:
		diff12 = 0.0
	return diff12

def ISI_1_last(df):
	try:
		diff1last = ISIs(df)[0][-1] - ISIs(df)[0][0]
	except:
		diff1last = 0.0
	return diff1last

def max_FR(df):
	maxFR = 1/(ISIs(df)[2]/1000)

	return maxFR

def min_FR(df):
	minFR = 1/(ISIs(df)[1]/1000)

	return minFR

def avg_FR(df):
	meanFR = 1/(ISIs(df)[3]/1000)

	return meanFR

def FR_adaptation(df):
	R4 = rheobasex4_df(df)
	peaks = find_peaks(R4['Voltage'],height=2.0,distance=20.0,width=5.0,prominence=10.0)
	ISIs = []
	peaks100 = []
	peaks400 = []
	for peak in peaks[0]:
		if peak<1000:
			peaks100.append(peak)
		if peak>4000:
			peaks400.append(peak)

	if len(peaks100)<=0:
		R2 = rheobasex2_df(df)
		peaks100 = find_peaks(R2['Voltage'][0:1000], height=2.0,distance=20.0,width=5.0,prominence=10.0)
		peaks400 = find_peaks(R2['Voltage'][len(R2['Voltage'])-4000:], height=2.0,distance=20.0,width=5.0,prominence=10.0)
	FRadapt = 100*((len(peaks100) - len(peaks400))/len(peaks100))
	return FRadapt
	
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''
Param_df creation calls all analysis functions and concatenates them into a pandas DataFrame of analyzed variables.
'''
def make_param_df(df):

	Parameter_dict ={'Recording_Date':get_date(df),
                 'Resting_Membrane_P':Resting_Potential(df),
                 'Baseline_shift':Baselineshift(df),
                 'Threshold':Threshold(df),
                 'Rheobase':Rheobase(df),
                 'AP1-2_drop':AP12_drop(df),
                 'AP1-last_drop':APfl_drop(df),
                 'AP2-last_drop':AP2l_drop(df),
                 'Max_AP_change':Max_AP_change(df),
                 'AP1_amp':AP1_amp(df),
                 'AP1_abs_amp':AP1_abs_amp(df),
                 # 'Onset_Voltage':AP1_amp(df)-AP1_abs_amp(df),
                 'AP_duration':AP_duration(df),
                 'AP_1/2_Width':AP_half_Width(df),
                 'Rise_time':Rise_time(df),
                 'Fall_time':Fall_time(df),
                 'Rise_rate':Rise_rate(df),
                 'Fall_rate':Fall_rate(df),
                 '10-90%_rise_time':ten_ninty_rise_time(df),
                 'fAHP_min_voltage':Refractory_period(df)[0],
                 'fAHP':Refractory_period(df)[1],
                 'sAHP_min_voltage':sAHP(df)[0],
                 'sAHP':sAHP(df)[1],
                 'ADP':Refractory_period(df)[3],
                 # 'Rheobase_Refractory_period':Refractory_period(df)[2],
                 'Tau':tau(df),
                 'Delay_to_Spike':delay_to_spike(df),
                 'Delay_to_fAHP':delay_to_fAHP(df),
                 'ISIs': ISIs(df)[0],
                 'Max_ISI':ISIs(df)[1],
                 'Min_ISI':ISIs(df)[2],
                 'Mean_ISI':ISIs(df)[3],
                 'Median_ISI':ISIs(df)[4],
				 'Avg_AP1_delay': AP1_delay(df)[0],
				 'STD_AP1_delay': AP1_delay(df)[1],
				 'Avg_AP2_delay': AP2_delay(df)[0],
				 'STD_AP2_delay': AP2_delay(df)[1],	
				 'Avg_Intl_Accom_ratio':ISI_accom_ratio(df)[0],
				 'STD_Intl_Accom_ratio':ISI_accom_ratio(df)[1],
				 'Avg_SS_Accom_ratio':ISI_accom_ratio(df)[2],
				 'STD_SS_Accom_ratio':ISI_accom_ratio(df)[3],
				 'Avg_Intl_Accom_diff':ISI_accom_diff(df)[0],
				 'STD_Intl_Accom_diff':ISI_accom_diff(df)[1],
				 'Avg_SS_Accom_diff':ISI_accom_diff(df)[2],
				 'STD_SS_Accom_diff':ISI_accom_diff(df)[3],
				 'Avg_2x_ISI_1_3': rb2x_ISI_1_3(df)[0],
				 'STD_2x_ISI_1_3': rb2x_ISI_1_3(df)[1],
				 'CV':CV(df),
                 'Spikes/sec':Avg_spike_rate(df),
                 'Ih':Ih(df),
                 'ISI_1_2': ISI_1_2(df),
                 'ISI_1_last': ISI_1_last(df),
                 'Avg_4x_ISI_1_3':ISI_stats(df)[0],
                 'STD_4x_ISI_1_3':ISI_stats(df)[1],
                 # 'Max_FR': max_FR(df),
                 # 'Min_FR': min_FR(df),
                 # 'Mean_FR': avg_FR(df),
                 'FR_adaptation': FR_adaptation(df),
                 'Input_Resistance':input_resistance(df),
}

	df_params = pd.DataFrame.from_dict(Parameter_dict,orient='index',columns = df.mouseID.unique()+': '+df.Cell.unique())

	return(df_params)
