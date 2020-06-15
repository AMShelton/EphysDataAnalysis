#For reading and analyzing .paq files in python because Matlab is stupid

import os
import sys
import pyabf
import paq2py
import numpy as np
import pandas as pd
import seaborn as sns
import DoC_tools as dt
import scipy.interpolate
from ams_utilities import *
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import visual_behavior.utilities as vbu

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

def Resting_Potential(df):
	Resting_p = np.mean(df['Voltage'][0:5000])

	return Resting_p

def Baselineshift(df):
	start = np.mean(df['Voltage'][0:5000])
	end = np.mean(df['Voltage'][(len(df['Voltage']) - 5000):])
	Baselineshift = end - start

	return Baselineshift

def Rheobase(df):
	rheobase = np.mean(df['Current'][282000:284000]) - np.mean(df['Current'][0:5000])

	return rheobase 

def find_spikes_range(df):
	AP_range = df['Voltage'][294000:311000]

	return AP_range

def AP12_drop(df):
	AP_range  = find_spikes_range(df)
	APs = find_peaks(AP_range,height=0.0)[1]
	AP12_drop = APs['peak_heights'][0] - APs['peak_heights'][1]

	return AP12_drop

def APfl_drop(df):
	AP_range  = find_spikes_range(df)
	APs = find_peaks(AP_range,height=0.0)[1]
	APfl_drop = APs['peak_heights'][0] - APs['peak_heights'][len(APs['peak_heights'])-1]

	return APfl_drop

def AP2l_drop(df):
	AP_range  = find_spikes_range(df)
	APs = find_peaks(AP_range,height=0.0)[1]
	AP2l_drop = APs['peak_heights'][1] - APs['peak_heights'][len(APs['peak_heights'])-1]

	return AP2l_drop

def Max_AP_change(df):
	AP_range  = find_spikes_range(df)
	APs = find_peaks(AP_range,height=0.0)[1]

	diffs = []

	for i,value in enumerate(APs['peak_heights']):
	    diff = abs(APs['peak_heights'][len(APs['peak_heights'])-1] - value)
	    diffs.append(diff)
	    
	max_diff = max(diffs)

	return max_diff


def AP1_amp(df):
	SFR = rheobase_df(df)
	AP1 = find_peaks(SFR['Voltage'],height=0.0)[1]['peak_heights'][0]

	return AP1

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

def get_onset_offset_index(df=None):

	onset_index = []
	v_diffs, index = get_v_diffs(df=df)
        
	for ii,rate in enumerate(v_diffs):
	    if rate>=2.0:
	        onset_idx = index[ii]
	        onset_index.append(onset_idx)

	onset_voltage = df['Voltage'][onset_index[0]]
	AP_v = (df['Voltage'].loc[onset_index[0]:])
	# offset_index = np.where(np.logical_and(AP_v>=onset_voltage-4, AP_v<=onset_voltage+4))[0][1] + onset_index[0]
	offset_index  = np.where(AP_v<onset_voltage)[0][0] + onset_index[0]

	return onset_index[0], offset_index

def AP_duration(df):

	SFR = rheobase_df(df)
	        
	onset_index, offset_index = get_onset_offset_index(df=SFR)
	onset_time = df['time'][onset_index]
	offset_time = SFR['time'].loc[offset_index]
	AP_duration = 1000*(offset_time - onset_time) 

	return AP_duration

def AP1_abs_amp(df):

	SFR = rheobase_df(df)
	onset_index, offset_index = get_onset_offset_index(df=SFR)
	AP_amp = AP1_amp(df) + abs(SFR['Voltage'].loc[onset_index])

	return AP_amp

def AP_half_Width(df):

	SFR = rheobase_df(df)

	AP_amp = AP1_abs_amp(df)
	AP1 = AP1_amp(df)
	AP_hw_amp = AP1 - AP_amp/2
	onset_index, offset_index = get_onset_offset_index(df=SFR)
	peak_idx = find_peaks(SFR['Voltage'],height=0.0)[0][0] + SFR.index[0]

	spike_range = df[onset_index:offset_index]

	hw_idx_1 = spike_range['Voltage'][(spike_range['Voltage']>=(AP1 - AP_amp/2))].index[0]
	hw_idx_2 = spike_range['Voltage'][(spike_range['Voltage'].loc[peak_idx:].loc[peak_idx:]<=AP_hw_amp+2)
                                  &(spike_range['Voltage']>=AP_hw_amp-8)].index[0]
	hw_time_1 = spike_range['time'].loc[hw_idx_1]
	hw_time_2 = spike_range['time'].loc[hw_idx_2]
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
	onset_voltage = onset_voltage = df['Voltage'][onset_index]
	AP_amp =  AP1_abs_amp(df)
	AP1 = AP1_amp(df)

	ten_ninty_idx = np.where(np.logical_and(SFR['Voltage']>=onset_voltage+(0.1*AP_amp), 
                                    SFR['Voltage']<=0.9*AP_amp))

	ninty_time = SFR['time'][SFR['Voltage']>=(AP1)*0.9].index[0]/(SFR['time'].index[0]/np.array(SFR['time'])[0])

	ten_time = SFR['time'][SFR['Voltage']>=(onset_voltage+(AP_amp*0.1))].index[0]/(SFR['time'].index[0]/np.array(SFR['time'])[0])

	ten_ninety_rt = (ninty_time - ten_time)*1000

	return ten_ninety_rt

def rheobase_df(df):
	rb_df = df[280050:285100]

	return rb_df

def rheobasex2_df(df):
	R2_df = df[294000:311000]

	return R2_df

def rheobasex4_df(df):
	R4_df = df[567000:572000]

	return R4_df

def Refractory_period(df):

	rb_df = rheobase_df(df)

	rb_v_diffs, rb_index = get_v_diffs(df=rb_df)
	rb_onset_index, rb_offset_index = get_onset_offset_index(df=rb_df)

	rb_onset_voltage = rb_df['Voltage'][rb_onset_index]       
	rb_onset_time = rb_df['time'][rb_onset_index]
	rb_offset_voltage = rb_df['Voltage'].loc[rb_offset_index]
	rb_offset_v = (rb_df['Voltage'].loc[rb_offset_index:])

	rebound_array = [i for i in rb_offset_v if i > rb_onset_voltage+2]

	if len(rebound_array)>1:

	    rb_rebound_index = np.where(rb_offset_v > rb_onset_voltage+2)[0][0] + rb_offset_index
	    rb_offset_time = rb_df['time'].loc[rb_offset_index]
	    rb_rebound_time = rb_df['time'].loc[rb_rebound_index]
	    rb_Refractory_period = 1000*(rb_rebound_time - rb_offset_time)

	else:
	    rb_rebound_index = np.where(np.diff(rb_df['Current'].loc[rb_offset_index:])==min(np.diff(rb_df['Current'].loc[rb_offset_index:])))[0][0] + rb_offset_index
	    rb_Refractory_period = 'N/A'

	sAHP_min = min(rb_df['Voltage'].loc[int(rb_offset_index):rb_rebound_index])
	sAHP = rb_onset_voltage - sAHP_min

	node = find_peaks(rb_df['Voltage'].loc[int(rb_offset_index):rb_rebound_index],
	                      distance=20,
	                      height=sAHP_min+1.0,
	                      threshold=sAHP-100,
	                      width=25)

	if len(node[0]>0):
	    node_idx = node[0][0] + rb_offset_index
	    fAHP_min = min(rb_df['Voltage'].loc[int(rb_offset_index):node_idx])
	    fAHP = rb_onset_voltage - fAHP_min
	    
	    if fAHP_min == sAHP_min:
	        sAHP_min = min(rb_df['Voltage'].loc[node_idx:rb_rebound_index])
	        sAHP = rb_onset_voltage - sAHP_min   
	else:#
	    node = np.nan
	    node_idx = np.nan
	    fAHP_min = sAHP_min
	    fAHP = sAHP   
	    sAHP_min = sAHP_min
	    sAHP = sAHP

	return fAHP_min,fAHP,sAHP_min,sAHP,node,node_idx

def input_resistance(df):

	# v1 = np.mean(df['Voltage'].loc[15000:20000])
	# v2 = np.mean(df['Voltage'].loc[35000:40000])
	v3 = np.mean(df['Voltage'].loc[55000:60000])
	v4 = np.mean(df['Voltage'].loc[75000:80000])
	v5 = np.mean(df['Voltage'].loc[95000:100000])
	v6 = np.mean(df['Voltage'].loc[115000:120000])
	v7 = np.mean(df['Voltage'].loc[135000:140000])
	v8 = np.mean(df['Voltage'].loc[155000:160000])
	v9 = np.mean(df['Voltage'].loc[175000:180000])
	v10 = np.mean(df['Voltage'].loc[195000:200000])
	v11 = np.mean(df['Voltage'].loc[215000:220000])

	# i1 = np.mean(df['Current'].loc[15000:20000])
	# i2 = np.mean(df['Current'].loc[35000:40000])
	i3 = np.mean(df['Current'].loc[55000:60000])
	i4 = np.mean(df['Current'].loc[75000:80000])
	i5 = np.mean(df['Current'].loc[95000:100000])
	i6 = np.mean(df['Current'].loc[115000:120000])
	i7 = np.mean(df['Current'].loc[135000:140000])
	i8 = np.mean(df['Current'].loc[155000:160000])
	i9 = np.mean(df['Current'].loc[175000:180000])
	i10 = np.mean(df['Current'].loc[195000:200000])
	i11 = np.mean(df['Current'].loc[215000:220000])
	x = [i3,i4,i5,i6,i7,i8,i9,i10,i11]
	y = [v3,v4,v5,v6,v7,v8,v9,v10,v11]

	#i1,i2,
	#v1,v2,
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

	tau = popt[1]*1000

	return tau

def delay_to_spike(df):
# time from curren pulse onset to first action potential
	df_delay = rheobase_df(df)
	onset_index, offset_index = get_onset_offset_index(df=df_delay)

	c_diffs, index = get_c_diffs(df=df_delay)
	c_baseline = np.mean(df['Current'][0:5000])
	c_baseline_std = np.std(df['Current'][0:5000])
	c_rise = []
	c_rise_index = []
	for i,c_diff in enumerate(c_diffs):
		if c_diff>c_baseline_std*2:
			c_rise.append(c_diff)
			c_rise_index.append(index[i])

	delay = (df['time'][onset_index] - df['time'][c_rise_index[0]])*1000

	return delay

def ISIs(df):

	R4 = rheobasex4_df(df)

	peaks = find_peaks(R4['Voltage'], height = 0.0,distance=50)

	if len(peaks[0])<3:
		R2 = rheobasex2_df(df)

		peaks = find_peaks(R2['Voltage'], height = 0.0,distance=50)

	ISIs = []

	for i,peak in enumerate(peaks[0]):
		if i<(len(peaks[0])-1):
			ISI = (peaks[0][i+1] - peaks[0][i])/10
			ISIs.append(ISI)

	max_ISI = max(ISIs)
	min_ISI = min(ISIs)
	mean_ISI = np.mean(ISIs)

	return ISIs,max_ISI,min_ISI,mean_ISI

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
	peaks = find_peaks(R4['Voltage'], height = 5.0,distance=50)
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
		peaks100 = find_peaks(R2['Voltage'][0:1000], height = 5.0,distance=50)
		peaks400 = find_peaks(R2['Voltage'][len(R2['Voltage'])-4000:], height = 5.0,distance=50)
	FRadapt = 100*((len(peaks100) - len(peaks400))/len(peaks100))
	return FRadapt

def make_param_df(df):

	Parameter_dict ={'Recording_Date':get_date(df),
                 'Resting_Membrane_P':Resting_Potential(df),
                 'Basline_shift':Baselineshift(df),
                 'Rheobase':Rheobase(df),
                 'AP1-2_drop':AP12_drop(df),
                 'AP1-last_drop':APfl_drop(df),
                 'AP2-last_drop':AP2l_drop(df),
                 'Max_AP_change':Max_AP_change(df),
                 'AP1_amp':AP1_amp(df),
                 'AP1_abs_amp':AP1_abs_amp(df),
                 'Onset_Voltage':AP1_amp(df)-AP1_abs_amp(df),
                 'AP_duration':AP_duration(df),
                 'AP_1/2_Width':AP_half_Width(df),
                 'Rise_time':Rise_time(df),
                 'Fall_time':Fall_time(df),
                 'Rise_rate':Rise_rate(df),
                 'Fall_rate':Fall_rate(df),
                 '10-90%_rise_time':ten_ninty_rise_time(df),
                 'fAHP_min_voltage':Refractory_period(df)[0],
                 'fAHP':Refractory_period(df)[1],
                 'sAHP_min_voltage':Refractory_period(df)[2],
                 'sAHP':Refractory_period(df)[3],
                 # 'Rheobase_Refractory_period':Refractory_period(df)[2],
                 'Tau':tau(df),
                 'Delay_to_Spike':delay_to_spike(df),
                 'ISIs': ISIs(df)[0],
                 'Max_ISI':ISIs(df)[1],
                 'Min_ISI':ISIs(df)[2],
                 'Mean_ISI':ISIs(df)[3],
                 'ISI_1_2': ISI_1_2(df),
                 'ISI_1_last': ISI_1_last(df),
                 'Max_FR': max_FR(df),
                 'Min_FR': min_FR(df),
                 # 'Mean_FR': avg_FR(df),
                 'FR_adaptation': FR_adaptation(df),
                 'Input_Resistance':input_resistance(df),
}

	df_params = pd.DataFrame.from_dict(Parameter_dict,orient='index',columns = df.mouseID.unique()+': '+df.Cell.unique())

	return(df_params)

def raw_df_convert(raw_df,lines_to_drop=None):
    
    datab = raw_df.drop(lines_to_drop, axis=0)
    datab.dropna(inplace = True)
    datab = datab.reset_index(drop=True)
    cols = datab['metric'].values
    # col

    # Convert to float array and standardise data ((x - mean)/std)
    datab = str_flt(datab.iloc[:,1:])
    datab -= np.mean(datab)
    datab /= np.std(datab)

    # Transpose array so variables arranged column-wise 
    data = datab.T
    data.columns = cols
    data.columns.names = ['metric']
    data.index.names = ["cell"]
    data
    
    return data
