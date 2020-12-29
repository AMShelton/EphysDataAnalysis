import os
import numpy as np
import pandas as pd
from ams_paq_utilities import *
from ams_utilities import *
import matplotlib.pyplot as plt
import seaborn as sns
import tifffile as tf
from datetime import datetime

sns.set()
sns.set_style('white')
params = {'legend.fontsize': 'x-large',
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
plt.rcParams.update(params)

def placeAxesOnGrid(fig,dim=[1,1],xspan=[0,1],yspan=[0,1],wspace=None,hspace=None,sharex=False,sharey=False,
                    frameon=True):
    '''
    Takes a figure with a gridspec defined and places an array of sub-axes on a portion of the gridspec

    Takes as arguments:
        fig: figure handle - required
        dim: number of rows and columns in the subaxes - defaults to 1x1
        xspan: fraction of figure that the subaxes subtends in the x-direction (0 = left edge, 1 = right edge)
        yspan: fraction of figure that the subaxes subtends in the y-direction (0 = top edge, 1 = bottom edge)
        wspace and hspace: white space between subaxes in vertical and horizontal directions, respectively

    returns:
        subaxes handles
    '''
    import matplotlib.gridspec as gridspec

    outer_grid = gridspec.GridSpec(100,100)
    inner_grid = gridspec.GridSpecFromSubplotSpec(dim[0],dim[1],
                                                  subplot_spec=outer_grid[int(100*yspan[0]):int(100*yspan[1]),int(100*xspan[0]):int(100*xspan[1])],
                                                  wspace=wspace, hspace=hspace)

    #NOTE: A cleaner way to do this is with list comprehension:
    # inner_ax = [[0 for ii in range(dim[1])] for ii in range(dim[0])]
    inner_ax = dim[0]*[dim[1]*[fig]] #filling the list with figure objects prevents an error when it they are later replaced by axis handles
    inner_ax = np.array(inner_ax)
    idx = 0
    for row in range(dim[0]):
        for col in range(dim[1]):
            if row > 0 and sharex == True:
                share_x_with = inner_ax[0][col]
            else:
                share_x_with = None

            if col > 0 and sharey == True:
                share_y_with = inner_ax[row][0]
            else:
                share_y_with = None

            inner_ax[row][col] = plt.Subplot(fig,
                                            inner_grid[idx],
                                            sharex=share_x_with,
                                            sharey=share_y_with,
                                            frameon=frameon,
                                            )

            if row == dim[0]-1 and sharex == True:
                inner_ax[row][col].xaxis.set_ticks_position('bottom')
            elif row < dim[0] and sharex == True:
                plt.setp(inner_ax[row][col].get_xticklabels(), visible=True)

            if col == 0 and sharey == True:
                inner_ax[row][col].yaxis.set_ticks_position('left')
            elif col > 0 and sharey == True:
                plt.setp(inner_ax[row][col].get_yticklabels(), visible=False)

            fig.add_subplot(inner_ax[row,col])
            idx += 1

    inner_ax = np.array(inner_ax).squeeze().tolist() #remove redundant dimension
    return inner_ax

def make_ephys_df(datapath):
    
    data = p2p(datapath)
    df = pd.DataFrame()

    df['Voltage'] = data['data'][0]
    df['Current'] = data['data'][1]
    df['mouseID'] = datapath.split('_')[3]
    df['time'] = df.index/data['rate']
    df['Cell#'] = datapath.split('_')[10]

    return df

def make_opto_df(datapath):
    data = p2p(datapath)

    df = pd.DataFrame()

    df['Membrane Voltage (mV)'] = data['data'][0]
    df['Current (pA)'] = data['data'][1]
    df['Blue Light (V)'] = data['data'][2]
    df['Orange Light (V)'] = data['data'][3]
    df['opsin type'] = datapath.split('_')[5]
    df['experiment'] = datapath.split('_')[11]
    df['mouseID'] = datapath.split('_')[3]
    df['Cell#'] = datapath.split('_')[10]
    df['time'] = df.index/data['rate']

    
    return df

def make_info_table(mouseID,cell,mice_df,cell_df,ax):
    '''
    generates a table with info extracted from the dataframe
    adapted by AMS on 2020-04-08 from DRO - 10/13/16
    '''
    idx_cell = cell_df[(cell_df['Pseudo ID']==mouseID)&(cell_df['Cell#']==cell)].index[0]
    idx_mouse = mice_df[mice_df['Pseudo ID']==mouseID].index[0]
    
    #I'm using a list of lists instead of a dictionary so that it maintains order
    #the second entries are in quotes so they can be evaluated below in a try/except
    from matplotlib.font_manager import FontProperties
    data = [
            # from all_cells['recording date'] 
            ['Date',"cell_df['recording date']{}".format('.loc[idx_cell]')],
            # from all_cells['Pseudo ID']
            ['Mouse ID',"cell_df['Pseudo ID']{}".format('.loc[idx_cell]')],
            # from all_cells['Strain']
            ['Strain',"cell_df['Strain']{}".format('.loc[idx_cell]')],
            # from all_cells['Slice#']
            ['Slice',"cell_df['Slice#']{}".format('.loc[idx_cell]')],
            # from all_cells['Cell#']
            ['Cell',"cell_df['Cell#']{}".format('.loc[idx_cell]')],
            # from all_mice['Virus#']
            ['Injection 1',"mice_df['Virus1']{}".format('.loc[idx_mouse]')],
            ['Injection 2',"mice_df['Virus2']{}".format('.loc[idx_mouse]')],
            # from all_mice[Surgery Notes]
            ['Injection Locations',"mice_df['Surgery Notes']{}".format('.loc[idx_mouse]')],
            # from all_mice['ExpressionTime1']
            ['Days of Expression',"mice_df['Expression Time1']{}".format('.loc[idx_mouse]')],
            # from all_cells['Add. Notes']
            ['Notes',"cell_df['Add. Notes']{}".format('.loc[idx_cell]')]]


    cell_text = []
    for x in data:
        try:
            cell_text.append([eval(x[1])])
        except:
            cell_text.append([np.nan])

    #define row colors
    row_colors = [['lightgray'],['white']]*(len(data))

    #make the table
    table = ax.table(cellText=cell_text,
                          rowLabels=[x[0] for x in data],
                          rowColours=vbu.flatten_list(row_colors)[:len(data)],
                          colLabels=None,
                          loc='center',
                          cellLoc='left',
                          rowLoc='right',
                          cellColours=row_colors[:len(data)])
    for (row, col), cell in table.get_celld().items():
        if (col == -1):
            cell.set_text_props(fontproperties=FontProperties(weight='bold'))
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    ax.grid(False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    sns.despine(left=True,bottom=True)
    #do some cell resizing
    cell_dict=table.get_celld()
    for cell in cell_dict:
        if cell[1] == -1:
            cell_dict[cell].set_width(0.25)
            cell_dict[cell].set_height(0.1)
        if cell[1] == 0:
            cell_dict[cell].set_width(0.7)
            cell_dict[cell].set_height(0.1)

def get_images(mouseID,cell,cell_df):
    
    cell_idx = cell_df[(cell_df['Pseudo ID']==mouseID)&(cell_df['Cell#']==cell)].index[0]
    
    imgs = []
    for filename in os.listdir(cell_df['Ephys Path'].loc[cell_idx]):
        if '10X_470nm' in filename:
            imgs.append(os.path.join(cell_df['Ephys Path'].loc[cell_idx],filename))
        elif '10X_595nm' in filename:
            imgs.append(os.path.join(cell_df['Ephys Path'].loc[cell_idx],filename))

        elif '10X_DIC' in filename:
            imgs.append(os.path.join(cell_df['Ephys Path'].loc[cell_idx],filename))
        elif '40X_DIC' in filename:
            imgs.append(os.path.join(cell_df['Ephys Path'].loc[cell_idx],filename))
        else:
            pass

    images = [tf.imread(img) for i,img in enumerate(imgs)]
    
    return images

def make_intrinsic_fig(mouseID,cell,cell_df,ax):
    
    folder = cell_df[(cell_df['Pseudo ID']==mouseID)&(cell_df['Cell#']==cell)]['Ephys Path'].unique()[0]
    for filename in os.listdir(folder):
        if 'allparams.paq' in filename:
            datapath = os.path.join(folder,filename)
            
            in_df = make_ephys_df(datapath) 
            ax.set_ylabel('Vm (mV)',x=-0.1)
            ax.set_xlabel('Recording time (s)')
            ax.plot(in_df['time'][275000:315000].values,
                            in_df['Voltage'][275000:315000].values,
                            color='k',
                            lw=0.5)

def make_powercheck_fig(mouseID,cell,cell_df,ax):
    
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    datapaths = []
    folder = cell_df[(cell_df['Pseudo ID']==mouseID)&(cell_df['Cell#']==cell)]['Ephys Path'].unique()[0]
    
    datapaths = [os.path.join(folder,filename) for filename in os.listdir(folder) if 'powercheck.paq' in filename]
    
    if len(datapaths)>1:
        df_all = {}
        for i,datapath in enumerate(datapaths): 
        
            dft = make_opto_df(datapath)

            df_all[''.join(dft.mouseID.unique()+'_'+dft['Cell#'].unique()+'_'+dft['experiment'].unique())] = dft
        
        df = pd.concat(df_all)
        
        df_o = df[df['experiment']=='orange']
        df_b = df[df['experiment']=='blue']

        # get response windows for all events. Response window (rw) is the 5ms before and the 50ms after a light onset

        blue_stims = find_pulse(df_b['Blue Light (V)'],std=10)

        b_response_windows = []

        for i,stim in enumerate(blue_stims):
            response_window = df_b['Membrane Voltage (mV)'].iloc[stim[0]-100:stim[79]+1000].values
            b_response_windows.append(response_window)    

        b_avg_rw = np.mean(b_response_windows,axis=0)

        orange_stims = find_pulse(df_o['Orange Light (V)'],std=10)

        o_response_windows = []

        for ii,stim in enumerate(orange_stims):
            response_window = df_o['Membrane Voltage (mV)'].iloc[stim[0]-100:stim[79]+1000].values
            o_response_windows.append(response_window)    

        # get timeseries windows for actual light pulses for plotting in smaller window (artistically nice, not explicitly necessary)

        b_stims = []

        for i,stim in enumerate(blue_stims):
            response_window = df_b['Blue Light (V)'].iloc[stim[0]-100:stim[79]+100].values
            b_stims.append(response_window)

        o_stims = []

        for i,stim in enumerate(orange_stims):
            response_window = df_o['Orange Light (V)'].iloc[stim[0]-100:stim[79]+100].values
            o_stims.append(response_window)


        # make and array for time that is based on the length of the averaged response window (for plotting purposes only)

        t = []

        for i in np.arange(len(b_avg_rw)):
            j = (i-100)/20000
            t.append(j)


        # plot the powercheck figure

        b_colors = sns.color_palette('Blues',15)
        o_colors = sns.color_palette('Reds',15)

        for i,rw in enumerate(o_response_windows[::-1]):
            ax[0].plot(t,rw,lw=0.75,c=o_colors[i])

        for ii,rw in enumerate(b_response_windows[::-1]):
            ax[1].plot(t,rw,lw=0.75,c=b_colors[ii])

        ax[0].axvspan(0,0.004,-100,100,color='orange',alpha = 0.5)
        ax[1].axvspan(0,0.004,-100,100,color='dodgerblue',alpha = 0.5)
        ax[1].set_xlabel('t(s) from stim onset',x=-0.0)    
        ax[0].set_ylabel('Vm (mV)')
        sns.despine()


        ax2 = inset_axes(ax[0],width=0.75,height=1.)
        for iii,stim in enumerate(o_stims[::-1]):
            ax2.plot(stim,color=o_colors[iii])
        ax2.set_xticks([])
        ax2.set_ylabel('LED PWR (mW)',rotation = 90,fontsize=10)

        ax3 = inset_axes(ax[1],width=0.75,height=1.)
        for iii,stim in enumerate(b_stims[::-1]):
            ax3.plot(stim,color=b_colors[iii])
        ax3.set_xticks([])
        ax3.set_ylabel('LED PWR (mW)',rotation = 90,fontsize=10)

def make_hooks_fig(mouseID,cell,cell_df,ax):
    
    folder = cell_df[(cell_df['Pseudo ID']==mouseID)&(cell_df['Cell#']==cell)]['Ephys Path'].unique()[0]
    for filename in os.listdir(folder):
        if 'b+o_interleaved_hooks.paq' in filename:
            datapath = os.path.join(folder,filename)

            df = make_opto_df(datapath)

            o_stims = find_pulse(df['Orange Light (V)'],std=10)

            b_stims = find_pulse(df['Blue Light (V)'],std=10)

            o_response_windows1 = []
            o_response_windows2 = []
            o_response_windows3 = []
            b_response_windows1 = []
            b_response_windows2 = []
            b_response_windows3 = []

            for i,stim in enumerate(o_stims[0::3]):
                response_window = df['Membrane Voltage (mV)'].iloc[stim[0]-5000:stim[79]+20000].values
                o_response_windows1.append(response_window)

            for i,stim in enumerate(o_stims[1::3]):
                response_window = df['Membrane Voltage (mV)'].iloc[stim[0]-5000:stim[79]+20000].values
                o_response_windows2.append(response_window)

            for i,stim in enumerate(o_stims[2::3]):
                response_window = df['Membrane Voltage (mV)'].iloc[stim[0]-5000:stim[79]+20000].values
                o_response_windows3.append(response_window)


            for i,stim in enumerate(b_stims[0::3]):
                response_window = df['Membrane Voltage (mV)'].iloc[stim[0]-5000:stim[79]+20000].values
                b_response_windows1.append(response_window)

            for i,stim in enumerate(b_stims[1::3]):
                response_window = df['Membrane Voltage (mV)'].iloc[stim[0]-5000:stim[79]+20000].values
                b_response_windows2.append(response_window)

            for i,stim in enumerate(b_stims[2::3]):
                response_window = df['Membrane Voltage (mV)'].iloc[stim[0]-5000:stim[79]+20000].values
                b_response_windows3.append(response_window)

            o_avg_rw1 = np.mean(o_response_windows1,axis=0)
            o_avg_rw2 = np.mean(o_response_windows2,axis=0)
            o_avg_rw3 = np.mean(o_response_windows3,axis=0)
            b_avg_rw1 = np.mean(b_response_windows1,axis=0)
            b_avg_rw2 = np.mean(b_response_windows2,axis=0)
            b_avg_rw3 = np.mean(b_response_windows3,axis=0)

            t = []

            for i in np.arange(len(o_avg_rw1)):
                j = (i - 5000)/20000
                t.append(j)

            sns.despine(ax=ax[0])
            sns.despine(ax=ax[1],left=True)
            sns.despine(ax=ax[2],left=True)
            sns.despine(ax=ax[3],left=True)

            #########################################################################################################    

            ax[0].axvspan(0,0.004,-100,100,color='orange',alpha=0.5,label = '590nm LED')

            for rw in o_response_windows1:
                ax[0].plot(t,rw,color='grey',alpha=0.4,lw=0.5)

            ax[0].plot(t,o_avg_rw1,color='k')
            ax[0].set_ylabel('Vm (mV)')
            ax[0].set_title('Orange Only')

            # ##########################################################################################################

            ax[1].axvspan(0,0.004,-100,100,color='dodgerblue',alpha=0.5,label = '470nm LED')
            for rw in b_response_windows1:
                ax[1].plot(t,rw,color='grey',alpha=0.4,lw=0.5)

            ax[1].plot(t,b_avg_rw1,color='k')
            ax[1].set_title('Blue Only')

            # ###########################################################################################################

            ax[2].axvspan(0,0.50,1,-100,color='orange',alpha=0.5,label = '590nm LED (1.86 mW)')

            ax[2].axvspan(0.5,0.504,-100,100,color='dodgerblue',label = '470nm LED (0.6 mW)')

            for rw in o_response_windows2:
            #     if max(rw)<0.0:
                    ax[2].plot(t,rw,color='grey',alpha=0.4,lw=0.5)
            ax[2].plot(t,o_avg_rw2,color='k')
            ax[2].set_title("Hooks'")
            ax[2].set_xlabel('t(s) from stim onset',x=-0.0)

            # ###########################################################################################################

            ax[3].axvspan(0,0.004,-100,100,color='purple',alpha=0.5,label = 'Both LEDs')
            for rw in o_response_windows3:
            #     if max(rw)<0.0:
                    ax[3].plot(t,rw,color='grey',alpha=0.4,lw=0.5)
            ax[3].plot(t,o_avg_rw3,color='k')
            ax[3].set_title('Dual Stim')
            

def make_summary_fig(mouseID=None,cell=None,cell_idx=None,cell_df=None,mice_df=None,save_path=None,fig_name=None,save=False):
    try:
        
        fig=plt.figure(figsize=(12,12),facecolor='w')
        # make axes
        ax_hooks = placeAxesOnGrid(fig,dim=(1,4),xspan=(0,1),yspan=(0.70,1),sharey=True,sharex=True, wspace=0.1)
        ax_pwchk = placeAxesOnGrid(fig,dim=(1,2),xspan=(0.0,1),yspan=(0.35,0.6),sharey=True,sharex=True, wspace=0.2)
        ax_table = placeAxesOnGrid(fig,xspan=(0.05,0.35),yspan=(0,0.25),frameon=False)
        ax_image = placeAxesOnGrid(fig,dim=(2,2),xspan=(0.225,0.725),yspan=(0.0,0.3),
                                        sharey=False,sharex=False,hspace=0.0,wspace=-0.62,frameon=False)
        ax_in = placeAxesOnGrid(fig,dim=(1,1),xspan=(0.7,1),yspan=(0.0,0.25))

        make_info_table(mouseID,cell,mice_df,cell_df,ax_table)

        images = get_images(mouseID,cell,cell_df)    
        img_titles = ['40X DIC','10X 470nm','10X 595nm', '10X DIC']

        for i,axis in enumerate(ax_image[0]):
            ax_image[0][i].axis('off')
            ax_image[0][i].set_title(img_titles[i],color='w',y=0.8,fontsize=12)
            try:
                ax_image[0][i].imshow(images[i],cmap='gray')
            except:
                pass  
        for i,axis in enumerate(ax_image[1]):
            ax_image[1][i].axis('off')
            ax_image[1][i].set_title(img_titles[i+2],color='w',y=0.8,fontsize=12)
            try:
                ax_image[1][i].imshow(images[i+2],cmap='gray')
            except:
                pass

        fig.suptitle(mouseID+' Cell'+str(cell),fontsize=25,y=0.95)

        make_intrinsic_fig(mouseID,cell,cell_df,ax_in)
        ax_in.set_title('Intrinsic Profile')
        try:
            make_powercheck_fig(mouseID,cell,cell_df,ax_pwchk)
            make_hooks_fig(mouseID,cell,cell_df,ax_hooks)
        except Exception:
            pass
    #     ax_hooks[1].set_yticklabels([])

        fig.tight_layout()

        fig.subplots_adjust(top=0.9)

        if save==True:
            save_figure(fig,save_path+"\{}".format(fig_name),formats=['.png'],dpi=300)

    except Exception:
        print("can't do it ya fuckin' idiot"+' ('+mouseID+'_Cell'+str(cell)+')',Exception)