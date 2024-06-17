import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from hipe4ml.model_handler import ModelHandler
from hipe4ml.tree_handler import TreeHandler
from hipe4ml.analysis_utils import train_test_generator
from hipe4ml import plot_utils
from typing import Union, Sequence,List
from scipy import optimize



#####################################################
### Getting useful subsets of TreeHandler objects ###
#####################################################

# Function to convert a TreeHandler object to a numpy array for a given variable
def TreetoArray(data:TreeHandler, var:str)->np.array:
    """
    Convert a TreeHandler object to a numpy array for a given variable.

    Parameters:
        data (TreeHandler): The TreeHandler object containing the data.
        var (str): The variable name.

    Returns:
        np.ndarray: The numpy array containing the variable data.
    """
    df = data.get_data_frame()
    numpy_array=df.to_numpy()
    dataArr = np.array(pd.DataFrame(numpy_array, columns=df.columns)[var]).astype(np.float64)
    return dataArr

# Function to get raw data from a TreeHandler object
def get_rawdata(file_directory_data:str, tree_data:str, folder_name:Union[str,None]=None)->TreeHandler:
    """
    Get raw data from a TreeHandler object.

    Parameters:
        file_directory_data (str): The file directory of the data.
        tree_data (str): The name of the data tree.

    Returns:
        TreeHandler: The TreeHandler object containing the raw data without not known particles.
    """
    data=TreeHandler(file_directory_data, tree_data, folder_name=folder_name)
    return data

# Function to get raw MC data from a TreeHandler object
def get_rawMC_data(file_directory_mc:str, tree_mc:str,folder_name:Union[str,None]=None)->TreeHandler:
    """
    Get raw Monte Carlo (MC) data from a TreeHandler object.

    Parameters:
        file_directory_mc (str): The file directory of the MC data.
        tree_mc (str): The name of the MC tree.

    Returns:
        TreeHandler: The TreeHandler object containing the raw MC data without not known particles.
    """
    data_mc=TreeHandler(file_directory_mc, tree_mc,folder_name=folder_name)
    raw_mc=data_mc.get_subset("fMass!=-999 and fIsReco")
    return raw_mc

# Function to get background data from a TreeHandler object based on cuts
def get_bckg(file_directory_data:str, tree_data:str, cuts, folder_name:Union[str,None]=None)->TreeHandler:
    """
    Get background data from a TreeHandler object based on cuts.

    Parameters:
        file_directory_data (str): The file directory of the data.
        tree_data (str): The name of the data tree.
        cuts (list): A list containing the lower and upper bounds for the cuts.

    Returns:
        TreeHandler: The TreeHandler object containing the background data.
    """
    data=TreeHandler(file_directory_data, tree_data, folder_name=folder_name)
    bckg=data.get_subset(f'fMass<{cuts[0]} or fMass>{cuts[1]} and fMass!=-999')
    return bckg

# Function to get background from MC TreeHandler object based on selecting PDGCode -999
def get_MC_bckg(file_directory_mc:str, tree_mc:str,folder_name:Union[str,None]=None)->TreeHandler:
    """
    Get background data from a TreeHandler object based on cuts.

    Parameters:
        file_directory_data (str): The file directory of the data.
        tree_data (str): The name of the data tree.
        cuts (list): A list containing the lower and upper bounds for the cuts.

    Returns:
        TreeHandler: The TreeHandler object containing the background data.
    """
    data=TreeHandler(file_directory_mc, tree_mc,folder_name=folder_name)
    bckg=data.get_subset('fPDGCode==-999 and fMass!=-999')
    return bckg

# Function to get prompt MC from a TreeHandler object based on cuts
def get_prompt(file_directory_mc:str, tree_mc:str,folder_name:Union[str,None]=None)->TreeHandler:
    """
    Get prompt data from a TreeHandler object based on cuts.

    Parameters:
        file_directory_mc (str): The file directory of the MC data.
        tree_mc (str): The name of the MC tree.
        cuts (list): A list containing the lower and upper bounds for the cuts.

    Returns:
        TreeHandler: The TreeHandler object containing the prompt data.
    """
    data_mc=TreeHandler(file_directory_mc, tree_mc, folder_name=folder_name)
    #prompt = data_mc.get_subset(f'fPDGCodeMother==0 and (fMass>{cuts[0]} and fMass<{cuts[1]}) and fCosPA!=-999')
    prompt = data_mc.get_subset('fPDGCodeMother==0 and (fPDGCode == 3122 or fPDGCode== -3122) and fMass!=-999 and fIsReco')
    return prompt

# Function to get non-prompt MC from a TreeHandler object based on cuts
def get_nonprompt(file_directory_mc:str, tree_mc:str,folder_name:Union[str,None]=None)->TreeHandler:
    """
    Get non-prompt data from a TreeHandler object based on cuts.

    Parameters:
        file_directory_mc (str): The file directory of the MC data.
        tree_mc (str): The name of the MC tree.
        cuts (list): A list containing the lower and upper bounds for the cuts.

    Returns:
        TreeHandler: The TreeHandler object containing the non-prompt data.
    """
    data_mc=TreeHandler(file_directory_mc, tree_mc,folder_name=folder_name)
    #nonprompt = data_mc.get_subset(f'not fPDGCodeMother==0 and (fMass>{cuts[0]} and fMass<{cuts[1]}) and fCosPA!=-999')
    nonprompt = data_mc.get_subset('fPDGCodeMother!=0 and (fPDGCode == 3122 or fPDGCode== -3122) and fMass!=-999 and fIsReco')
    return nonprompt

def get_sampleSize(data):
    return len(TreetoArray(data,var="fCt"))

def filter_posneg_Pt(data:TreeHandler):
    data_pos=data.get_subset("fPt>0")
    data_neg=data.get_subset("fPt<0")
    return data_pos, data_neg

def proton_pion_division(data:TreeHandler)->TreeHandler:
    data_pos=data.get_subset("fPt>0")
    data_neg=data.get_subset("fPt<0")
    df=data.get_data_frame()
    df["fDcaPVProton"]=list(TreetoArray(data_pos, "fDcaPosPV"))+list(TreetoArray(data_neg, "fDcaNegPV"))
    df["fDcaPVPion"]=list(TreetoArray(data_neg, "fDcaPosPV"))+list(TreetoArray(data_pos, "fDcaNegPV"))
    df["fTpcNsigmaProton"]=list(TreetoArray(data_pos, "fTpcNsigmaPos"))+list(TreetoArray(data_neg, "fTpcNsigmaNeg"))
    df["fTpcNsigmaPion"]=list(TreetoArray(data_pos, "fTpcNsigmaNeg"))+list(TreetoArray(data_neg, "fTpcNsigmaPos"))

    tr=TreeHandler()
    tr.set_data_frame(df)
    return tr




#################################
### Distrubutions and Fitting ###
#################################

# Gauss distribution with offset
def gauss_offset(x: np.ndarray, a: float, mu: float, sigma: float, offset=float) -> np.ndarray:
    """
    Gauss distribution with offset

    Parameters:
        x (numpy.ndarray): x values
        a (flaot): height
        mu (float): mean value
        sigma (float): standard deviation
        offset (float): offset
    """
    return a * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))+offset

# Gauss distribution
def gauss(x: np.ndarray, a: float, mu: float, sigma: float) -> np.ndarray:
    """
    Gauss distribution with offset

    Parameters:
        x (numpy.ndarray): x values
        a (flaot): height
        mu (float): mean value
        sigma (float): standard deviation
    """
    return a * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

# Function to fit a TreeHandler object with a Gaussian
def fit_gauss_offset(data:TreeHandler, var:str, no_bins:int=100, fitting_range:list=[45,55] ,p0:Union[Sequence[float],None]=None, sig_cut:float=9.):
    dataArr=TreetoArray(data,var)
    hist, bin = np.histogram(dataArr, bins=no_bins)
    if p0:
        par,unc=optimize.curve_fit(gauss_offset,  bin[fitting_range[0]+1:fitting_range[1]+1], np.array(hist[fitting_range[0]:fitting_range[1]]),p0=(p0[0],p0[1],p0[2],p0[3]),sigma=np.sqrt(np.array(hist[fitting_range[0]:fitting_range[1]])), absolute_sigma=True)
    else: 
        par,unc=optimize.curve_fit(gauss_offset,  bin[fitting_range[0]+1:fitting_range[1]+1], np.array(hist[fitting_range[0]:fitting_range[1]]),sigma=np.sqrt(np.array(hist[fitting_range[0]:fitting_range[1]])), absolute_sigma=True)
    cuts=[par[1]-sig_cut*abs(par[2]),par[1]+sig_cut*abs(par[2])]
    return par, unc,cuts, fitting_range, bin, sig_cut

def fit_gauss(data:TreeHandler, var:str, no_bins:int=100, fitting_range:list=[45,55] ,p0:Union[Sequence[float],None]=None, sig_cut:float=9.):
    dataArr=TreetoArray(data,var)
    hist, bin = np.histogram(dataArr, bins=no_bins)
    if p0:
        par,unc=optimize.curve_fit(gauss,  bin[fitting_range[0]+1:fitting_range[1]+1], np.array(hist[fitting_range[0]:fitting_range[1]]),p0=(p0[0],p0[1],p0[2]),sigma=np.sqrt(np.array(hist[fitting_range[0]:fitting_range[1]])), absolute_sigma=True)
    else: 
        par,unc=optimize.curve_fit(gauss,  bin[fitting_range[0]+1:fitting_range[1]+1], np.array(hist[fitting_range[0]:fitting_range[1]]),sigma=np.sqrt(np.array(hist[fitting_range[0]:fitting_range[1]])), absolute_sigma=True)
    cuts=[par[1]-sig_cut*abs(par[2]),par[1]+sig_cut*abs(par[2])]
    return par, unc,cuts, fitting_range, bin, sig_cut

# Function to fit a TreeHandler object with a Gaussian by using recursivly smaller fit ranges
def fit_gauss_rec(data,var:str,no_bins=100,p0:Union[Sequence[float],None]=None, rec_len:int=2, sig:float=3.,sig_cut:float=9.):
    """
    Fit a Gaussian distribution to data with recursive fitting.

    Parameters:
        data (TreeHandler): The TreeHandler object containing the data.
        var (str): The variable name.
        no_bins (int, optional): The number of bins for the histogram. Defaults to 100.
        p0 (Union[Sequence[float], None], optional): Initial parameters for the Gaussian fit. Defaults to None.
        rec_len (int, optional): The number of recursive fits. Defaults to 2.
        sig (float, optional): The sigma range that includes the data for the fit in the next recursiv step. Defaults to 2.
        fs (Union[tuple, None], optional): The figure size. Defaults to (10, 7).
        sig_cut (float, optional): The sigma cut for the fit. Defaults to 7.

    Returns:
        tuple: The parameters, uncertainties, cuts, and fitting range from the final fit.
    """
    par, unc, cuts, fitting_range,bin,sig_cut=fit_gauss_offset(data, var, no_bins=no_bins, fitting_range=[1,no_bins],p0=p0,sig_cut=sig_cut)
    for i in range(1,rec_len):
        indx_onesig=find_closest_float_position(bin, par[1]-sig*abs(par[2]))
        indx_mu=find_closest_float_position(bin, par[1])
        new_fitting_range=[indx_onesig,indx_mu+(indx_mu-indx_onesig)]
        par, unc, cuts, fitting_range,newbin, sig_cut=fit_gauss(data, var, no_bins=no_bins, fitting_range=new_fitting_range, p0=[par[0],par[1],par[2]],sig_cut=sig_cut)
    return par, unc, cuts, fitting_range, newbin, sig_cut


# Function to find the position of a float in an array closest to a given float
def find_closest_float_position(array, target):
    """
    Finds the position of a float in an array that is closest to a given float.

    Parameters:
        array (list): List of floats.
        target (float): The target float.

    Returns:
        int: The position/index of the closest float in the array.
    """
    closest_index = min(range(len(array)), key=lambda i: abs(array[i] - target))
    return closest_index

#######################
### Plotting Tools ####
#######################

# Function to plot a histogram of a Treehandler object (or a Sequence of TreeHandler)
def plot_hist(to_plot:List[TreeHandler],vars_to_draw:List[str],no_bins:int=100,leg_labels:Union[Sequence[str],str,None]=None, fs:Union[tuple,None]=(10,7), alpha:Union[float, None]=0.3):
    if type(leg_labels)!=list:
        leg_labels=[leg_labels]
    plot_utils.plot_distr(to_plot, vars_to_draw, bins=no_bins, labels=leg_labels, log=True, density=True, figsize=fs, alpha=alpha, grid=False)
    plt.subplots_adjust(left=0.06, bottom=0.06, right=0.99, top=0.96, hspace=0.55, wspace=0.55)

# Plots a fitted distribuition (at this point only Gauss) to a given TreeHandler columns 
def plot_dist_fit(data:TreeHandler, var:str, par:list , no_bins:int=100 ,fitting_range:list=[45,55], cuts:Union[tuple,None]=None,dist:Union[str,None]="Gauss",fs:Union[tuple,None]=(10,7), title:Union[str, None]=None, sig_cut:float=9.,label:str="fit"):
    """
    Plot a fitted distribution (at this point only Gauss) to a given histogram.

    Parameters:
        data (TreeHandler): The TreeHandler object containing the data.
        var (str): The variable to be fitted, column of TreeHandler
        par (list): The parameters of the already fitted function.
        cuts (tuple): The lower and upper bounds of the cuts chosen for excluding data.
        no_bins (int, optional): The number of bins for the histogram. Defaults to 100.
        fitting_range (list, optional): The range for fitting the histogram. Defaults to [45, 55].
        dist (Union[str, None], optional): The distribution type. Defaults to "Gauss".
        fs (Union[tuple, None], optional): The figure size. Defaults to (10, 7).
        title (Union[str, None], optional): The title of the plot. Defaults to None.
        sig_cut (Union[float, None], optional): The sigma interval that was chosen fot the cuts. Defaults to None.
    """
    dataArr=TreetoArray(data,var)
    fig, ax=plt.subplots(figsize=fs)
    ax.set_title(title)
    hist, bins = np.histogram(dataArr, bins=no_bins)
    binwidth=bins[1]-bins[0]
    yerr=np.sqrt(np.array(hist))
    ax.errorbar(np.array(bins[1:]),hist,xerr=binwidth/2,yerr=yerr,linestyle="", label="data")
    if cuts:
        ax.plot([cuts[0],cuts[0]],[0, max(hist)],color="red")
        ax.plot([cuts[1],cuts[1]],[0, max(hist)],color="red",label=f"cuts: {sig_cut}-sigma")
    if dist=="Gauss":
        ax.plot(bins[fitting_range[0]:fitting_range[1]], gauss(bins[int(fitting_range[0]):int(fitting_range[1])],*par),label=label)
    if dist=="Gauss_offset":
        ax.plot(bins[fitting_range[0]:fitting_range[1]], gauss_offset(bins[int(fitting_range[0]):int(fitting_range[1])],*par),label=label)
    x=bins[1:]
    y=hist
    for xi, yi, err in zip(x, y, np.full_like(x, binwidth/2)):
        ax.fill_betweenx([0, yi], xi - err, xi + err, color='blue', alpha=0.2)
    ax.set_yscale("log")
    ax.legend()
    ax.set_ylabel("counts")
    plt.subplots_adjust(left=0.06, bottom=0.06, right=0.99, top=0.96, hspace=0.55, wspace=0.55)

# Function to Fit and plot a Gaussian distribution to data with recursive fitting.
def fitplot_gauss_rec(data,var:str,no_bins=100,p0:Union[Sequence[float],None]=None, rec_len:int=2, sig:float=3., fs:Union[tuple,None]=(10,7),sig_cut:float=9.):
    """
    Fit and plot a Gaussian distribution to data with recursive fitting.

    Parameters:
        data (TreeHandler): The TreeHandler object containing the data.
        var (str): The variable name.
        no_bins (int, optional): The number of bins for the histogram. Defaults to 100.
        p0 (Union[Sequence[float], None], optional): Initial parameters for the Gaussian fit. Defaults to None.
        rec_len (int, optional): The number of recursive fits. Defaults to 2.
        sig (float, optional): The sigma range that includes the data for the fit in the next recursiv step. Defaults to 2.
        fs (Union[tuple, None], optional): The figure size. Defaults to (10, 7).
        sig_cut (float, optional): The sigma cut for the fit. Defaults to 7.

    Returns:
        tuple: The parameters, uncertainties, cuts, and fitting range from the final fit.
    """
    par, unc, cuts, fitting_range,bin,sig_cut=fit_gauss_offset(data, var, no_bins=no_bins, fitting_range=[1,no_bins],p0=p0,sig_cut=sig_cut)
    print("Recursion: 0","\nParameter: ", par)
    print("range: ", fitting_range)
    plot_dist_fit(data, var=var,par=par ,cuts=cuts ,fitting_range=[1,no_bins], dist="Gauss_offset",title="rec 0",fs=fs,sig_cut=sig_cut,label="fit in whole range")
    for i in range(1,rec_len):
        indx_onesig=find_closest_float_position(bin, par[1]-sig*abs(par[2]))
        indx_mu=find_closest_float_position(bin, par[1])
        new_fitting_range=[indx_onesig+1,indx_mu+(indx_mu-indx_onesig)]
        par, unc, new_cuts, fitting_range,newbin,sig_cut=fit_gauss(data, var, no_bins=no_bins, fitting_range=new_fitting_range, p0=[par[0],par[1],par[2]],sig_cut=sig_cut)
        print("Recursion: ", i,"\nParameter: ", par)
        print("range: ", new_fitting_range)
        plot_dist_fit(data,var=var, par=par ,cuts=new_cuts ,fitting_range=fitting_range,title=f"rec {i}", fs=fs,sig_cut=sig_cut,label=f"fit in {sig}-range of prev. fit")

    return par, unc, new_cuts, fitting_range


# Function to get all possible variables for a comparison plot
def var_draw_all(tree1:TreeHandler, tree2:TreeHandler)->list:
    THs=[tree1,tree2]
    return [dat for dat in THs[0].get_var_names() if all(dat in entry.get_var_names() for entry in THs)]

def plot_2dhist(data:TreeHandler, var1:str, var2:str,ax, bins:int=100, cmap:str="rainbow"):
    df = data.get_data_frame()
    numpy_array=df.to_numpy()
    dataArr1 = np.array(pd.DataFrame(numpy_array, columns=df.columns)[var1]).astype(np.float64)
    dataArr2 = np.array(pd.DataFrame(numpy_array, columns=df.columns)[var2]).astype(np.float64)
    hist, binsx,binsy=np.histogram2d(dataArr1,dataArr2, bins=100)
    hist=hist.T

    ax.pcolormesh(binsx, binsy, hist, cmap=cmap)
    ax.set_xlim(binsx.min(), binsx.max())
    ax.set_ylim(binsy.min(), binsy.max())
