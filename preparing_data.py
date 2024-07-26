import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hipe4ml.tree_handler import TreeHandler
from hipe4ml.model_handler import ModelHandler
from hipe4ml import plot_utils
from typing import Union, Sequence,List, Tuple
from scipy import optimize
import ROOT
import uproot
import awkward as ak
import os
import seaborn as sns
from ROOT import RooFit, RooRealVar, RooDataSet, RooArgList, RooAddPdf, RooPlot, RooArgList, RooPolynomial, RooDataHist, RooHist, TCanvas
import sys
import io

class ErrorFilter(io.StringIO):
    def __init__(self):
        super().__init__()

    def write(self, msg):
        if 'Error in <TTree::Fill>' not in msg:
            sys.__stderr__.write(msg)

# Redirect the standard error
sys.stderr = ErrorFilter()

###################################################
### Convert TreeHandler, Arrays and .root Files ###
###################################################

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

# Function to save a TreeHandler as a .root file
def get_root_from_TreeHandler(treehdl:TreeHandler,output_name:str,save_dir:str,treename:str):
    """
    Convert a TreeHandler object to a .root for a given variable.

    Parameters:
        data (TreeHandler): The TreeHandler object containing the data.
        save_dir (str): Name of the direction, in which the .root file should be safed.
        output_name (str): name of the output file. Should end with .root
    """
    # Convert the TreeHandler DataFrame to a ROOT file
    df=treehdl.get_data_frame()
    ak_array = ak.Array(df.to_dict(orient='list'))
    with uproot.recreate(save_dir+output_name) as file:
        file[treename] = ak_array


def get_sampleSize(data:TreeHandler)->int:
    """
     Parameters:
        data(TreeHandler): data for that the sample size is desired

    Returns:
        int: sample size
    """
    return len(TreetoArray(data,var="fCt"))

def get_variable_names(obj_list, namespace):
    result = {}
    for obj in obj_list:
        names = [name for name, value in namespace.items() if value is obj]
        if len(names)==1:
            names=names[0]
        if names:
            result[obj] = names
    return list(result.values())

# Function to rename a column if a root file
def rename_column(file_name:str,file_dir:str,old_column_name:str, new_column_name:str,tree:str="tree"):

    th=TreeHandler(file_name=file_dir+file_name, tree_name=tree)
    df=th.get_data_frame()
    df.rename(columns={f'{old_column_name}': f'{new_column_name}'}, inplace=True)
    th2=TreeHandler()
    th2.set_data_frame(df)
    get_root_from_TreeHandler(treehdl=th2,output_name=file_name,save_dir=file_dir,treename=tree)
    
##################################################################
### Getting useful data subsets in form of TreeHandler objects ###
##################################################################

# Function to get a TreeHandler containing the data from a .root file
def get_rawdata(file_directory_data:str, tree_data:str, folder_name:Union[str,None]=None)->TreeHandler:
    """
    Get raw data from a TreeHandler object.

    Parameters:
        file_directory_data (str): The file directory of the .root file
        tree_data (str): The name of the tree with the desired data
        folder_name (str): The folder in which the tree is contained. Default to None (no folders)

    Returns:
        TreeHandler: The TreeHandler object containing the data
    """
    data=add_Radius(proton_pion_division(TreeHandler(file_directory_data, tree_data, folder_name=folder_name)))
    return data


# Function to get a TreeHandler containing the Monte Carlo data from a .root file
def get_rawMC_data(file_directory_mc:str, tree_mc:str,folder_name:Union[str,None]=None)->TreeHandler:
    """
    Get Monte Carlo data from a TreeHandler object.

    Parameters:
        file_directory_mc (str): The file directory of the .root file
        tree_mc (str): The name of the tree with the desired Monte Carlo data
        folder_name (str): The folder in which the tree is contained. Default to None (no folders)

    Returns:
        TreeHandler: The TreeHandler object containing the Monte Carlo
    """
    data_mc=TreeHandler(file_directory_mc, tree_mc,folder_name=folder_name)
    raw_mc=data_mc.get_subset("fMass!=-999 and fIsReco")
    return raw_mc

# Function to get a TreeHandler containing the backgrund from a .root data file 
def get_bckg(file_directory_data:str, tree_data:str, cuts, folder_name:Union[str,None]=None)->TreeHandler:
    """
    Get a TreeHandler containing the backgrund from a .root file. Background is chosen from real data by 
    setting cuts for the invariant Mass.

    Parameters:
        file_directory_data (str): The file directory of the data.
        tree_data (str): The name of the data tree.
        cuts (list): A list containing the lower and upper bounds for the cuts.
        folder_name (str): The folder in which the tree is contained. Default to None (no folders)

    Returns:
        TreeHandler: The TreeHandler object containing the background data.
    """
    data=TreeHandler(file_directory_data, tree_data, folder_name=folder_name)
    bckg=data.get_subset(f'fMass<{cuts[0]} or fMass>{cuts[1]} and fMass!=-999')
    return bckg

# Function to get a TreeHandler containing the backgrund from a .root file with Monte Carlo generated data 
def get_MC_bckg(file_directory_mc:str, tree_mc:str, folder_name:Union[str,None]=None)->TreeHandler:
    """
    Get a TreeHandler containing the background from a .root file with Monte Carlo generated data.
    Background is chosen by using only events with PDGCode==-999

    Parameters:
        file_directory_mc (str): The file directory of the data.
        tree_mc (str): The name of the data tree.
        folder_name (str): The folder in which the tree is contained. Default to None (no folders)

    Returns:
        TreeHandler: The TreeHandler object containing the background data.
    """
    data=TreeHandler(file_directory_mc, tree_mc,folder_name=folder_name)
    bckg=data.get_subset('fPDGCode==-999 and fMass!=-999')
    return bckg

# Function to get a TreeHandler containing the prompt from a .root file with Monte Carlo generated data 
def get_prompt(file_directory_mc:str, tree_mc:str, folder_name:Union[str,None]=None)->TreeHandler:
    """
    Get a TreeHandler containing the prompt from a .root file with Monte Carlo generated data.
    Prompt is chosen by using only events with PDGCode==3122 or -3122 and PDGCodeMother==0

    Parameters:
        file_directory_mc (str): The file directory of the data.
        tree_mc (str): The name of the data tree.
        folder_name (str): The folder in which the tree is contained. Default to None (no folders)

    Returns:
        TreeHandler: The TreeHandler object containing the prompt data.
    """
    data_mc=TreeHandler(file_directory_mc, tree_mc, folder_name=folder_name)
    #prompt = data_mc.get_subset(f'fPDGCodeMother==0 and (fMass>{cuts[0]} and fMass<{cuts[1]}) and fCosPA!=-999')
    prompt = data_mc.get_subset('fPDGCodeMother==0 and (fPDGCode == 3122 or fPDGCode== -3122) and fMass!=-999 and fIsReco')
    return prompt

# Function to get a TreeHandler containing the non-prompt from a .root file with Monte Carlo generated data 
def get_nonprompt(file_directory_mc:str, tree_mc:str, folder_name:Union[str,None]=None)->TreeHandler:
    """
    Get a TreeHandler containing the non-prompt from a .root file with Monte Carlo generated data.
    Prompt is chosen by using only events with PDGCode==3122 or -3122 and PDGCodeMother!=0

    Parameters:
        file_directory_mc (str): The file directory of the data.
        tree_mc (str): The name of the data tree.
        folder_name (str): The folder in which the tree is contained. Default to None (no folders)

    Returns:
        TreeHandler: The TreeHandler object containing the prompt data.
    """
    data_mc=TreeHandler(file_directory_mc, tree_mc,folder_name=folder_name)
    #nonprompt = data_mc.get_subset(f'not fPDGCodeMother==0 and (fMass>{cuts[0]} and fMass<{cuts[1]}) and fCosPA!=-999')
    nonprompt = data_mc.get_subset('fPDGCodeMother!=0 and (fPDGCode == 3122 or fPDGCode== -3122) and fMass!=-999 and fIsReco')
    return nonprompt

# Function to get a TreeHandler containing the subsets with positive/negative fPt 
def filter_posneg_Pt(data:TreeHandler)->Sequence[TreeHandler]:
    '''
    Filters the TreeHandler in two subsets: divided in positive/negative fPt

    Parameters:
        data (TreeHandler): TreeHandler containing the data to be divided
    
    Returns:
        data_pos(TreeHandler): The TreeHandler containg events with positive fPt
        data_neg(TreeHandler): The TreeHandler containg events with negative fPt
    '''
    data_pos=data.get_subset("fPt>0")
    data_neg=data.get_subset("fPt<0")
    return data_pos, data_neg

def cut_data(data:TreeHandler, var:str, lower_cut:Union[float,None]=None,upper_cut:Union[float,None]=None, inclusive:bool=True)->TreeHandler:
    '''
    Parameters:
        data (TreeHandler): TreeHandler containing the data to be cutted
        var (str): Column of TreeHandler, for that the cut is applied
        lower_cut (float): lower cut
        upper_cut (float): upper cut (if same as lower_cut, only data equal/(! equal) to this specific value is used)
        inclusive (bool): If True, data inside the cuts is used, if false, data outside the cuts is used. Default to inclusive (True)

    Returns:
        TreeHandler -> containing the desired subset
    '''
    if inclusive:
        if upper_cut and lower_cut:
            if upper_cut==lower_cut:
                cutted=data.get_subset(f'{var}=={lower_cut}')
            else:
                cutted=data.get_subset(f'{var}>{lower_cut} and {var}<{upper_cut}')
        if upper_cut and not lower_cut:
            cutted=data.get_subset(f'{var}<{upper_cut}')
        if lower_cut and not upper_cut:
            cutted=data.get_subset(f'{var}>{lower_cut}')
        if not upper_cut and not lower_cut:
            cutted=data
    else:
        if upper_cut and lower_cut:
            if upper_cut==lower_cut:
                cutted=data.get_subset(f'{var}!={lower_cut}')
            else:
                cutted=data.get_subset(f'{var}<{lower_cut} or {var}>{upper_cut}')

    return cutted



#############################################
### Adding useful columns to TreeHandlers ###
#############################################

# Function to get a TreeHandler for which the fDcaPV and the fPpcNsigma is sorted for protons and pions
def proton_pion_division(data:TreeHandler)->TreeHandler:
    '''
    Add new columns in TreeHanler: "fDcaPVProton","fDcaPVPion","fTpcNsigmaProton","fTpcNsigmaPion"

    Parameters:
        data (TreeHandler): TreeHandler containing the data to be sorted 
    
    Returns:
        TreeHandler with sorted data
    '''
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

# Function to add the generated radius to a TreeHandler, only applicable on MC data
def add_GenRadius(data:TreeHandler)->TreeHandler:
    '''
    Parameter: 
        data (TreeHanlder)

    Returns: 
        TreeHandler with an extra column (fGenRadius added
    '''
    df=data.get_data_frame()
    M=1.11568
    Pt=TreetoArray(data, "fGenPt")
    Ct=TreetoArray(data, "fGenCt")
    L=(Ct*Pt)/M
    df["fGenRadius"]=np.abs(L)
    tr=TreeHandler()
    tr.set_data_frame(df)
    return tr

# Function to add the calculated radius to a TreeHandler
def add_Radius(data:TreeHandler)->TreeHandler:
    '''
    Parameter: 
        data (TreeHanlder)

    Returns: 
        TreeHandler with an extra column (fRadius_calc) added
    '''
    df=data.get_data_frame()
    M=TreetoArray(data, "fMass")
    Pt=TreetoArray(data, "fPt")
    Ct=TreetoArray(data, "fCt")
    L=(Ct*Pt)/M
    df["fRadius_calc"]=np.abs(L)
    tr=TreeHandler()
    tr.set_data_frame(df)
    return tr

def get_base_sets(file_directory_data:str, tree_data:str, file_directory_mc:str, tree_mc:str, fname:Union[str,None]=None)->Sequence[TreeHandler]:
    '''
    Parameters:
        file_directory_data (str): The file directory of the data.
        tree_data (str): The name of the data tree.        
        file_directory_mc (str): The file directory of the MC data.
        tree_mc (str): The name of the MC tree.
        fname (str): The folder in which the tree is contained. Default to None (no folders)
    
    Returns:
        Sequence[TreeHandler]: prompt, nonprompt, bckg, bckg_MC
    
    '''
    prompt=add_GenRadius(proton_pion_division(get_prompt(file_directory_mc, tree_mc, folder_name=fname)))
    nonprompt=add_GenRadius(proton_pion_division(get_nonprompt(file_directory_mc, tree_mc, folder_name=fname)))
    cu=fit_gauss_rec(get_rawdata(file_directory_data, tree_data, folder_name=fname), var="fMass",p0=[300000,1.115,0.005,1000])[2]
    bckg_data=add_Radius(proton_pion_division(get_bckg(file_directory_data, tree_data, cu, folder_name=fname)))
    bckg_MC=add_GenRadius(proton_pion_division(get_MC_bckg(file_directory_mc, tree_mc, folder_name=fname)))

    return prompt, nonprompt, bckg_data, bckg_MC




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
    Gauss distribution

    Parameters:
        x (numpy.ndarray): x values
        a (flaot): height
        mu (float): mean value
        sigma (float): standard deviation
    """
    return a * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

# Function to fit data from a TreeHandler object with a Gaussian with offset
def fit_gauss_offset(data:TreeHandler, var:str, no_bins:int=100, fitting_range:Sequence[int]=[45,55] ,p0:Union[Sequence[float],None]=None, sig_cut:float=9.):
    """
    Fit a Gauss to a TreeHandler column

    Parameters:
        data (TreeHandler): TreeHandler that contains the data to be fitter
        var (str): column to be fitted
        no_bins (int): Number of bins, that are used for the fitting. Default to 100
        fitting_range (List[int]): range that is used for the fit. Default to [45,55]
        p0 (Sequence[float],None): start parameters used for fit. Default to None
        sig_cut (float): The sigma interval where the cuts for the chosen data are visualised. Default to 9

    Returns:
        par (List[float]): The optimal parameters
        unc (List[float]): The uncertainty array of the optimal parameters
        cuts (List[float]): The data cuts, based on the given sig_cut intervall
        fitting_range(List[int]): The fitting range that was used
        bin (int): no. of bins
        sig_cut (float): The sigma interval where the cuts for the chosen data are visualised.
    """
    dataArr=TreetoArray(data,var)
    hist, bin = np.histogram(dataArr, bins=no_bins)
    if p0:
        par,unc=optimize.curve_fit(gauss_offset,  bin[fitting_range[0]+1:fitting_range[1]+1], np.array(hist[fitting_range[0]:fitting_range[1]]),p0=(p0[0],p0[1],p0[2],p0[3]),sigma=np.sqrt(np.array(hist[fitting_range[0]:fitting_range[1]])), absolute_sigma=True)
    else: 
        par,unc=optimize.curve_fit(gauss_offset,  bin[fitting_range[0]+1:fitting_range[1]+1], np.array(hist[fitting_range[0]:fitting_range[1]]),sigma=np.sqrt(np.array(hist[fitting_range[0]:fitting_range[1]])), absolute_sigma=True)
    cuts=[par[1]-sig_cut*abs(par[2]),par[1]+sig_cut*abs(par[2])]
    return par, unc,cuts, fitting_range, bin, sig_cut


# Function to fit data from a TreeHandler object with a Gaussian
def fit_gauss(data:TreeHandler, var:str, no_bins:int=100, fitting_range:list=[45,55] ,p0:Union[Sequence[float],None]=None, sig_cut:float=9.):
    """
    Fit a Gauss to a TreeHandler column

    Parameters:
        data (TreeHandler): TreeHandler that contains the data to be fitter
        var (str): column to be fitted
        no_bins (int): Number of bins, that are used for the fitting. Default to 100
        fitting_range (List[int]): range that is used for the fit. Default to [45,55]
        p0 (Sequence[float],None): start parameters used for fit. Default to None
        sig_cut (float): The sigma interval where the cuts for the chosen data are visualised. Default to 9

    Returns:
        tuple(
        par (List[float]): The optimal parameters
        unc (List[float]): The uncertainty array of the optimal parameters
        cuts (List[float]): The data cuts, based on the given sig_cut intervall
        fitting_range(List[int]): The fitting range that was used
        bin (int): no. of bins
        sig_cut (float): The sigma interval where the cuts for the chosen data are visualised.
        )
    """
    dataArr=TreetoArray(data,var)
    hist, bin = np.histogram(dataArr, bins=no_bins)
    if p0:
        par,unc=optimize.curve_fit(gauss,  bin[fitting_range[0]+1:fitting_range[1]+1], np.array(hist[fitting_range[0]:fitting_range[1]]),p0=(p0[0],p0[1],p0[2]),sigma=np.sqrt(np.array(hist[fitting_range[0]:fitting_range[1]])), absolute_sigma=True)
    else: 
        par,unc=optimize.curve_fit(gauss,  bin[fitting_range[0]+1:fitting_range[1]+1], np.array(hist[fitting_range[0]:fitting_range[1]]),sigma=np.sqrt(np.array(hist[fitting_range[0]:fitting_range[1]])), absolute_sigma=True)
    cuts=[par[1]-sig_cut*abs(par[2]),par[1]+sig_cut*abs(par[2])]
    return par, unc,cuts, fitting_range, bin, sig_cut

# Function to fit a TreeHandler object with a Gaussian by using recursivly smaller fit ranges
def fit_gauss_rec(data:TreeHandler, var:str, no_bins:int=100, p0:Union[Sequence[float],None]=None, rec_len:int=2, sig:float=3.,sig_cut:float=9.):
    """
    Fit a Gaussian distribution to data with recursive fitting.

    Parameters:
        data (TreeHandler): The TreeHandler object containing the data.
        var (str): The column to be fitted
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
def plot_hist(to_plot:Union[TreeHandler,Sequence[TreeHandler]], vars_to_draw:Sequence[str], leg_labels:Union[Sequence[str],str], no_bins:int=100,  fs:Union[tuple,None]=(10,7), alpha:Union[float, None]=0.3, colors=None, logy:bool=True,fontsize:int=20):
    
    ax=plot_utils.plot_distr(to_plot, vars_to_draw, bins=no_bins, labels=leg_labels, log=logy, density=True, figsize=fs, alpha=alpha, grid=False,colors=colors)
    if len(vars_to_draw)>1:
        for ax_i, var in zip(ax,vars_to_draw):
            legend = ax_i.get_legend()
            if legend:
                for label in legend.get_texts():
                    label.set_fontsize(fontsize)
            ax_i.set_xlabel(var,fontsize=fontsize)
            ax_i.set_ylabel("Counts",fontsize=fontsize)
            ax_i.set_title("")
    else:
        legend = ax.get_legend()
        if legend:
            for label in legend.get_texts():
                label.set_fontsize(fontsize)
        ax.set_xlabel(vars_to_draw[0],fontsize=fontsize)
        ax.set_ylabel("Counts",fontsize=fontsize)
        ax.set_title("")
    plt.subplots_adjust(left=0.06, bottom=0.06, right=0.99, top=0.96, hspace=0.55, wspace=0.55)
    return ax

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
    return ax

# Function to Fit and plot a Gaussian distribution to data with recursive fitting.
def fitplot_gauss_rec(data:TreeHandler, var:str,no_bins=100,p0:Union[Sequence[float],None]=None, rec_len:int=2, sig:float=3., fs:Union[tuple,None]=(10,7),sig_cut:float=9.):
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
def var_draw_all(trees:Sequence[TreeHandler])->Sequence[str]:
    """
    Get all possible variables for a comparison plot

    Parameters:
        tree1 (TreeHandler): First TreeHandler for the comparison
        tree2 (TreeHandler): Second TreeHandler for the comparison

    Returns:
        Sequence[str]: List containing all shared columns
    """
    return [dat for dat in trees[0].get_var_names() if all(dat in entry.get_var_names() for entry in trees)]

# Function to plot and save a 2d histogram using PYthon
def plot_2dhist_numpy(data:TreeHandler, var1:str, var2:str, ax, hist_name:Union[str,None]=None, binsx:int=100,binsy:int=100,cmap:str="rainbow"):
    """
    Plot a 2 dimensional histogram using numpy.

    Parameters:
        data (TreeHandler): TreeHandler that contains the desired data for the 2d histogram
        var1 (str): x-data for the histogram, column name of data TreeHandler
        var2 (str): y-data for the histogram, column name of data TreeHandler
        ax: numpy axis object in which the histogram will be plotted
        bins (int): no. of bins in the histogram
        cmap (str): Cmap, used in the plot. Default to "rainbow"
    """
    df = data.get_data_frame()
    numpy_array=df.to_numpy()
    dataArr1 = np.array(pd.DataFrame(numpy_array, columns=df.columns)[var1]).astype(np.float64)
    dataArr2 = np.array(pd.DataFrame(numpy_array, columns=df.columns)[var2]).astype(np.float64)
    hist, binsx,binsy=np.histogram2d(dataArr1,dataArr2, bins=[binsx,binsy], range=[(np.min(dataArr1),np.max(dataArr1)),(np.min(dataArr2),np.max(dataArr2))])
    hist=hist.T
    if hist_name:
        ax.set_title(hist_name)
    cax=ax.pcolormesh(binsx, binsy, hist, cmap=cmap)
    #ax.set_xlim(binsx.min(), binsx.max())
    #ax.set_ylim(binsy.min(), binsy.max())
    return cax

def plot_hist_root(var_name:str, hist_name:str, title:str="title", data:Sequence[float]=[0,1], file_name:str="root", no_bins:int=100,logy:bool=True, save_name_pdf:Union[str,None]=None, save_name_file:Union[str,None]=None, from_file:bool=False):
    
    hist = ROOT.TH1F(hist_name, hist_name, no_bins, 0, 1)
    if from_file:
        file=uproot.open(file_name)
        tree = file["tree"]
        branch_array = tree[var_name].array(library="np")
        for value in list(branch_array):
            hist.Fill(value)
    else:
        for value in data:
            hist.Fill(value)
    canvas = ROOT.TCanvas("canvas", title, 800, 600)
    hist.Draw()

    hist.SetXTitle(var_name)
    if logy:
        ROOT.gPad.SetLogy(1)
    canvas.Draw()
    # Save the canvas as an image file if needed
    if save_name_pdf:
        canvas.SaveAs(save_name_pdf)
        canvas.Close()
    # Create a new ROOT file to save the histogram
    if save_name_file:
        if os.path.exists(save_name_file):
            output_file = ROOT.TFile(save_name_file, "UPDATE")
        else:
            output_file = ROOT.TFile(save_name_file, "RECREATE")
        hist.Write()
        output_file.Close()


# Function to draw and save a 2d histogram using ROOT
def plot_2dhist_root(file:str,  var1:str, var2:str, save_name_file:str, hist_name:str, title:str, save_name_pdf:Union[str,None]=None ,binsx:int=7,binsy:int=7,cmap=ROOT.kRainbow, minx:float=0, maxx:float=0, miny:float=0 , maxy:float=0,cuts:Union[str,None]=None, save_file:bool=False, logz:bool=True):
    """
    Plot a 2 dimensional histogram using ROOT. Saves the histogram as pdf and as a .root file.

    Parameters:
        file (str): .root File that contains the desired data for the 2d histogram
        tree_name (str): name of tree, in which teh data is stored
        var1 (str): x-data for the histogram, column name of data TreeHandler
        var2 (str): y-data for the histogram, column name of data TreeHandler
        save_name_file (str): Name of the .root file that will contain the histogram. If already existing, it will be opened.
        hist_name (str): Name of the histogram that will be stored in the .root file
        save_name_pdf (str): Name of the pdf file, that will be saved.
        title (str): title of the histogram
        bins (int): no. of bins in the histogram. Default to 7
        cmap: Cmap, used in the plot. Default to ROOT.kRainbow
    """
    root_file = ROOT.TFile(file)
    if cuts:
        hist = ROOT.TH2F(hist_name, title+" Cut: " + cuts, binsx , minx, maxx, binsy, miny,maxy)
    else:
        hist = ROOT.TH2F(hist_name, title, binsx , minx, maxx, binsy, miny,maxy)

    all_trees = find_trees(root_file)
    for tree_name in all_trees:
        hist_name = "temp_hist_{tree_name}"
        hist_help = ROOT.TH2F(hist_name, "", binsx , minx, maxx, binsy, miny,maxy)
        #print(tree_name)
        tree = root_file.Get(tree_name)
        if cuts:
            tree_cutted = tree.CopyTree(cuts)
            tree_cutted.Draw(f"{var2}:{var1} >> {hist_name}")
        else:
            tree.Draw(f"{var2}:{var1} >> {hist_name}")
        hist.Add(hist_help)      

    print(f"ok, draw {var2} -> {var1}")
    canvas = ROOT.TCanvas("canvas", "", 1000, 600)
    canvas.SetRightMargin(0.33)
    hist.Draw("COLZ")
    legend = hist.GetListOfFunctions().FindObject("TPaveStats") 
    if legend:
        legend.SetX1NDC(0.80)  # Adjust X1 position in normalized coordinates
        legend.SetX2NDC(0.98)  # Adjust X2 position in normalized coordinates
        legend.SetY1NDC(0.80)  # Adjust Y1 position in normalized coordinates
        legend.SetY2NDC(0.95)  # Adjust Y2 position in normalized coordinates

    
    hist.SetFillStyle(3000)
    hist.SetFillColor(ROOT.kAzure+6)

    ROOT.gStyle.SetPalette(cmap)
    hist.Draw("COLZ")  # "COLZ" draws a 2D plot with a color palette
    hist.SetXTitle(var1)
    hist.SetYTitle(var2)
    if logz:
        ROOT.gPad.SetLogz(1)
    canvas.Draw()
    # Save the canvas as an image file if needed
    if save_name_pdf:
        canvas.SaveAs(save_name_pdf)
        canvas.Close()
    # Create a new ROOT file to save the histogram
    if save_file:
        if os.path.exists(save_name_file):
            output_file = ROOT.TFile(save_name_file, "UPDATE")
        else:
            output_file = ROOT.TFile(save_name_file, "RECREATE")
        hist.Write()
        output_file.Close()

def add_projection(file_name:str, hist_name:str, save_name_pdf:Union[str,None]=None, axis:int=0):

    file = ROOT.TFile.Open(file_name)
    h2 = file.Get(hist_name)
    if axis==0:
        h1 = h2.ProjectionX()
    if axis==1:
        h1 = h2.ProjectionY()
    print(f"ok, projection done!")
    canvas = ROOT.TCanvas("canvas", "", 1000, 600)
    canvas.SetRightMargin(0.33)
    h1.Draw()
    legend = h1.GetListOfFunctions().FindObject("TPaveStats") 
    if legend:
        legend.SetX1NDC(0.80)  # Adjust X1 position in normalized coordinates
        legend.SetX2NDC(0.98)  # Adjust X2 position in normalized coordinates
        legend.SetY1NDC(0.80)  # Adjust Y1 position in normalized coordinates
        legend.SetY2NDC(0.95)  # Adjust Y2 position in normalized coordinates
    h1.SetFillStyle(3000)
    h1.SetFillColor(ROOT.kAzure+6)
    if save_name_pdf:
        canvas.SaveAs(save_name_pdf)
    canvas.Close()
    output_file = ROOT.TFile(file_name, "UPDATE")
    h1.Write()
    output_file.Close()


# Function to draw and save a 3d histogram using ROOT
def plot_3dhist_root(file:str,  var1:str, var2:str,var3:str, save_name_file:str, hist_name:str, save_name_pdf:Union[str,None]=None , title:str="hist_3d" ,binsx:int=100, binsy:int=100, binsz:int=100, cuts:Union[str,None]=None, cmap=ROOT.kRainbow,save_file:bool=False):
   
    """
    Plot a 2 dimensional histogram using ROOT. Saves the histogram as pdf and as a .root file.

    Parameters:
        file (str): .root File that contains the desired data for the 2d histogram
        tree_name (str): name of tree, in which teh data is stored
        var1 (str): x-data for the histogram, column name of data TreeHandler
        var2 (str): y-data for the histogram, column name of data TreeHandler
        var3 (str): z-data for the histogram, column in the TreeHandler
        save_name_file (str): Name of the .root file that will contain the histogram. If already existing, it will be opened.
        hist_name (str): Name of the histogram that will be stored in the .root file
        save_name_pdf (str): Name of the pdf file, that will be saved.
        title (str): title of the histogram
        bins (int): no. of bins in the histogram. Default to 7
        cmap: Cmap, used in the plot. Default to ROOT.kRainbow
    """
    root_file = ROOT.TFile(file, "READ")
    #tree = file.Get(tree_name)
    #hist = ROOT.TH3F(hist_name, title, binsx, 0,0, binsy, 0,0,binsz,0,0)

    #tree.Draw(f"{var3}:{var2}:{var1} >> {hist_name}")

    hist = ROOT.TH3F(hist_name, title, binsx, 0,0, binsy, 0,0, binsz,0,0)
    all_trees = find_trees(root_file)
    for tree_name in all_trees:
        hist_name = "temp_hist_{tree_name}"
        hist_help = ROOT.TH3F(hist_name, "",binsx, 0,0, binsy, 0,0, binsz,0,0)
        #print(tree_name)
        tree = root_file.Get(tree_name)
        if cuts:
            tree_cutted = tree.CopyTree(cuts)
            tree_cutted.Draw(f"{var3}:{var2}:{var1} >> {hist_name}")
        else:
            tree.Draw(f"{var3}:{var2}:{var1} >> {hist_name}")
        hist.Add(hist_help)     

    print(f"ok, tree draws {var1}, {var2},{var3}")
    canvas = ROOT.TCanvas("canvas", "", 1000, 600)
    canvas.SetRightMargin(0.33)
    hist.Draw("LEGO")
    legend = hist.GetListOfFunctions().FindObject("TPaveStats") 
    if legend:
        legend.SetX1NDC(0.80)  # Adjust X1 position in normalized coordinates
        legend.SetX2NDC(0.98)  # Adjust X2 position in normalized coordinates
        legend.SetY1NDC(0.80)  # Adjust Y1 position in normalized coordinates
        legend.SetY2NDC(0.95)  # Adjust Y2 position in normalized coordinates

    
    hist.SetFillStyle(3000)
    hist.SetFillColor(ROOT.kAzure+6)

    ROOT.gStyle.SetPalette(cmap)
    hist.Draw("LEGO")  # "COLZ" draws a 2D plot with a color palette
    hist.SetXTitle(var1)
    hist.SetYTitle(var2)
    hist.SetZTitle(var3)
    #ROOT.gPad.SetLogz(1)

    # Save the canvas as an image file if needed
    canvas.SaveAs(save_name_pdf)
    canvas.Close()
    # Create a new ROOT file to save the histogram
    if save_file:
        if os.path.exists(save_name_file):
            output_file = ROOT.TFile(save_name_file, "UPDATE")
        else:
            output_file = ROOT.TFile(save_name_file, "RECREATE")
        hist.Write()
        output_file.Close()


# Function to plot several histograms in one pdf
def plot_histograms_grid(root_file:str,save_name_pdf:str):
    """
    Parameters:
        root_file (str): Direction of root file that contains all histograms
        save_name_pdf (str): Name of the PDF that will be saved
    """

    file = ROOT.TFile(root_file, "READ")
    hist1=file.Get(file.GetListOfKeys()[0].GetName())
    hist2=file.Get(file.GetListOfKeys()[1].GetName())
    hist3=file.Get(file.GetListOfKeys()[2].GetName())
    hist4=file.Get(file.GetListOfKeys()[3].GetName())

    canvas = ROOT.TCanvas("canvas", "", 1000, 600)
    canvas.Divide(2, 2)
    
    canvas.cd(4)
    hist1.Draw()
    ROOT.gPad.SetLogz(1)


    canvas.cd(2)
    hist2.Draw()
    ROOT.gPad.SetLogz(1)


    canvas.cd(3)
    hist3.Draw()
    ROOT.gPad.SetLogz(1)


    canvas.cd(1)
    ROOT.gPad.SetLogz(1)
    hist4.Draw()

    canvas.Update()
    canvas.SaveAs(save_name_pdf)


def find_trees(directory, path=""):
    trees = []
    keys = directory.GetListOfKeys()
    for key in keys:
        item = key.ReadObj()
        if item.InheritsFrom("TTree"):
            trees.append(f"{path}/{key.GetName()}".lstrip('/'))
        elif item.InheritsFrom("TDirectory"):
            trees.extend(find_trees(item, f"{path}/{key.GetName()}"))
    return trees


def fit_chrystalball(hist_given:Union[ROOT.TH1F, None]=None,file_name:Union[str,None]=None, save_name_pdf:Union[str,None]=None, var:str="fMass", save_file:Union[str,None]=None,x_min_fit:float=1.086,x_max_fit:float=1.14,x_min_data:float=1.05,x_max_data:float=1.16,no_bins:int=100,title:str="crystalball+background fit",cheb:bool=False, logy:bool=True,n_sig:float=4, fixed_cbParams:bool=False,give_fittedValues:bool=False):

    ROOT.gROOT.SetBatch()
    ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.ERROR)
    ROOT.RooMsgService.instance().setSilentMode(ROOT.kTRUE)
    ROOT.gErrorIgnoreLevel = ROOT.kError
    # Create a ROOT application

    if hist_given != None:
        hist=hist_given  
    else:
        root_file = ROOT.TFile(file_name)    
        hist = ROOT.TH1F("hist", "Data", no_bins, 0,0)  # Adjust binning and range as necessary

    x = RooRealVar(var, var, x_min_fit, x_max_fit)
    fit_range = ROOT.RooFit.Range(x_min_fit, x_max_fit)
    print("min, max value for fitting: ", x_min_fit, x_max_fit)       


    if not hist_given:
        all_trees = find_trees(root_file)
        for tree_name in all_trees:
            tree = root_file.Get(tree_name)
            cut = "fCosPA > 0.999 && fMass > 1.085 && fMass < 1.140"
            temp_tree = tree.CopyTree(cut)
            temp_tree.Draw(f"{var}>>hist")
        data = RooDataHist("data_hist", "RooDataHist from TH1", RooArgList(x), hist)


        mean = RooRealVar("mean", "mean of gaussian", 1.1155, 1.115, 1.116)
        sigma = RooRealVar("sigmaLR", "width of gaussian", 0.001, 0.00001, 0.01)
        alphaL = RooRealVar("alphaL", "alphaL", 1.38, 0.1, 10)
        nL = RooRealVar("nL", "nL", 6, 0.01, 10)
        alphaR = RooRealVar("alphaR", "alphaR", 1.4, 0.1, 10)
        nR = RooRealVar("nR", "nR", 9, 0.01, 10)
        # Define the Crystal Ball function
        crystal_ball = ROOT.RooCrystalBall("crystal_ball", "Crystal Ball PDF", x, mean, sigma, alphaL, nL,alphaR, nR)

    else:
        data = RooDataHist("data_hist", "RooDataHist from TH1", RooArgList(x), hist_given)
        #mean = RooRealVar("mean", "mean of gaussian", 1.1155, 1.115, 1.116)
        #sigma = RooRealVar("sigmaLR", "width of gaussian", 0.001, 0.00001, 0.01)
        alphaL_fixed = RooRealVar("alphaL", "alphaL", 1.3197)
        nL_fixed = RooRealVar("nL", "nL", 6.7180)
        alphaR_fixed = RooRealVar("alphaR", "alphaR",  1.1630)
        nR_fixed = RooRealVar("nR", "nR", 9.6692)
        #Define the Crystal Ball function
        #crystal_ball = ROOT.RooCrystalBall("crystal_ball", "Crystal Ball PDF", x, mean, sigma, alphaL, nL,alphaR, nR)
        mean, sigma, alphaL, nL, alphaR, nR = fit_chrystalball_manuel(file_name=None, hist_given=hist_given)
        if fixed_cbParams:
            crystal_ball = ROOT.RooCrystalBall("crystal_ball", "Crystal Ball PDF", x, mean, sigma, alphaL_fixed, nL_fixed,alphaR_fixed, nR_fixed)
        else:
            crystal_ball = ROOT.RooCrystalBall("crystal_ball", "Crystal Ball PDF", x, mean, sigma, alphaL, nL,alphaR, nR)

    if not cheb:   
        if not hist_given:
            coef1 = RooRealVar("coef1", "coefficient of chrystal ball", 100000,0,100000000)
            coef2 = RooRealVar("coef2", "coefficient of quadratic", 1000,0,1000000000)
            p0 = RooRealVar("p0", "coefficient of constant term", 69,-10000,10000)
            p1 = RooRealVar("p1", "coefficient of linear term", -54,-100,10000)
            quadratic = ROOT.RooPolynomial("quadratic", "Quadratic function", x, RooArgList(p0,p1))

        else:
            coef1 = RooRealVar("coef1", "coefficient of chrystal ball",1000000,0,100000000)
            coef2 = RooRealVar("coef2", "coefficient of quadratic", 1000,0,1000000000)
            p0 = RooRealVar("p0", "coefficient of constant term", 100,-10000,10000)
            p1 = RooRealVar("p1", "coefficient of linear term", -50,-10000,10000)
            quadratic = ROOT.RooPolynomial("quadratic", "Quadratic function", x, RooArgList(p0,p1))
            #fit_range_help = ROOT.RooFit.Range(1.086, 1.1)
            #quadratic.fitTo(data, fit_range_help)
            fit_range_help2 = ROOT.RooFit.Range(1.13, 1.14)
            quadratic.fitTo(data, fit_range_help2)
        
        
    else:
        p0 = RooRealVar("p0", "coefficient of constant term", -0.115,-10000,10000)
        p1 = RooRealVar("p1", "coefficient of linear term", -0.158,-10000,1) 
        chebychev = ROOT.RooChebychev("chebychev", "chebychev polynomial",x, RooArgList(p0,p1))
        if not hist_given:
            coef1 = RooRealVar("coef1", "coefficient of chrystal ball", 10000,0,100000000)
            coef2 = RooRealVar("coef2", "coefficient of quadratic", 1000,0,1000000000)
        else:
            coef1 = RooRealVar("coef1", "coefficient of chrystal ball", 900000,0,100000000)
            coef2 = RooRealVar("coef2", "coefficient of quadratic", 9000,0,1000000000)
            fit_range_help = ROOT.RooFit.Range(1.086, 1.1)
            chebychev.fitTo(data, fit_range_help)
            fit_range_help2 = ROOT.RooFit.Range(1.13, 1.14)
            chebychev.fitTo(data, fit_range_help2)                

    # Combine the Crystal Ball function and the background function
    if not cheb:
        model = RooAddPdf("model", "Crystal Ball + Quadratic", RooArgList(crystal_ball, quadratic), RooArgList(coef1, coef2))
    else:
        model = RooAddPdf("model", "Crystal Ball + Chebychev", RooArgList(crystal_ball, chebychev), RooArgList(coef1, coef2))


    # Perform the fit
    result=model.fitTo(data, fit_range, RooFit.Save())

    if give_fittedValues:
        finalParams = result.floatParsFinal()
        return finalParams
    else:
        chi2 = model.createChi2(data).getVal()

        mean_val = result.floatParsFinal().find("mean").getVal()
        sigma_val = abs(result.floatParsFinal().find("sigmaLR").getVal())
        mean_val=mean.getVal()
        sigma_val=sigma.getVal()
        print(coef1, coef2)

        x.setRange("integrationRange", mean_val-n_sig*sigma_val, mean_val+n_sig*sigma_val)

        components = model.getComponents()
        cb_component = components.find("crystal_ball")
        if not cheb:
            bckg_component= components.find("quadratic")
        else:
            bckg_component= components.find("chebychev")

        integral_bckg = bckg_component.createIntegral(ROOT.RooArgSet(x), RooFit.Range("integrationRange"),RooFit.NormSet(ROOT.RooArgSet(x)))
        integral_signal = cb_component.createIntegral(ROOT.RooArgSet(x), RooFit.Range("integrationRange"),RooFit.NormSet(ROOT.RooArgSet(x)))

        integral_bckg_val = integral_bckg.getVal()
        integral_signal_val = integral_signal.getVal()

        purity = integral_signal_val / (integral_bckg_val*(coef2.getVal()/coef1.getVal())+integral_signal_val)

        # Plot the results
        xframe = x.frame(RooFit.Title(title))
        data.plotOn(xframe)
        model.plotOn(xframe, RooFit.LineColor(ROOT.kBlue))
        model.plotOn(xframe, RooFit.Components("crystal_ball"), RooFit.LineStyle(ROOT.kDashed), RooFit.LineColor(ROOT.kRed))
        if not cheb:
            model.plotOn(xframe,  RooFit.Components("quadratic"), RooFit.LineStyle(ROOT.kDotted), RooFit.LineColor(ROOT.kGreen))
        else:
            model.plotOn(xframe, RooFit.Components("chebychev"), RooFit.LineStyle(ROOT.kDotted), RooFit.LineColor(ROOT.kGreen))


        line1 = ROOT.TLine(mean_val+n_sig*sigma_val, xframe.GetMinimum(), mean_val+n_sig*sigma_val, xframe.GetMaximum())
        line1.SetLineColor(ROOT.kBlack)
        line1.SetLineWidth(2)
        xframe.addObject(line1)
        line2 = ROOT.TLine(mean_val-n_sig*sigma_val, xframe.GetMinimum(), mean_val-n_sig*sigma_val, xframe.GetMaximum())
        line2.SetLineColor(ROOT.kBlack)
        line2.SetLineWidth(2)
        xframe.addObject(line2)

        legend = ROOT.TLegend(0.7, 0.7, 0.95, 0.9)  # (x1, y1, x2, y2) coordinates in normalized canvas units
        legend.SetHeader("Legend")  # Optional legend title
        legend.AddEntry(xframe.findObject("data_hist"), "Data", "p")  # Add data to legend
        legend.AddEntry(xframe.findObject("model"), "Model", "l")
        legend.AddEntry(xframe.findObject("crystal_ball"), "Crystal Ball", "l")
        if not cheb:
            legend.AddEntry(xframe.findObject("quadratic"), "Quadratic", "l")
        else:
            legend.AddEntry(xframe.findObject("chebychev"), "Chebychev", "l")
        # Match legend entries with their respective line styles and colors
        legend.GetListOfPrimitives().At(1).SetMarkerStyle(ROOT.kFullCircle)  # Data marker style
        legend.GetListOfPrimitives().At(2).SetLineColor(ROOT.kBlue)  # Model line color
        legend.GetListOfPrimitives().At(3).SetLineColor(ROOT.kRed)    # Crystal Ball line color
        legend.GetListOfPrimitives().At(3).SetLineStyle(ROOT.kDashed)  # Crystal Ball line style
        legend.GetListOfPrimitives().At(4).SetLineColor(ROOT.kGreen)  # Quadratic line color
        legend.GetListOfPrimitives().At(4).SetLineStyle(ROOT.kDotted)  # Quadratic line style

        notes = ROOT.TPaveText(0.7, 0.05, 0.95, 0.65, "NDC")  # (x1, y1, x2, y2) in normalized canvas units
        notes.AddText("Fit parameter:")
        #print("Fit parameters:")
        pdf_list = model.pdfList()
        for pdf_index in range(pdf_list.getSize()):
            notes.AddText("\n")
            pdf = pdf_list.at(pdf_index)
            notes.AddText(f"{pdf.GetName()}:")
            params = pdf.getParameters(data)     
            iterator = params.createIterator()
            param = iterator.Next()
            while param:
                notes.AddText(f"{param.GetName()}: {param.getVal():.4f} \pm {param.getError():.4f}")
                param = iterator.Next()

        notes.AddText("\n")
        notes.AddText(f"Chi2: {chi2:.5f}")
        notes.AddText("\n")
        notes.AddText(f"Chi2/bin: {chi2/no_bins:.5f}")
        notes.AddText("\n")
        notes.AddText(f"Purity in {n_sig}-sig.:")
        notes.AddText(f"={integral_signal_val:.5f} / {integral_bckg_val*(coef2.getVal()/coef1.getVal())+integral_signal_val:.5f}")
        notes.AddText(f"={purity:.5f}")

        notes.SetTextAlign(12)  # 12 means left alignment and top vertical alignment
        notes.SetTextSize(0.03)  # Adjust the text size as needed
        notes.SetTextFont(42)

        notes.GetLineWith("Fit").SetTextFont(62)
        notes.GetLineWith("crys").SetTextFont(62)
        if not cheb:
            notes.GetLineWith("qua").SetTextFont(62)
        else:
            notes.GetLineWith("cheb").SetTextFont(62)
        notes.GetLineWith("Pur").SetTextFont(62)
        notes.GetLineWith("Chi2:").SetTextFont(62)


        canvas = ROOT.TCanvas("canvas", "Crystal Ball Fit", 800, 600)
        canvas.SetRightMargin(0.33)
        if logy:
            ROOT.gPad.SetLogy(1)
        xframe.Draw()
        legend.Draw()
        notes.Draw()


        canvas.SaveAs(save_name_pdf)

        if save_file:
            if os.path.exists(save_file):
                output_file = ROOT.TFile(save_file, "UPDATE")
            else:
                output_file = ROOT.TFile(save_file, "RECREATE")
            xframe.Write()
            output_file.Close()
    


def crystalball_plus_quadratic(x, params):
    # Crystal Ball parameters
    alpha_l = params[0]  # Alpha for the left side
    n_l = params[1]      # n for the left side
    alpha_r = params[2]  # Alpha for the right side
    n_r = params[3]      # n for the right side
    mu = params[4]       # Mean
    sigma = params[5]    # Standard deviation
    norm_cb = params[6]

    # Quadratic background parameters
    a = params[7]
    b = params[8]

    # Crystal Ball function
    t = (x[0] - mu) / sigma

    if t < -alpha_l:
        a_l = ROOT.TMath.Power(n_l / abs(alpha_l), n_l) * ROOT.TMath.Exp(-0.5 * alpha_l * alpha_l)
        b_l = n_l / abs(alpha_l) - abs(alpha_l)
        cb= norm_cb * a_l / ROOT.TMath.Power(b_l - t, n_l)
    elif t > alpha_r:
        a_r = ROOT.TMath.Power(n_r / abs(alpha_r), n_r) * ROOT.TMath.Exp(-0.5 * alpha_r * alpha_r)
        b_r = n_r / abs(alpha_r) - abs(alpha_r)
        cb= norm_cb * a_r / ROOT.TMath.Power(b_r + t, n_r)
    else:
        cb=norm_cb * ROOT.TMath.Exp(-0.5 * t * t)
    
    # Quadratic background function
    quad = a * x[0]**2 + b * x[0] 

    return cb + quad



def fit_chrystalball_manuel(file_name:Union[str,None]=None,tree_name:Union[str,None]=None,save_name_file:Union[str,None]=None,save_name_pdf:Union[str,None]=None,var:str="fMass",x_min:float=1.086,x_max:float=1.14,folders:bool=None,hist_given:Union[ROOT.TH1F,None]=None, logy:bool=True,save_file:bool=False):

    if not hist_given:
        root_file = ROOT.TFile(file_name)
    canvas = ROOT.TCanvas("canvas", "Fit Canvas", 800, 600)
    canvas.SetRightMargin(0.33)
    if logy:
        ROOT.gPad.SetLogy(1)

    if not hist_given:
        hist = ROOT.TH1F("hist", "Data", 100, 1.06, 1.16)  # Adjust binning and range as necessary      
        if not folders:
            tree = root_file.Get(tree_name)
            #cut = "fCosPA > 0.999"
            cut = "fCosPA > 0.999 && fMass > 1.085 && fMass < 1.140"
            temp_tree = tree.CopyTree(cut)
            temp_tree.Draw(f"{var}>>hist")
        else: 
            all_trees = find_trees(root_file)
            i=0
            for tree_name in all_trees:
                tree = root_file.Get(tree_name)
                #cut = "fCosPA > 0.999"
                cut = "fCosPA > 0.999 && fMass > 1.085 && fMass < 1.140"
                temp_tree = tree.CopyTree(cut)
                temp_tree.Draw(f"{var}>>hist")
    
    else:
        hist=hist_given
    n_params = 9
    combined_function = ROOT.TF1("combined_function", crystalball_plus_quadratic, 1.086, 1.14, n_params)
    combined_function.SetParameters(2, 1.5, 2, 1.2, 1.1155, 0.002, 400000, -1.0, 5.0)
    combined_function.SetParNames("alphaL", "nL","alphaR", "nR", "mean", "sigmaLR", "norm_cb", "a", "b")


    hist.Fit(combined_function, "R")

    combined_function.Draw("SAME")
    #consfunc.Draw("Same")
    fit_results = combined_function.GetParameters()
    fit_errors = combined_function.GetParErrors()

    #    Print the results
    for i in range(n_params):
        print(f"Parameter {i}: {fit_results[i]} ± {fit_errors[i]}")
    if save_name_pdf:
        canvas.SaveAs(save_name_pdf)

    mean = RooRealVar("mean", "mean of gaussian", fit_results[4], fit_results[4]-fit_results[5],fit_results[4]+fit_results[5])
    sigma = RooRealVar("sigmaLR", "width of gaussian", fit_results[5],0,1)
    alphaL = RooRealVar("alphaL", "alphaL", fit_results[0])
    nL = RooRealVar("nL", "nL", fit_results[1])
    alphaR = RooRealVar("alphaR", "alphaR", fit_results[2])
    nR = RooRealVar("nR", "nR", fit_results[3])
    p0 = RooRealVar("p0", "coefficient of linear term", fit_results[8]*10000/fit_results[6])
    p1 = RooRealVar("p1", "coefficient of quadratic term", fit_results[7]*10000/fit_results[6])



    if save_file:
        if os.path.exists(save_name_file):
            output_file = ROOT.TFile(save_name_file, "UPDATE")
        else:
            output_file = ROOT.TFile(save_name_file, "RECREATE")
        hist.Write()
        output_file.Close()
    
    return mean, sigma, alphaL, nL, alphaR, nR



def create_1d_histograms_from_2d(file_name:str, hist_name:str, already_saved:bool=False)->Sequence[ROOT.TH1F]:
    # Get the number of bins in y-axis from the original histogram
    file = ROOT.TFile(file_name)
    original_hist2d=file.Get(hist_name)
    y_bins = original_hist2d.GetNbinsY()

    if not already_saved:

        # Get the x-axis properties from the original histogram
        x_bins = original_hist2d.GetNbinsX()
        x_min = original_hist2d.GetXaxis().GetXmin()
        x_max = original_hist2d.GetXaxis().GetXmax()

        # Create a list to hold the 1D histograms
        histograms = []

        # Loop over each y-bin and create a corresponding 1D histogram
        for y_bin in range(1, y_bins + 1):
            y_bin_low_edge = original_hist2d.GetYaxis().GetBinLowEdge(y_bin)
            y_bin_up_edge = original_hist2d.GetYaxis().GetBinUpEdge(y_bin)
            hist_name = f"hist_ybin_{y_bin}"
            hist_title = f"1D Projection for Y Bin [{y_bin_low_edge}, {y_bin_up_edge}]"
            hist = ROOT.TH1F(hist_name, hist_title, x_bins, x_min, x_max)

            # Fill the 1D histogram with the x-values for this y-bin
            for x_bin in range(1, x_bins + 1):
                bin_content = original_hist2d.GetBinContent(x_bin, y_bin)
                bin_error = original_hist2d.GetBinError(x_bin, y_bin)
                hist.SetBinContent(x_bin, bin_content)
                #hist.SetBinError(x_bin, bin_error)

            # Append the 1D histogram to the list
            histograms.append(hist)
            output_file = ROOT.TFile(file_name, "UPDATE")
            hist.Write()
            output_file.Close()
    else:
        histograms=[]
        for i in range(1,y_bins+1):
            hist=file.Get(f"hist_ybin_{i}")
            print(hist)
            histograms.append(hist)
    print(histograms)
    return histograms


def new_bin_edges(file_name:str, new_bins:Sequence[float],hist_name:str,reb_y:bool=False):

    file=ROOT.TFile.Open(file_name)
    original_hist=file.Get(hist_name)
    new_bins=np.array(new_bins)
    if reb_y:
        x_bins = original_hist.GetXaxis().GetNbins()
        x_min = original_hist.GetXaxis().GetXmin()
        x_max = original_hist.GetXaxis().GetXmax()
        rebinned_hist = ROOT.TH2F("rebinned_hist", "Rebinned Y-Axis 2D Histogram",x_bins, x_min, x_max, len(new_bins) - 1, new_bins)
    else:
        y_bins = original_hist.GetYaxis().GetNbins()
        y_min = original_hist.GetYaxis().GetXmin()
        y_max = original_hist.GetYaxis().GetXmax()
        rebinned_hist = ROOT.TH2F("rebinned_hist", "Rebinned X-Axis 2D Histogram",len(new_bins) - 1, new_bins, y_bins, y_min, y_max)

    # Fill the new histogram with content from the original histogram
    for i in range(1, original_hist.GetNbinsX() + 1):
        for j in range(1, original_hist.GetNbinsY() + 1):
            x_bin_center = original_hist.GetXaxis().GetBinCenter(i)
            y_bin_center = original_hist.GetYaxis().GetBinCenter(j)
            bin_content = original_hist.GetBinContent(i, j)
            bin_error = original_hist.GetBinError(i, j)
            rebinned_hist.Fill(x_bin_center, y_bin_center, bin_content)
            new_bin = rebinned_hist.FindBin(x_bin_center, y_bin_center)
            rebinned_hist.SetBinError(new_bin, np.sqrt(rebinned_hist.GetBinError(new_bin)**2 + bin_error**2))

    output_file = ROOT.TFile(file_name, "UPDATE")
    rebinned_hist.Write()
    output_file.Close()

    

def get_minmax_of_tree(file_name:str, branch1:str):
    file = ROOT.TFile(file_name)
    all_trees = find_trees(file)
    min_val=float('inf')
    max_val=-float("inf")
    for tree_name in all_trees:
        tree_pre = file.Get(tree_name)
        tree=tree_pre.CopyTree(f"{branch1} > 0 ")
        if tree.GetMinimum(branch1)<min_val:
            min_val = tree.GetMinimum(branch1)
        if tree.GetMaximum(branch1)>max_val:
            max_val = tree.GetMaximum(branch1)
    #print("min: ", min_val)
    #print("max: ", max_val)
    return min_val,max_val



def fit_histogram(file_data:str, hist_data:str, file_model:str, hists_model:Sequence[str], save_name_file:Union[str,None]=None, title:str="title", logy:bool=True, save_name_pdf:str="distribution_fit.pdf"):
    
    #Open Files
    filedata = ROOT.TFile.Open(file_data)
    filemodel = ROOT.TFile.Open(file_model)

    # Get Histograms
    data_hist = filedata.Get(hist_data)
    model_hist1= filemodel.Get(hists_model[0])
    model_hist2= filemodel.Get(hists_model[1])
    model_hist3= filemodel.Get(hists_model[2])


    # Define the observable
    x = RooRealVar("x", "x", 0,1)

    if not isinstance(data_hist, ROOT.TH1F):
        raise TypeError(f"Expected a ROOT.TH1F histogram for data, but got {type(data_hist)}")
    # Convert histogram to RooDataHist
    data_rdh = RooDataHist("data_hist", "data_hist",  RooArgList(x), data_hist)
    model_rdh_class0 = RooDataHist("model_rdh1", "model_rdh1", RooArgList(x), model_hist1)
    model_rdh_class1 = RooDataHist("model_rdh2", "model_rdh2", RooArgList(x), model_hist2)
    model_rdh_class2 = RooDataHist("model_rdh3", "model_rdh3", RooArgList(x), model_hist3)

    # Create RooHistPdf from each model RooDataHist
    model_pdf_class0 = ROOT.RooHistPdf("model_pdf1", "model_pdf1", ROOT.RooArgSet(x), model_rdh_class0)
    model_pdf_class1 = ROOT.RooHistPdf("model_pdf2", "model_pdf2", ROOT.RooArgSet(x), model_rdh_class1)
    model_pdf_class2 = ROOT.RooHistPdf("model_pdf3", "model_pdf3", ROOT.RooArgSet(x), model_rdh_class2)

    # Define a normalization factor
    norm_class0 = RooRealVar("norm_class0", "normalization factor 1", 1.0, 0.0, 100000000.0)
    norm_class1 = RooRealVar("norm_class1", "normalization factor 2", 1.0, 0.0, 100000000.0)
    norm_class2 = RooRealVar("norm_class2", "normalization factor 3", 1.0, 0.0, 100000000.0)

    # Create extended PDFs by combining each model PDF with its normalization factor
    ext_pdf_class0 = ROOT.RooExtendPdf("ext_pdf_class0", "extended pdf 1", model_pdf_class0, norm_class0)
    ext_pdf_class1 = ROOT.RooExtendPdf("ext_pdf_class1", "extended pdf 2", model_pdf_class1, norm_class1)
    ext_pdf_class2 = ROOT.RooExtendPdf("ext_pdf_class2", "extended pdf 3", model_pdf_class2, norm_class2)

    total_pdf = RooAddPdf("total_pdf", "sum of extended pdfs", RooArgList(ext_pdf_class0, ext_pdf_class1, ext_pdf_class2))

    # Fit the extended model to the data
    result = total_pdf.fitTo(data_rdh, RooFit.Save())
#

    xframe = x.frame(RooFit.Title(title))
    data_rdh.plotOn(xframe)
    total_pdf.plotOn(xframe, RooFit.LineColor(ROOT.kBlue))
    total_pdf.plotOn(xframe, RooFit.Components("ext_pdf_class0"), RooFit.LineStyle(ROOT.kDashed), RooFit.LineColor(ROOT.kRed))
    total_pdf.plotOn(xframe, RooFit.Components("ext_pdf_class1"), RooFit.LineStyle(ROOT.kDashed), RooFit.LineColor(ROOT.kGreen))
    total_pdf.plotOn(xframe, RooFit.Components("ext_pdf_class2"), RooFit.LineStyle(ROOT.kDashed), RooFit.LineColor(ROOT.kOrange))
    legend = ROOT.TLegend(0.7, 0.7, 0.95, 0.9)  # (x1, y1, x2, y2) coordinates in normalized canvas units
    legend.SetHeader("Legend")  # Optional legend title
    legend.AddEntry(data_rdh, "Data", "p")  # Add data to legend
    legend.AddEntry(total_pdf, "Total", "L")  # Add data to legend
    legend.AddEntry(ext_pdf_class0, "Class0", "L")  # Add data to legend
    legend.AddEntry(ext_pdf_class1, "Class1", "L")  # Add data to legend
    legend.AddEntry(ext_pdf_class2, "Class2", "L")  # Add data to legend
    # Match legend entries with their respective line styles and colors
    legend.GetListOfPrimitives().At(2).SetMarkerStyle(ROOT.kDotted)  # Data marker style
    legend.GetListOfPrimitives().At(2).SetLineColor(ROOT.kBlue)  # Data marker style
    legend.GetListOfPrimitives().At(3).SetLineColor(ROOT.kRed)  # Data marker style
    legend.GetListOfPrimitives().At(4).SetLineColor(ROOT.kGreen)  # Data marker style
    legend.GetListOfPrimitives().At(5).SetLineColor(ROOT.kOrange)  # Data marker style
    fit_parameters = result.floatParsFinal()
    notes = ROOT.TPaveText(0.7, 0.05, 0.95, 0.65, "NDC")  # (x1, y1, x2, y2) in normalized canvas units
    notes.AddText("Fit parameter:")
    for i in range(fit_parameters.getSize()):
        param = fit_parameters.at(i)
        notes.AddText(f"{param.GetName()}:")
        notes.AddText(f"{param.getVal():.1e} \pm {param.getError():.1e}")
        notes.AddText("\n")
    notes.GetLineWith("Fit").SetTextFont(62)
    notes.SetTextAlign(12)  # 12 means left alignment and top vertical alignment
    notes.SetTextSize(0.03)  # Adjust the text size as needed
    notes.SetTextFont(42)
    canvas = ROOT.TCanvas("canvas", "Fitted Distribution", 800, 600)
    canvas.SetRightMargin(0.33)
    if logy:
        ROOT.gPad.SetLogy(1)
    xframe.Draw()
    legend.Draw()
    notes.Draw()
    canvas.SaveAs(save_name_pdf)
    if save_name_file:
        if os.path.exists(save_name_file):
            output_file = ROOT.TFile(save_name_file, "UPDATE")
        else:
            output_file = ROOT.TFile(save_name_file, "RECREATE")
        xframe.Write()
        output_file.Close()



def fit_simultanously(file_hist_data:str, hist_bdt:str, hist_mass:str, hists_model:Sequence[str], file_model:str, save_name_file:Union[str,None]=None):
    
    ROOT.gROOT.SetBatch()
    ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.ERROR)
    ROOT.RooMsgService.instance().setSilentMode(ROOT.kTRUE)
    ROOT.gErrorIgnoreLevel = ROOT.kError
    # Create a ROOT application
     
    hist_file=ROOT.TFile(file_hist_data)
    histbdt = hist_file.Get(hist_bdt) # Adjust binning and range as necessary
    histmass = hist_file.Get(hist_mass) # Adjust binning and range as necessary

    if histmass is None:
        print("Error: 'hist_mass' not found in the file.")
    else:
        if isinstance(histmass, ROOT.TH1F):
            print("Mass Histogram successfully retrieved and is of type TH1F.")
        else:
            print("The retrieved mass object is not of type TH1F.")

    if isinstance(histbdt, ROOT.TH1F):
        print("bdt Histogram successfully retrieved and is of type TH1F.")
    else:
        print("The retrieved bdt object is not of type TH1F.")
    histbdt.SetName("histbdt")
    histmass.SetName("histmass")

    print(histmass.GetEntries())
    print(histbdt.GetEntries())

    xMass = ROOT.RooRealVar("xMass", "Mass", histmass.GetXaxis().GetXmin(), histmass.GetXaxis().GetXmax())  
    xBdt = ROOT.RooRealVar("xBdt", "Bdt class 2",histbdt.GetXaxis().GetXmin(),histbdt.GetXaxis().GetXmax())

    rdhBdt = ROOT.RooDataHist("rdhBdt", "RooDataHist for BDT", ROOT.RooArgList(xBdt), ROOT.RooFit.Import(histbdt))
    rdhMass = ROOT.RooDataHist("rdhMass", "RooDataHist for Mass", ROOT.RooArgList(xMass), ROOT.RooFit.Import(histmass))

    norm_signal = RooRealVar("norm_signal", "Signal Norm", 20000000,0,100000000000)
    norm_class0 = RooRealVar("norm_class0", "Bckg Norm", 3000,0,1000000000)    
    frac_p=RooRealVar("frac_p", "p-np ratio",0.8, 0, 1)

    print("rdhBbdtentries:", rdhBdt.sumEntries())
    print("rdhMass entries:", rdhMass.sumEntries())

    mean = RooRealVar("mean", "mean of gaussian", 1.1155,1.114,1.116)
    sigma = RooRealVar("sigmaLR", "width of gaussian", 0.0013,0,1)
    alphaL = RooRealVar("alphaL", "alphaL", 1.4178,0,2)
    nL = RooRealVar("nL", "nL", 6.7769,1,10)
    alphaR = RooRealVar("alphaR", "alphaR", 1.43941,10)
    nR = RooRealVar("nR", "nR", 10,1,11)
    p0= RooRealVar("p0", "p0", -0.1311,-10,1)
    p1= RooRealVar("p1", "p1", -0.1267,-10,0)

    crystalball = ROOT.RooCrystalBall("crystalball", "Crystal Ball PDF", xMass, mean, sigma, alphaL, nL,alphaR, nR)
    chebychev = ROOT.RooChebychev("chebychev", "chebychev polynomial", xMass, RooArgList(p0,p1))

    model_mass = RooAddPdf("model", "Crystal Ball + Chebychev", RooArgList(crystalball, chebychev), RooArgList(norm_signal, norm_class0))

    model_mass.fitTo(rdhMass)

    mean.setConstant(True)
    sigma.setConstant(True)
    alphaL.setConstant(True)
    nL.setConstant(True)
    alphaR.setConstant(True)
    nR.setConstant(True)


    filemodel = ROOT.TFile.Open(file_model)

    model_hist1= filemodel.Get(hists_model[0])
    model_hist2= filemodel.Get(hists_model[1])
    model_hist3= filemodel.Get(hists_model[2])

    # Convert histogram to RooDataHist
    model_rdh_class0 = RooDataHist("model_rdh1", "model_rdh1", RooArgList(xBdt), model_hist1)
    model_rdh_class1 = RooDataHist("model_rdh2", "model_rdh2", RooArgList(xBdt), model_hist2)
    model_rdh_class2 = RooDataHist("model_rdh3", "model_rdh3", RooArgList(xBdt), model_hist3)

    # Create RooHistPdf from each model RooDataHist
    model_pdf_class0 = ROOT.RooHistPdf("model_pdf_class0", "model_pdf1", ROOT.RooArgSet(xBdt), model_rdh_class0)
    model_pdf_class1 = ROOT.RooHistPdf("model_pdf2", "model_pdf2", ROOT.RooArgSet(xBdt), model_rdh_class1)
    model_pdf_class2 = ROOT.RooHistPdf("model_pdf3", "model_pdf3", ROOT.RooArgSet(xBdt), model_rdh_class2)

    # Define a normalization factor between prompt and non-prompt
    signal_model = RooAddPdf("signal_model", "Prompt+Nonprompt",  RooArgList(model_pdf_class2, model_pdf_class1), RooArgList(frac_p))
    model_dist_fit= RooAddPdf("total_pdf", "sum of extended pdfs", RooArgList(model_pdf_class0, signal_model), RooArgList(norm_class0, norm_signal))

    ## Define combined Data
    sample = ROOT.RooCategory("sample","sample")
    sample.defineType("bdt")
    sample.defineType("mass")
    combData = ROOT.RooDataHist("combData", "Combined data", ROOT.RooArgList(sample, xBdt, xMass))

    # Add entries to combined RooDataHist
    for i in range(1, histmass.GetNbinsX() + 1):
        xMass.setVal(histmass.GetXaxis().GetBinCenter(i))
        sample.setIndex(1)  # Set category index for mass
        combData.add(ROOT.RooArgSet(sample, xMass), histmass.GetBinContent(i))
#
    for j in range(1, histbdt.GetNbinsX() + 1):
        xBdt.setVal(histbdt.GetXaxis().GetBinCenter(j))
        sample.setIndex(0)  # Set category index for bdt
        combData.add(ROOT.RooArgSet(sample, xBdt), histbdt.GetBinContent(j))
    
    # Define simultaneous fit
    simPdf = ROOT.RooSimultaneous("simPdf","simPdf", sample)
    simPdf.addPdf(model_dist_fit, "bdt")
    simPdf.addPdf(model_mass, "mass")
    result_combined=simPdf.fitTo(combData,ROOT.RooFit.Save())
    print(result_combined.status())

    mass_pdf = simPdf.getPdf("mass")
    bdt_pdf = simPdf.getPdf("bdt")

    legend_mass = ROOT.TLegend(0.7, 0.7, 0.95, 0.9)  # (x1, y1, x2, y2) coordinates in normalized canvas units
    legend_mass.SetHeader("Legend")  # Optional legend title
    legend_mass.AddEntry(rdhMass, "Data", "p")  # Add data to legend
    legend_mass.AddEntry(mass_pdf, "Total", "L")  # Add data to legend
    legend_mass.AddEntry(crystalball, "Signal", "L")  # Add data to legend
    legend_mass.AddEntry(chebychev, "Background", "L")  # Add data to legend

    ## Match legend entries with their respective line styles and colors
    legend_mass.GetListOfPrimitives().At(1).SetMarkerStyle(ROOT.kDotted)  # Data marker style
    legend_mass.GetListOfPrimitives().At(2).SetLineColor(ROOT.kBlue)  # Data marker style
    legend_mass.GetListOfPrimitives().At(3).SetLineColor(ROOT.kGreen)  # Data marker style
    legend_mass.GetListOfPrimitives().At(3).SetLineStyle(ROOT.kDashed)  # Data marker style
    legend_mass.GetListOfPrimitives().At(4).SetLineColor(ROOT.kRed)  # Data marker style
    legend_mass.GetListOfPrimitives().At(4).SetLineStyle(ROOT.kDashed)  # Data marker style


    canvas = ROOT.TCanvas("canvas", "Fitted Distribution", 1000, 600)
    canvas.Divide(2,1)


    canvas.cd(1)
    ROOT.gPad.SetLogy(1)
    xframe = xMass.frame(RooFit.Title("invariant Mass"))
    rdhMass.plotOn(xframe)
    mass_pdf.plotOn(xframe)
    mass_pdf.plotOn(xframe, RooFit.Components("chebychev"), RooFit.LineStyle(ROOT.kDashed), RooFit.LineColor(ROOT.kRed))
    mass_pdf.plotOn(xframe, RooFit.Components("crystalball"), RooFit.LineStyle(ROOT.kDashed), RooFit.LineColor(ROOT.kGreen))

    xframe.Draw()
    legend_mass.Draw()

    canvas.cd(2)
    canvas.SetRightMargin(0.33)
    ROOT.gPad.SetLogy(1)
    fit_parameters = result_combined.floatParsFinal()
    notes = ROOT.TPaveText(0.15, 0.8, 0.7, 0.9, "NDC")  # (x1, y1, x2, y2) in normalized canvas units
    notes.AddText("Fit parameter:")
    for i in range(fit_parameters.getSize()):
        param = fit_parameters.at(i)
        if param.GetName()=="norm_signal" or param.GetName()=="norm_class0" or param.GetName()=="frac_p":
            notes.AddText(str(param.GetTitle())+f": {param.getVal():.2e} \pm {param.getError():.2e}")
            #notes.AddText(f"{param.getVal():.2e} \pm {param.getError():.2e}")
    notes.GetLineWith("Fit").SetTextFont(62)
    notes.SetTextAlign(12)  # 12 means left alignment and top vertical alignment
    notes.SetTextSize(0.03)  # Adjust the text size as needed
    notes.SetTextFont(42)
    xframe_bdt = xBdt.frame(RooFit.Title("BDT score class 2"))
    rdhBdt.plotOn(xframe_bdt)
    bdt_pdf.plotOn(xframe_bdt)
    bdt_pdf.plotOn(xframe_bdt, RooFit.Components("model_pdf_class0"), RooFit.LineStyle(ROOT.kDashed), RooFit.LineColor(ROOT.kRed))
    bdt_pdf.plotOn(xframe_bdt, RooFit.Components("signal_model"), RooFit.LineStyle(ROOT.kDashed), RooFit.LineColor(ROOT.kGreen))    
    xframe_bdt.Draw()
    notes.Draw()

    canvas.Update()
    canvas.SaveAs(save_name_file)

    Params=result_combined.floatParsFinal()

    print("Result:")
    for i in range(len(Params)):
        param = Params[i]
        print(f"Parameter: {param.GetName()} Value: {param.getVal()} Error: {param.getError()}")

    canvas.Close()
    if save_name_file:
        if os.path.exists(save_name_file):
            output_file = ROOT.TFile(save_name_file, "UPDATE")
        else:
            output_file = ROOT.TFile(save_name_file, "RECREATE")
        xframe.Write()
        output_file.Close()


def scans_bdt(vals:Sequence[float], prompt:TreeHandler, nonprompt:TreeHandler, var:str="trainBckgMC_class2", uppercuts:bool=True, base:Union[float,None]=None):
    ratio={}
    if uppercuts:
        if base:
            for val in vals:
                prompt_cutted=cut_data(data=prompt, var=var, upper_cut=val, lower_cut=base)
                nonprompt_cutted=cut_data(data=nonprompt, var=var, upper_cut=val, lower_cut=base)
                ratio[val]=nonprompt_cutted.get_n_cand()/prompt_cutted.get_n_cand()
        else:
            for val in vals:
                prompt_cutted=cut_data(data=prompt, var=var, upper_cut=val)
                nonprompt_cutted=cut_data(data=nonprompt, var=var, upper_cut=val)
                ratio[val]=nonprompt_cutted.get_n_cand()/prompt_cutted.get_n_cand()
    else:
        if base:
            for val in vals:
                prompt_cutted=cut_data(data=prompt, var=var, upper_cut=base, lower_cut=val)
                nonprompt_cutted=cut_data(data=nonprompt, var=var, upper_cut=base, lower_cut=val)
                ratio[val]=nonprompt_cutted.get_n_cand()/prompt_cutted.get_n_cand()
        else:
            for val in vals:
                prompt_cutted=cut_data(data=prompt, var=var, lower_cut=val)
                nonprompt_cutted=cut_data(data=nonprompt, var=var, lower_cut=val)
                ratio[val]=nonprompt_cutted.get_n_cand()/prompt_cutted.get_n_cand()

    return ratio
