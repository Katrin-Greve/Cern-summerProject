import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hipe4ml.tree_handler import TreeHandler
from hipe4ml import plot_utils
from typing import Union, Sequence,List, Tuple
from scipy import optimize
import ROOT
import uproot
import awkward as ak
import PyPDF2
import os
import seaborn as sns

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
    data=TreeHandler(file_directory_data, tree_data, folder_name=folder_name)
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

def add_GenRadius(data:TreeHandler)->TreeHandler:

    df=data.get_data_frame()
    M=1.11568
    Pt=TreetoArray(data, "fGenPt")
    Ct=TreetoArray(data, "fGenCt")
    L=(Ct*Pt)/M
    df["fGenRadius"]=np.abs(L)
    tr=TreeHandler()
    tr.set_data_frame(df)
    return tr

def add_Radius(data:TreeHandler)->TreeHandler:

    df=data.get_data_frame()
    M=TreetoArray(data, "fMass")
    Pt=TreetoArray(data, "fPt")
    Ct=TreetoArray(data, "fCt")
    L=(Ct*Pt)/M
    df["fRadius_calc"]=np.abs(L)
    tr=TreeHandler()
    tr.set_data_frame(df)
    return tr

def filter_data(data:TreeHandler, var:Union[str,Sequence[str]], values:Union[float, Sequence[float]], equal:Union[bool,Sequence[bool]]=[True])->TreeHandler:

    for vr,vl, eq in zip(var,values, equal):
        if eq:
            data=data.get_subset(f'{vr}=={vl}')
        else:
            data=data.get_subset(f'{vr}!={vl}')
    return data


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

# Function to save the desired subsets
def save_sets(sets:Sequence[TreeHandler],set_names:Sequence[str], dir_tree:str,tree:str="tree"):
    '''
    Parameters:
        sets(Sequence[TreeHandler]): sets, to eb stored
        set_names (Sequence[str]): names of the sets
        dir_tree (str): direction, in which the output .root files will be saved
        tree (str): Name of the tree in which the output will be saved. Default to "tree"
        no_set (int): no. of used data set.
    '''
    if type(sets)!=list:
        sets=[sets]
    if type(set_names)!=list:
        set_names=set_names
    for (i,name) in zip(sets,set_names):
        get_root_from_TreeHandler(treehdl=i, save_dir=dir_tree, output_name=f"{name}.root",treename=tree)


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
def plot_hist(to_plot:Union[TreeHandler,Sequence[TreeHandler]], vars_to_draw:Sequence[str], leg_labels:Union[Sequence[str],str], no_bins:int=100,  fs:Union[tuple,None]=(10,7), alpha:Union[float, None]=0.3, colors=None):
    
    plot_utils.plot_distr(to_plot, vars_to_draw, bins=no_bins, labels=leg_labels, log=True, density=True, figsize=fs, alpha=alpha, grid=False,colors=colors)
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
def plot_2dhist_numpy(data:TreeHandler, var1:str, var2:str,ax, bin:int=100, cmap:str="rainbow"):
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
    hist, binsx,binsy=np.histogram2d(dataArr1,dataArr2, bins=bin)
    hist=hist.T

    ax.pcolormesh(binsx, binsy, hist, cmap=cmap)
    ax.set_xlim(binsx.min(), binsx.max())
    ax.set_ylim(binsy.min(), binsy.max())


# Function to draw and save a 2d histogram using ROOT
def plot_2dhist_root(file:str, tree_name:str,  var1:str, var2:str, save_name_file:str, hist_name:str, save_name_pdf:str, title:str ,bins:int=7,cmap=ROOT.kRainbow):
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
    file = ROOT.TFile(file, "READ")
    tree = file.Get(tree_name)
    hist = ROOT.TH2F(hist_name, title, bins, 0,0, bins, 0,0)

    tree.Draw(f"{var2}:{var1} >> {hist_name}")

    print(f"ok, tree draws {var1}, {var2}")
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
    ROOT.gPad.SetLogz(1)

    # Save the canvas as an image file if needed
    canvas.SaveAs(save_name_pdf)
    canvas.Close()
    # Create a new ROOT file to save the histogram
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

# Function to merge several PDFs in one PDF file
def merge_pdfs(pdf_list:Sequence[str], output_path:str):
    """
    Parameters:
        pdf_list (Sequence[str]): List that contains all PDFs that should be merged
        output_path (str): Name of the merged PDF that will be saved
    """
    merger = PyPDF2.PdfMerger()

    for pdf in pdf_list:
        merger.append(pdf)

    merger.write(output_path)
    merger.close()

    # Remove individual PDF files
    for pdf in pdf_list:
        os.remove(pdf)


    