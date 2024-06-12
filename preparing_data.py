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
import ROOT


def get_rawdata(file_directory_data, tree_data)->TreeHandler:
    data=TreeHandler(file_directory_data, tree_data)
    return data

def get_rawMC(file_directory_mc, tree_mc)->TreeHandler:
    data_mc=TreeHandler(file_directory_mc, tree_mc)
    raw_mc=data_mc.get_subset("fMass!=-999")
    return raw_mc

def get_bckg(file_directory_data, tree_data,cuts)->TreeHandler:
    data=TreeHandler(file_directory_data, tree_data)
    bckg=data.get_subset(f'fMass<{cuts[0]} or fMass>{cuts[1]} and fCosPA!=-999')
    return bckg

def get_prompt(file_directory_mc, tree_mc, cuts)->TreeHandler:
    data_mc=TreeHandler(file_directory_mc, tree_mc)
    prompt = data_mc.get_subset(f'fPDGCodeMother==0 and (fMass>{cuts[0]} and fMass<{cuts[1]}) and fCosPA!=-999')
    return prompt

def get_nonprompt(file_directory_mc, tree_mc, cuts)->TreeHandler:
    data_mc=TreeHandler(file_directory_mc, tree_mc)
    nonprompt = data_mc.get_subset(f'not fPDGCodeMother==0 and (fMass>{cuts[0]} and fMass<{cuts[1]}) and fCosPA!=-999')
    return nonprompt

def plot_dist(to_plot:Union[Sequence[TreeHandler],TreeHandler],vars_to_draw:Union[Sequence[str],str],leg_labels:Union[Sequence[str],str,None]=None, fs:Union[tuple,None]=(10,7), alpha:Union[float, None]=0.3):
    if type(to_plot)!=list:
        to_plot=[to_plot]
    if type(leg_labels)==str:
        leg_labels=[leg_labels]
    plot_utils.plot_distr(to_plot, vars_to_draw, bins=100, labels=leg_labels, log=True, density=True, figsize=fs, alpha=alpha, grid=False)
    plt.subplots_adjust(left=0.06, bottom=0.06, right=0.99, top=0.96, hspace=0.55, wspace=0.55)

def plot_correl(to_plot:Union[Sequence[TreeHandler],TreeHandler],vars_to_draw:Union[Sequence[str],str], leg_labels:Union[Sequence[str],None]=None):
    if type(to_plot)!=list:
        to_plot=[to_plot]
    plot_utils.plot_corr(to_plot, vars_to_draw, leg_labels)

def var_draw_all(file_directory_mc, tree_mc,file_directory_data, tree_data)->list:
    data_mc=TreeHandler(file_directory_mc, tree_mc)
    data=TreeHandler(file_directory_data, tree_data)
    THs=[data,data_mc]
    return [dat for dat in THs[0].get_var_names() if all(dat in entry.get_var_names() for entry in THs)]
    
def gauss(x: np.ndarray, a: float, mu: float, sigma: float) -> np.ndarray:
    return a * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

def fit_gauss(data:ROOT.RDataFrame, var:str, no_bins:int=100, fitting_range:list=[45,55] ,p0:Union[Sequence[float],None]=None):
    dataArr=data.to_numpy()[var]
    hist, bin = np.histogram(dataArr, bins=no_bins)
    if p0:
        par,unc=optimize.curve_fit(gauss,  bin[fitting_range[0]+1:fitting_range[1]+1], hist[fitting_range[0]:fitting_range[1]],p0=(p0[0],p0[1],p0[2]))
    else: 
        par,unc=optimize.curve_fit(gauss,  bin[fitting_range[0]+1:fitting_range[1]+1], hist[fitting_range[0]:fitting_range[1]])
    print("fitting params= ",par)
    cuts=[par[1]-5*abs(par[2]),par[1]+5*abs(par[2])]
    return par, unc, cuts, fitting_range

def plot_gauss(data:ROOT.RDataFrame,par:list ,cuts:tuple, no_bins:int=100 ,fitting_range:list=[45,55], fs:Union[tuple,None]=(10,7), title:Union[str, None]=None):


    dataArr=data.AsNumpy(columns=["fMass"])["fMass"]
    fig, ax=plt.subplots(figsize=fs)
    fig.suptitle(title)
    hist, bins = np.histogram(dataArr, bins=no_bins)
    ax.plot(bins[1:],hist, label="data")
    ax.plot([cuts[0],cuts[0]],[0, max(hist)],color="red")
    ax.plot([cuts[1],cuts[1]],[0, max(hist)],color="red",label="cuts")
    ax.plot(bins[fitting_range[0]:fitting_range[1]], gauss(bins[fitting_range[0]:fitting_range[1]],*par),label="fit")
    ax.set_yscale("log")
    ax.legend()
    ax.set_ylabel("counts")
    plt.subplots_adjust(left=0.06, bottom=0.06, right=0.99, top=0.96, hspace=0.55, wspace=0.55)

