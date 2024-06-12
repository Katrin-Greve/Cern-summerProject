import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from hipe4ml.model_handler import ModelHandler
from hipe4ml.tree_handler import TreeHandler
from hipe4ml.analysis_utils import train_test_generator
from hipe4ml import plot_utils
from typing import Union, Sequence
from scipy import optimize
import ROOT


file_directory_data="/home/katrin/Cern_Summer/AnalysisResults_treesML_data.root"
tree_data="O2lambdatableml"
file_directory_mc="/home/katrin/Cern_Summer/AnalysisResults_treesML.root"
tree_mc="O2mclambdatableml"


data=ROOT.RDataFrame(tree_data,file_directory_data)
dataArr=data.AsNumpy(columns=["fMass"])["fMass"]
hist, bins = np.histogram(dataArr, bins=100)
print(hist,bins)



def gauss(x: np.ndarray, a: float, mu: float, sigma: float, offset:float) -> np.ndarray:
    return a * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))+offset

par,unc=optimize.curve_fit(gauss,  bins[46:56], hist[45:55],p0=(1000, 1.115, 0.005,1))
print(par)

plt.plot(bins[1:],hist)
plt.plot(bins[45:55], gauss(bins[45:55],*par))
plt.show()
