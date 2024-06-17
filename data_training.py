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
import preparing_data as prep
from matplotlib.backends.backend_pdf import PdfPages


save_dir="/home/katrin/Cern_summerProject/imgs/ml_plots/"
no_set=2

if no_set==3:
    file_directory_data="/home/katrin/Cern_summerProject/data/AO2D_data.root"
    tree_data="O2lambdatableml"
    file_directory_mc="/home/katrin/Cern_summerProject/data/AO2D_MC.root"
    tree_mc="O2mclambdatableml"
    fname="DF*"


if no_set==2:
    file_directory_data="/home/katrin/Cern_summerProject/data/AnalysisResults_treesML_data_LHC22o_apass6_small.root"
    tree_data="O2lambdatableml"
    file_directory_mc="/home/katrin/Cern_summerProject/data/AnalysisResults_treesML_mc_LHC24b1b_small.root"
    tree_mc="O2mclambdatableml"
    fname=None

if no_set==1:
    file_directory_data="/home/katrin/Cern_summerProject/data/AnalysisResults_treesML_data.root"
    tree_data="O2lambdatableml"
    file_directory_mc="/home/katrin/Cern_summerProject/data/AnalysisResults_treesML.root"
    tree_mc="O2mclambdatableml"
    fname=None


def get_traindata():
    prompt=prep.proton_pion_division(prep.get_prompt(file_directory_mc, tree_mc, folder_name=fname))
    nonprompt=prep.proton_pion_division(prep.get_nonprompt(file_directory_mc, tree_mc, folder_name=fname))
    cu=prep.fit_gauss_rec(prep.get_rawdata(file_directory_data, tree_data, folder_name=fname), var="fMass",p0=[300000,1.115,0.005,1000])[2]
    bckg=prep.proton_pion_division(prep.get_bckg(file_directory_data, tree_data, cu, folder_name=fname))
    bckg_MC=prep.proton_pion_division(prep.get_MC_bckg(file_directory_mc, tree_mc, folder_name=fname))

    train_test_data = train_test_generator([bckg,nonprompt,prompt], [0, 1, 2], test_size=0.5, random_state=42)
    train_test_dataMC = train_test_generator([bckg_MC,nonprompt,prompt], [0, 1, 2], test_size=0.5, random_state=42)

    return train_test_data,train_test_dataMC

def define_model():
    model_clf = xgb.XGBClassifier()
    features_to_learn=["fDcaV0PV","fCosPA","fDcaV0Tracks","fDcaPVProton","fDcaPVPion"]
    model_hdl = ModelHandler(model_clf, features_to_learn)

    return model_hdl

def plot_roc(train_test_data, model_hdl:ModelHandler,save_fig:bool=False,filename:str="ouput.pdf"):
    
    y_pred_train = model_hdl.predict(train_test_data[0], False) #prediction for training data set
    y_pred_test = model_hdl.predict(train_test_data[2], False)  #prediction for test data set
    plot_utils.plot_roc(train_test_data[3], y_pred_test, None, multi_class_opt="ovo")
    if save_fig:
        plt.savefig(save_dir+filename)


def plot_bdt(train_test_data, model_hdl:ModelHandler,marg:bool=False,save_fig:bool=False, filename:str="ouput.pdf"):

    pdf_filename = save_dir+filename
    pdf = PdfPages(pdf_filename)
    plot_utils.plot_output_train_test(model_hdl,train_test_data, bins=100, density=True, output_margin=marg)
    if save_fig:
        for fig_num in plt.get_fignums():
            fig = plt.figure(fig_num)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
        pdf.close()


