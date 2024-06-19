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

no_set=3

directory_sets=f"/home/katrin/Cern_summerProject/root_trees/set_{no_set}/"
save_dir_plots="/home/katrin/Cern_summerProject/imgs/ml_plots/"

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

def define_model():
    model_clf = xgb.XGBClassifier()
    features_to_learn=["fDcaV0PV","fCosPA","fDcaV0Tracks","fDcaPVProton","fDcaPVPion"]
    model_hdl = ModelHandler(model_clf, features_to_learn)

    return model_hdl

def traindata(training_sets:Sequence[TreeHandler],set_names:Sequence[str],name_training:str="train0"):

    train_test_data = train_test_generator(training_sets, [0, 1, 2], test_size=0.5, random_state=42)
    model=define_model()
    trained_model=model.train_test_model(train_test_data, multi_class_opt="ovo")
    for st,name in zip(training_sets,set_names):
        st.apply_model_handler(model_handler=model,column_name=["mloutput_class0","mloutput_class1","mloutput_class2"])
        prep.get_root_from_TreeHandler(treehdl=st,output_name=name+"_"+name_training+".root",save_dir=directory_sets,treename="tree")

    return train_test_data,trained_model


def plot_roc(train_test_data, model_hdl:ModelHandler,save_fig:bool=False,filename:str="ouput.pdf"):
    
    y_pred_train = model_hdl.predict(train_test_data[0], False) #prediction for training data set
    y_pred_test = model_hdl.predict(train_test_data[2], False)  #prediction for test data set
    plot_utils.plot_roc(train_test_data[3], y_pred_test, None, multi_class_opt="ovo")
    if save_fig:
        plt.savefig(save_dir_plots+filename)


def plot_bdt(train_test_data, model_hdl:ModelHandler,marg:bool=False,save_fig:bool=False, filename:str="ouput.pdf"):

    pdf_filename = save_dir_plots+filename
    pdf = PdfPages(pdf_filename)
    plot_utils.plot_output_train_test(model_hdl,train_test_data, bins=100, density=True, output_margin=marg)
    if save_fig:
        for fig_num in plt.get_fignums():
            fig = plt.figure(fig_num)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
        pdf.close()


def get_subsets(already_saved:bool=True):

    if already_saved==False:
        print("sets not saved yet")
        prompt, nonprompt, bckg_data, bckg_MC=prep.get_base_sets(file_directory_data=file_directory_data, file_directory_mc=file_directory_mc, tree_data=tree_data, tree_mc=tree_mc, fname=fname)
        var_to_cut="fMass"
        lower_cut=1.165
        upper_cut=None
        bckg_cutted_Mass=prep.cut_data(data=bckg_data,var=var_to_cut,lower_cut=lower_cut,upper_cut=upper_cut)
    

        var_to_cut="fCt"
        lower_cut=None
        upper_cut=15
        bckg_cutted_Ct=prep.cut_data(data=bckg_data,var=var_to_cut,lower_cut=lower_cut,upper_cut=upper_cut)

        allsets=[prompt, nonprompt, bckg_MC, bckg_data,  bckg_cutted_Ct, bckg_cutted_Mass]
        names=prep.get_variable_names(allsets,locals())
        prep.save_sets(allsets,set_names=names,dir_tree=directory_sets,no_set=no_set, tree="tree")
        print("Subsets saved!")


        #names=prep.get_variable_names([prompt, nonprompt, bckg_MC,bckg_data,  bckg_cutted_Ct, bckg_cutted_Mass], locals())

    else:
        prompt=TreeHandler(directory_sets+f"prompt_{no_set}.root", "tree")
        nonprompt=TreeHandler(directory_sets+f"nonprompt_{no_set}.root", "tree")
        bckg_data=TreeHandler(directory_sets+f"bckg_data_{no_set}.root", "tree")
        bckg_MC=TreeHandler(directory_sets+f"bckg_MC_{no_set}.root", "tree")
        bckg_cutted_Ct=TreeHandler(directory_sets+f"bckg_cutted_Ct_{no_set}.root", "tree")
        bckg_cutted_Mass=TreeHandler(directory_sets+f"bckg_cutted_Mass_{no_set}.root", "tree")

        allsets=[prompt, nonprompt, bckg_MC,bckg_data,  bckg_cutted_Ct, bckg_cutted_Mass]
        names=prep.get_variable_names([prompt, nonprompt, bckg_MC, bckg_data,  bckg_cutted_Ct, bckg_cutted_Mass], locals())

    return allsets, names

allsets,names=get_subsets(already_saved=True)
traindata([allsets[5],allsets[1],allsets[0]],[names[3],names[1],names[0]],name_training="bckg_masscut")
traindata([allsets[3],allsets[1],allsets[0]],[names[3],names[1],names[0]],name_training="train0")
