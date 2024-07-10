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
import os


no_set=4

directory_sets=f"/home/katrin/Cern_summerProject/root_trees/set_{no_set}/"
save_dir_plots=f"/home/katrin/Cern_summerProject/ml_plots/set_{no_set}/"

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

if no_set==4:
    file_directory_data="/home/katrin/Cern_summerProject/data/AO2D_data_new.root"
    tree_data="O2lambdatableml"
    file_directory_mc="/home/katrin/Cern_summerProject/data/A02D_MC_mothdau.root"
    tree_mc="O2mclambdatableml"
    fname="DF*"

def define_model(features_to_learn:Sequence[str]):
    model_clf = xgb.XGBClassifier()
    model_hdl = ModelHandler(model_clf, features_to_learn)

    return model_hdl

def traindata_plotting(training_sets:Sequence[TreeHandler], set_names:Sequence[str],features_to_learn:Sequence[str]=["fDcaV0PV","fCosPA","fDcaV0Tracks","fDcaPVProton","fDcaPVPion"], name_training:str="train0", equal_no_cand:bool=False):

    if not equal_no_cand:
        train_test_data = train_test_generator(training_sets, [0, 1, 2], test_size=0.5, random_state=42)
    else:
        training_sets_equ=[]
        minim_cand=min([tset.get_n_cand() for tset in training_sets])
        print("No. of candidates in smallest set: ", minim_cand)
        for i in range(len(training_sets)):
            training_sets_equ.append(training_sets[i].get_subset(size=minim_cand))
        train_test_data = train_test_generator(training_sets_equ, [0, 1, 2], test_size=0.5, random_state=42)

    model=define_model(features_to_learn=features_to_learn)
    model.train_test_model(train_test_data, multi_class_opt="ovo")
    for st,name in zip(training_sets,set_names):
        st.apply_model_handler(model_handler=model,column_name=[name_training+"_class0",name_training+"_class1",name_training+"_class2"])
        prep.get_root_from_TreeHandler(treehdl=st,output_name=name+".root",save_dir=directory_sets,treename="tree")

    plot_bdt(train_test_data, model_hdl=model, marg=True, save_fig="True", filename=name_training+"_bdt.pdf")
    plot_roc(train_test_data=train_test_data, model_hdl=model, filename=name_training+"_roc.pdf")
    plot_featimport(data=train_test_data,model=model,filename=name_training+"_featimp.pdf")

    
def traindata(training_sets:Sequence[TreeHandler], set_names:Sequence[str],features_to_learn:Sequence[str]=["fDcaV0PV","fCosPA","fDcaV0Tracks","fDcaPVProton","fDcaPVPion"], name_training:str="train0", equal_no_cand:bool=False):

    if not equal_no_cand:
        train_test_data = train_test_generator(training_sets, [0, 1, 2], test_size=0.5, random_state=42)
    else:
        training_sets_equ=[]
        minim_cand=min([tset.get_n_cand() for tset in training_sets])
        print(minim_cand)
        for i in range(len(training_sets)):
            training_sets_equ.append(training_sets[i].get_subset(size=minim_cand))

        train_test_data = train_test_generator(training_sets_equ, [0, 1, 2], test_size=0.5, random_state=42)

    model=define_model(features_to_learn=features_to_learn)
    model.train_test_model(train_test_data, multi_class_opt="ovo")
    for st,name in zip(training_sets,set_names):
        st.apply_model_handler(model_handler=model,column_name=[name_training+"_class0",name_training+"_class1",name_training+"_class2"])
        prep.get_root_from_TreeHandler(treehdl=st,output_name=name+".root",save_dir=directory_sets,treename="tree")

    return train_test_data,model


def plot_roc(train_test_data, model_hdl:ModelHandler,save_fig:bool=True,filename:str="roc_ouput.pdf"):
    
    y_pred_train = model_hdl.predict(train_test_data[0], False) #prediction for training data set
    y_pred_test = model_hdl.predict(train_test_data[2], False)  #prediction for test data set
    plot_utils.plot_roc(train_test_data[3], y_pred_test, None, multi_class_opt="ovo")
    if save_fig:
        plt.savefig(save_dir_plots+filename)
        plt.close()


def plot_bdt(train_test_data, model_hdl:ModelHandler, marg:bool=False, save_fig:bool=True, filename:str="bdt_ouput.pdf"):

    pdf_filename = save_dir_plots+filename
    pdf = PdfPages(pdf_filename)
    plot_utils.plot_output_train_test(model_hdl,train_test_data, bins=100, density=True, output_margin=marg)
    if save_fig:
        for fig_num in plt.get_fignums():
            fig = plt.figure(fig_num)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
        pdf.close()

def plot_featimport(data:Sequence[Union[pd.DataFrame,np.array]],model:ModelHandler,save_fig:bool=True, filename:str="output_featimp.pdf"):

    pdf_filename = save_dir_plots+filename
    pdf = PdfPages(pdf_filename)
    plot_utils.plot_feature_imp(data[0],y_truth=data[1], model=model,labels=["Bckg", "nonprompt","prompt"])
    plot_utils.plot_feature_imp(data[2],y_truth=data[3], model=model,labels=["Bckg", "nonprompt","prompt"])

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
        
        #get Mass cutted Baackground
        var_to_cut="fMass"
        lower_cut=1.165
        upper_cut=None
        bckg_cutted_Mass=prep.cut_data(data=bckg_data,var=var_to_cut,lower_cut=lower_cut,upper_cut=upper_cut)
    
        #get Ct cutted Background
        var_to_cut="fCt"
        lower_cut=None
        upper_cut=15
        bckg_cutted_Ct=prep.cut_data(data=bckg_data,var=var_to_cut,lower_cut=lower_cut,upper_cut=upper_cut)

        allsets=[prompt, nonprompt, bckg_MC, bckg_data,  bckg_cutted_Ct, bckg_cutted_Mass]
        names=prep.get_variable_names(allsets,locals())
        prep.save_sets(allsets,set_names=names,dir_tree=directory_sets, tree="tree")
        print("Subsets saved!")
    else:
        print("Subsets already saved")
    try:
        filenames= os.listdir(directory_sets)
    except Exception as e:
        print("something went wrong: ", e)

    allsets={}
    for file in filenames:
        allsets[file[:-5]]=TreeHandler(directory_sets+file, "tree")
    
    return allsets

#sets=get_subsets(already_saved=True)


#bckg_MC_cuttedMass=prep.cut_data(data=allsets["bckg_MC"],var="fMass",lower_cut=1.09, upper_cut=1.13,inclusive=False)
#bckg_MC_cuttedMass2=prep.cut_data(data=sets["bckg_MC"],var="fMass",lower_cut=1.09, upper_cut=1.15,inclusive=False)

#bckg_MC_cuttedRadius=prep.cut_data(data=allsets["bckg_MC"],var="fRadius",lower_cut=None, upper_cut=20)
#bckg_MC_cuttedMassRadius=prep.cut_data(data=bckg_MC_cuttedMass, var="fRadius", lower_cut=None, upper_cut=20)
#prompt_cuttedRadius=prep.cut_data(data=allsets["prompt"],var="fRadius",lower_cut=None, upper_cut=20)
#nonprompt_cuttedRadius=prep.cut_data(data=allsets["nonprompt"],var="fRadius",lower_cut=None, upper_cut=20)
#prep.save_sets(sets=[bckg_MC_cuttedMass,bckg_MC_cuttedMass2,bckg_MC_cuttedRadius, bckg_MC_cuttedMassRadius,prompt_cuttedRadius,nonprompt_cuttedRadius],set_names=["bckg_MC_cuttedMass","bckg_MC_cuttedMass2","bckg_MC_cuttedRadius","bckg_MC_cuttedMassRadius","prompt_cuttedRadius","nonprompt_cuttedRadius"],dir_tree=directory_sets)
#prep.save_sets([bckg_MC_cuttedMass2],set_names=["bckg_MC_cuttedMass2"],dir_tree=directory_sets)
#allsets=get_subsets(already_saved=True)



#traindata_plotting([allsets["bckg_MC_cuttedRadius"],allsets["nonprompt_cuttedRadius"],allsets["prompt_cuttedRadius"]],set_names=["bckg_MC_cuttedRadius","nonprompt_cuttedRadius", "prompt_cuttedRadius"],name_training="trainBckgMC_cuttedRadius",features_to_learn=["fDcaV0PV","fCosPA","fDcaV0Tracks","fDcaPVProton","fDcaPVPion"])
#traindata_plotting([allsets["bckg_MC_cuttedMassRadius"],allsets["nonprompt_cuttedRadius"],allsets["prompt_cuttedRadius"]],set_names=["bckg_MC_cuttedMassRadius","nonprompt_cuttedRadius", "prompt_cuttedRadius"],name_training="trainBckgMC_cuttedMassRadius",features_to_learn=["fDcaV0PV","fCosPA","fDcaV0Tracks","fDcaPVProton","fDcaPVPion"])
#traindata_plotting([allsets["bckg_MC"],allsets["nonprompt"],allsets["prompt"]],set_names=["bckg_MC","nonprompt", "prompt"],name_training="trainBckgMC_withall",features_to_learn=["fCosPA","fDcaV0Tracks","fDcaPVProton","fDcaPVPion","fDcaV0PV","fCt", "fRadius","fEta","fTpcNsigmaProton", "fTpcNsigmaPion"])


#traindata([allsets["bckg_MC"],allsets["nonprompt"],allsets["prompt"]],set_names=["bckg_MC","nonprompt", "prompt"],name_training="trainBckgMC")


allsets=get_subsets(already_saved=True)
traindata_plotting([allsets["bckg_MC"],allsets["nonprompt"],allsets["prompt"]],set_names=["bckg_MC","nonprompt", "prompt"],name_training="trainBckgMC",features_to_learn=["fDcaV0PV","fCosPA","fDcaV0Tracks","fDcaPVProton","fDcaPVPion","fCt"])
traindata_plotting([allsets["bckg_MC_filtered"],allsets["nonprompt"],allsets["prompt"]],set_names=["bckg_MC_filtered","nonprompt", "prompt"],name_training="trainBckgMC_filtered",features_to_learn=["fDcaV0PV","fCosPA","fDcaV0Tracks","fDcaPVProton","fDcaPVPion","fCt"])
