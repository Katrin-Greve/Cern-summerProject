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
import preparing_data as prep
from matplotlib.backends.backend_pdf import PdfPages



no_set=5

directory_sets=f"/home/katrin/Cern_summerProject/root_trees/set_{no_set}/"
save_dir_plots=f"/home/katrin/Cern_summerProject/ml_plots/set_{no_set}/"

def define_model(features_to_learn:Sequence[str]):
    model_clf = xgb.XGBClassifier()
    model_hdl = ModelHandler(model_clf, features_to_learn)

    return model_hdl

def traindata_plotting(training_sets:Sequence[TreeHandler], set_names:Sequence[str],features_to_learn:Sequence[str]=["fDcaV0PV","fCosPA","fDcaV0Tracks","fDcaPVProton","fDcaPVPion"], name_training:str="train0", equal_no_cand:bool=False, multi:bool=False, marg:bool=False, save_fig:bool=False):

    if not equal_no_cand:
        if multi:
            train_test_data = train_test_generator(training_sets, [0, 1, 2], test_size=0.5, random_state=42)
        else:
            train_test_data = train_test_generator(training_sets, [0, 1], test_size=0.5, random_state=42)
    else:
        training_sets_equ=[]
        minim_cand=min([tset.get_n_cand() for tset in training_sets])
        print(minim_cand)
        for i in range(len(training_sets)):
            training_sets_equ.append(training_sets[i].get_subset(size=minim_cand))
    
        if multi:
            train_test_data = train_test_generator(training_sets_equ, [0, 1, 2], test_size=0.5, random_state=42)
        else:
            train_test_data = train_test_generator(training_sets_equ, [0, 1], test_size=0.5, random_state=42)

    model=define_model(features_to_learn=features_to_learn)
    model.train_test_model(train_test_data, multi_class_opt="ovo")
    for st,name in zip(training_sets,set_names):
        if multi:
            st.apply_model_handler(model_handler=model,column_name=[name_training+"_class0",name_training+"_class1",name_training+"_class2"],output_margin=marg)
        else:
            st.apply_model_handler(model_handler=model,column_name=name_training+"_bdt",output_margin=marg)
        prep.get_root_from_TreeHandler(treehdl=st,output_name=name+".root",save_dir=directory_sets,treename="tree")

    plot_bdt(train_test_data, model_hdl=model, marg=marg, save_fig=save_fig, filename=name_training+"_bdt.pdf",labels=set_names)
    plot_roc(train_test_data=train_test_data, model_hdl=model,save_fig=save_fig,  filename=name_training+"_roc.pdf",multi=multi,labels=set_names)
    plot_featimport(data=train_test_data,model=model,save_fig=save_fig,filename=name_training+"_featimp.pdf",multi=multi,labels=set_names)

    
def traindata(training_sets:Sequence[TreeHandler], set_names:Sequence[str],features_to_learn:Sequence[str]=["fDcaV0PV","fCosPA","fDcaV0Tracks","fDcaPVProton","fDcaPVPion"], name_training:str="train0", equal_no_cand:bool=False,multi:bool=False,marg:bool=False,):

    if not equal_no_cand:
        if multi:
            train_test_data = train_test_generator(training_sets, [0, 1, 2], test_size=0.5, random_state=42)
        else:
            train_test_data = train_test_generator(training_sets, [0, 1], test_size=0.5, random_state=42)
    else:
        training_sets_equ=[]
        minim_cand=min([tset.get_n_cand() for tset in training_sets])
        print(minim_cand)
        for i in range(len(training_sets)):
            training_sets_equ.append(training_sets[i].get_subset(size=minim_cand))
        if multi:
            train_test_data = train_test_generator(training_sets_equ, [0, 1, 2], test_size=0.5, random_state=42)
        else:
            train_test_data = train_test_generator(training_sets_equ, [0, 1], test_size=0.5, random_state=42)


    model=define_model(features_to_learn=features_to_learn)
    if not multi:
        hyper_pars_ranges = {'n_estimators': (200, 1000), 'max_depth': (2, 4), 'learning_rate': (0.01, 0.1)}
        model.optimize_params_optuna(train_test_data, hyper_pars_ranges, cross_val_scoring='roc_auc', timeout=120,n_jobs=-1, n_trials=100, direction='maximize')  

    model.train_test_model(train_test_data,multi_class_opt="ovo")
    for st,name in zip(training_sets,set_names):
        if multi:
            st.apply_model_handler(model_handler=model,column_name=[name_training+"_class0",name_training+"_class1",name_training+"_class2"],output_margin=marg)
        else:
            st.apply_model_handler(model_handler=model,column_name=name_training+"_bdt",output_margin=marg)
        prep.get_root_from_TreeHandler(treehdl=st,output_name=name+".root",save_dir=directory_sets,treename="tree")

    return train_test_data,model


def plot_roc(train_test_data, model_hdl:ModelHandler,labels:Sequence[str], save_fig:bool=True,filename:str="roc_ouput.pdf",multi:bool=False):
    
    y_pred_train = model_hdl.predict(train_test_data[0], False) #prediction for training data set
    y_pred_test = model_hdl.predict(train_test_data[2], False)  #prediction for test data set
    if multi:
        plot_utils.plot_roc(train_test_data[3], y_pred_test, None, multi_class_opt="ovo",labels=labels)
    else:
        plot_utils.plot_roc(train_test_data[3], y_pred_test, None,labels=labels)
    plt.show()
    if save_fig:
        plt.savefig(save_dir_plots+filename)
        plt.close()


def plot_bdt(train_test_data, model_hdl:ModelHandler, labels:Sequence[str], marg:bool=False, save_fig:bool=True, filename:str="bdt_ouput.pdf"):


    plot_utils.plot_output_train_test(model_hdl,train_test_data, bins=100,labels=labels ,density=True, output_margin=marg)
    plt.show()
    if save_fig:
        pdf_filename = save_dir_plots+filename
        pdf = PdfPages(pdf_filename)
        for fig_num in plt.get_fignums():
            fig = plt.figure(fig_num)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
        pdf.close()

def plot_featimport(data:Sequence[Union[pd.DataFrame,np.array]],model:ModelHandler,labels:Sequence[str],save_fig:bool=False, filename:str="output_featimp.pdf",multi:bool=False):


    plot_utils.plot_feature_imp(data[0],y_truth=data[1], model=model,labels=labels)
    plot_utils.plot_feature_imp(data[2],y_truth=data[3], model=model,labels=labels)
    plt.show()
        #plot_utils.plot_feature_imp(data[0],y_truth=data[1], model=model,labels=["bckg", "nonprompt","prompt"])
        #plot_utils.plot_feature_imp(data[2],y_truth=data[3], model=model,labels=["bckg", "nonprompt","prompt"])        
    if save_fig:
        pdf_filename = save_dir_plots+filename
        pdf = PdfPages(pdf_filename)
        for fig_num in plt.get_fignums():
            fig = plt.figure(fig_num)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
        pdf.close()    

#allsets=plotting.get_sets(already_saved=True, onlynewMC=True)


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

#bckg_MC_leftsideband=prep.cut_data(data=allsets["bckg_MC_radiuscut"],var="fMass", upper_cut=1.1)
#prompt_radiuscut=prep.cut_data(data=allsets["nonprompt"],var="fRadius", upper_cut=40)
#nonprompt_radiuscut=prep.cut_data(data=allsets["prompt"], var="fRadius", upper_cut=40)

#bckg_MC_radiuscut=prep.cut_data(data=allsets["bckg_MC"], var="fRadius", upper_cut=40)
#traindata_plotting([allsets["nonprompt"],allsets["prompt"]],set_names=["nonprompt", "prompt"],name_training="trainBckgMC_dual",features_to_learn=["fDcaV0PV","fCosPA","fDcaV0Tracks","fDcaPVProton","fDcaPVPion","fCt"],multi=False)
#traindata_plotting([allsets["nonprompt"],allsets["prompt"]],set_names=["nonprompt", "prompt"],name_training="train_dual",features_to_learn=["fCosPA","fDcaV0Tracks","fDcaPVProton","fDcaPVPion","fDcaV0PV","fCt","fEta"],multi=False)