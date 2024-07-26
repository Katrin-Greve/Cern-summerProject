import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from hipe4ml.model_handler import ModelHandler
from hipe4ml.tree_handler import TreeHandler
from hipe4ml.analysis_utils import train_test_generator
from hipe4ml import plot_utils
from typing import Union, Sequence, Tuple
import preparing_data as prep
from matplotlib.backends.backend_pdf import PdfPages


def define_model(features_to_learn:Sequence[str]):
    model_clf = xgb.XGBClassifier()
    model_hdl = ModelHandler(model_clf, features_to_learn)

    return model_hdl

def trainmodel_plotting(training_sets:Sequence[TreeHandler], set_names:Sequence[str],directory_sets:str, save_dir_mlplots:str,features_to_learn:Sequence[str]=["fCosPA","fDcaV0Tracks","fDcaPVProton","fDcaPVPion","fDcaV0PV","fCt","fEta"], name_training:str="train0", equal_no_cand:bool=False, multi:bool=False, marg:bool=False, save_fig:bool=False,fs:Tuple[float]=(10,4)):

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

    plot_bdt(train_test_data, model_hdl=model,save_dir_mlplots=save_dir_mlplots, marg=marg, save_fig=save_fig, filename=name_training+"_bdt.pdf",labels=set_names)
    plot_roc(train_test_data=train_test_data, model_hdl=model,save_dir_mlplots=save_dir_mlplots,save_fig=save_fig,  filename=name_training+"_roc.pdf",multi=multi,labels=set_names)
    plot_featimport(data=train_test_data,model=model,save_dir_mlplots=save_dir_mlplots,save_fig=save_fig, filename=name_training+"_featimp.pdf",multi=multi,labels=set_names,figsize=fs)

    return train_test_data, model

    
def trainmodel(training_sets:Sequence[TreeHandler], set_names:Sequence[str],directory_sets:str,features_to_learn:Sequence[str]=["fCosPA","fDcaV0Tracks","fDcaPVProton","fDcaPVPion","fDcaV0PV","fCt","fEta"], name_training:str="train0", equal_no_cand:bool=False,multi:bool=False,marg:bool=False,):

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
    #if not multi:
    #    hyper_pars_ranges = {'n_estimators': (200, 1000), 'max_depth': (2, 4), 'learning_rate': (0.01, 0.1)}
    #    model.optimize_params_optuna(train_test_data, hyper_pars_ranges, cross_val_scoring='roc_auc', timeout=120,n_jobs=-1, n_trials=100, direction='maximize')  

    model.train_test_model(train_test_data,multi_class_opt="ovo")
    for st,name in zip(training_sets,set_names):
        if multi:
            st.apply_model_handler(model_handler=model,column_name=[name_training+"_class0",name_training+"_class1",name_training+"_class2"],output_margin=marg)
        else:
            st.apply_model_handler(model_handler=model,column_name=name_training+"_bdt",output_margin=marg)
        prep.get_root_from_TreeHandler(treehdl=st,output_name=name+".root",save_dir=directory_sets,treename="tree")

    return train_test_data, model


def plot_roc(train_test_data, model_hdl:ModelHandler,labels:Sequence[str],save_dir_mlplots:str, save_fig:bool=True,filename:str="roc_ouput.pdf",multi:bool=False):
    
    y_pred_train = model_hdl.predict(train_test_data[0], False) #prediction for training data set
    y_pred_test = model_hdl.predict(train_test_data[2], False)  #prediction for test data set
    if multi:
        plot_utils.plot_roc(train_test_data[3], y_pred_test, None, multi_class_opt="ovo",labels=labels)
    else:
        plot_utils.plot_roc(train_test_data[3], y_pred_test, None,labels=labels)
    plt.show()
    if save_fig:
        plt.savefig(save_dir_mlplots+filename)
        plt.close()


def plot_bdt(train_test_data, model_hdl:ModelHandler, labels:Sequence[str],save_dir_mlplots:str, marg:bool=False, save_fig:bool=True, filename:str="bdt_ouput.pdf"):

    plot_utils.plot_output_train_test(model_hdl,train_test_data, bins=100,labels=labels ,density=True, output_margin=marg)
    plt.show()
    if save_fig:
        pdf_filename = save_dir_mlplots+filename
        pdf = PdfPages(pdf_filename)
        for fig_num in plt.get_fignums():
            fig = plt.figure(fig_num)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
        pdf.close()

def plot_featimport(data:Sequence[Union[pd.DataFrame,np.array]],model:ModelHandler,labels:Sequence[str],save_dir_mlplots:str,save_fig:bool=False, filename:str="output_featimp.pdf",multi:bool=False,figsize:Tuple[float]=(10,3)):


    figs=plot_utils.plot_feature_imp(data[0],y_truth=data[1], model=model,labels=labels)
    for fig in figs:
        fig.set_size_inches(figsize)
    #plot_utils.plot_feature_imp(data[2],y_truth=data[3], model=model,labels=labels)
    plt.show()
        #plot_utils.plot_feature_imp(data[0],y_truth=data[1], model=model,labels=["bckg", "nonprompt","prompt"])
        #plot_utils.plot_feature_imp(data[2],y_truth=data[3], model=model,labels=["bckg", "nonprompt","prompt"])        
    if save_fig:
        pdf_filename = save_dir_mlplots+filename
        pdf = PdfPages(pdf_filename)
        for fig_num in plt.get_fignums():
            fig = plt.figure(fig_num)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
        pdf.close()

def save_model(model: ModelHandler,file_name:str):
    model.dump_model_handler(filename=file_name)

def get_model(file_name:str):
    model=ModelHandler()
    model.load_model_handler(filename=file_name)
    return model

def model_prediction_data(model:ModelHandler, directory_data:str, tree_data:str="O2lambdatableml", fname:str="DF*", marg:bool=True, applyModelHandler:bool=False,name_training:str="train",cut:bool=False):
    data=prep.get_rawdata(file_directory_data=directory_data, tree_data=tree_data,folder_name=fname)
    if not applyModelHandler:
        predictions=model.predict(data, output_margin=marg)
        return predictions
    
    else:
        if cut:
            var=str(input("Variable to apply cut:"))
            lowerlimit=float(input("lower limit:"))
            upperlimit=float(input("upper limit: "))
            prep.cut_data(data=data,var=var, lower_cut=lowerlimit,upper_cut=upperlimit)
            data.apply_model_handler(model_handler=model,output_margin=marg, column_name=[name_training+"_class0",name_training+"_class1",name_training+"_class2"])
        else: 
            data.apply_model_handler(model_handler=model,output_margin=marg, column_name=[name_training+"_class0",name_training+"_class1",name_training+"_class2"])
        return data
        
