import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import preparing_data as prep
from hipe4ml import plot_utils
import itertools
from hipe4ml.tree_handler import TreeHandler
from hipe4ml.model_handler import ModelHandler
import rooFitting as rfit
from typing import Union, Sequence, Tuple
import seaborn as sns 
import os
import numpy as np
import ROOT
import PyPDF2
import matplotlib.image as mpimg

#no_set=5
#directory_sets=f"/home/katrin/Cern_summerProject/root_trees/set_{no_set}/"
#directory_hists=f"/home/katrin/Cern_summerProject/root_histograms/set_{no_set}/"
#save_dir_plots = f'/home/katrin/Cern_summerProject/imgs/set_{no_set}/'
#
#
#if no_set==1:
#    file_directory_data="/home/katrin/Cern_summerProject/data/AnalysisResults_treesML_data.root"
#    tree_data="O2lambdatableml"
#    file_directory_mc="/home/katrin/Cern_summerProject/data/AnalysisResults_treesML.root"
#    tree_mc="O2mclambdatableml"
#    fname=None
#
#if no_set==3:
#    file_directory_data="/home/katrin/Cern_summerProject/data/AO2D_data.root"
#    tree_data="O2lambdatableml"
#    file_directory_mc="/home/katrin/Cern_summerProject/data/AO2D_MC.root"
#    tree_mc="O2mclambdatableml"
#    fname="DF*"
#
#
#if no_set==2:
#    file_directory_data="/home/katrin/Cern_summerProject/data/AnalysisResults_treesML_data_LHC22o_apass6_small.root"
#    tree_data="O2lambdatableml"
#    file_directory_mc="/home/katrin/Cern_summerProject/data/AnalysisResults_treesML_mc_LHC24b1b_small.root"
#    tree_mc="O2mclambdatableml"
#    fname=None
#
#if no_set==4:
#    file_directory_data="/home/katrin/Cern_summerProject/data/AO2D_data_new.root"
#    tree_data="O2lambdatableml"
#    #file_directory_mc="/home/katrin/Cern_summerProject/data/AO2D_MC_mothdau.root"
#    file_directory_mc="/home/katrin/Cern_summerProject/data/AO2D_MC_daughters.root"
#    tree_mc="O2mclambdatableml"
#    fname="DF*"
#
#if no_set==5:
#    file_directory_data="/home/katrin/Cern_summerProject/data/AO2D_data_woK0.root"
#    tree_data="O2lambdatableml"
#    #file_directory_mc="/home/katrin/Cern_summerProject/data/AO2D_MC_mothdau.root"
#    #file_directory_mc="/home/katrin/Cern_summerProject/data/AO2D_MC_woK0.root"
#    file_directory_mc="/home/katrin/Cern_summerProject/data/AO2D_MC_daughters.root"

#    tree_mc="O2mclambdatableml"
#    fname="DF*"


# function to get the usual needed data sets
def get_sets(directory_sets:str, file_directory_data:str, tree_data:str, file_directory_mc:str, tree_mc:str, already_saved:bool=True, onlynewMC:bool=False, onlynewDataBckg:bool=False, fname:str="DF*"):
    """
    Get the usual needed data sets

    Parameters:
        already_saved (bool): Decides whether the data sets have to be calculated or are already saved as a root file. Default to True
    """
    if onlynewMC==True and already_saved==True:
        print("MC sets not saved yet")
        bckg_MC=prep.add_GenRadius(prep.proton_pion_division(prep.get_MC_bckg(file_directory_mc, tree_mc, folder_name=fname)))
        prompt=prep.add_GenRadius(prep.proton_pion_division(prep.get_prompt(file_directory_mc=file_directory_mc,tree_mc=tree_mc,folder_name=fname)))
        nonprompt=prep.add_GenRadius(prep.proton_pion_division(prep.get_nonprompt(file_directory_mc=file_directory_mc,tree_mc=tree_mc,folder_name=fname)))
        mcsets=[prompt, nonprompt, bckg_MC]
        names=prep.get_variable_names(mcsets,locals())
        save_sets(mcsets,set_names=names,dir_tree=directory_sets)
        print("MC Subsets saved!")
    
    elif already_saved==True and onlynewDataBckg == True:
        cu=prep.fit_gauss_rec(prep.get_rawdata(file_directory_data, tree_data, folder_name=fname), var="fMass",p0=[300000,1.115,0.005,1000],sig_cut=13)[2]
        bckg_data=prep.add_Radius(prep.proton_pion_division(prep.get_bckg(file_directory_data, tree_data, cu, folder_name=fname)))
        save_sets(sets=[bckg_data],set_names=["bckg_data"],dir_tree=directory_sets)

    elif already_saved==False:
        print("sets not saved yet")
        prompt, nonprompt, bckg_data, bckg_MC=prep.get_base_sets(file_directory_data=file_directory_data, file_directory_mc=file_directory_mc, tree_data=tree_data, tree_mc=tree_mc, fname=fname)
        allsets=[prompt, nonprompt, bckg_MC,bckg_data]
        names=prep.get_variable_names(allsets,locals())
        save_sets(allsets,set_names=names,dir_tree=directory_sets, tree="tree")
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

# function to visualize, how one gets the background out of the data by using cuts on the mass
def plot_bck_cuts(file_directory_data:str, tree_data:str, pdf_filename:str, fname:str="DF*",save_fig:bool=False):

    data=prep.get_rawdata(file_directory_data,tree_data, folder_name=fname)
    prep.plot_hist(data,"fMass",leg_labels="raw data", fs=(8,3),alpha=0.5)

    cu2=prep.fitplot_gauss_rec(data, var="fMass",p0=[300000,1.115,0.005,1000],fs=(8,3),sig_cut=6)[2]
    bckg=prep.get_bckg(file_directory_data, tree_data, cuts=cu2, folder_name=fname)

    prep.plot_hist(bckg,"fMass",leg_labels=f"bckg, cuts=]{cu2[0]:.4f}{cu2[1]:.4f}[",fs=(8,3),alpha=0.5)
    if save_fig:
        pdf = PdfPages(pdf_filename) 
        for fig_num in plt.get_fignums():
            fig = plt.figure(fig_num)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
        pdf.close()
    else:
        plt.show()

#function that shows the distribution of the MC data, the prompt and the nonprompt distribution
def plot_prompt_nonprompt(file_directory_mc:str, tree_mc:str,  pdf_filename:str="Mass_Prompt_Nonprompt.pdf", fname:str="DF*", save_fig:bool=False):
    
    data=prep.get_rawMC_data(file_directory_mc, tree_mc ,folder_name=fname)
    prompt= prep.get_prompt(file_directory_mc, tree_mc,folder_name=fname)
    nonprompt=prep.get_nonprompt(file_directory_mc, tree_mc,folder_name=fname)
    prep.plot_hist(data,"fMass",leg_labels="raw MC", fs=(8,3),alpha=0.5)
    prep.plot_hist(prompt,"fMass",leg_labels="prompt",fs=(8,3),alpha=0.5)
    prep.plot_hist(nonprompt,"fMass",leg_labels="nonprompt",fs=(8,3),alpha=0.5)
    if save_fig:
        pdf = PdfPages(pdf_filename)
        for fig_num in plt.get_fignums():
            fig = plt.figure(fig_num)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
        pdf.close()
    else:
        plt.show()


def plot_allvar_dist(pdf_filename:str="Distributions.pdf", already_saved:bool=True, save_fig:bool=False, fontsize:int=20):

    allsets=get_sets(already_saved=already_saved)
    sets=list(allsets.values())
    set_names=list(allsets.keys())
    
    vars=[st.get_var_names() for st in sets]
    vars_shared=[var for var in vars[0] if all(var in sublist for sublist in vars)]
    prep.plot_hist([allsets["prompt"],allsets["nonprompt"],allsets["bckg_data"]], vars_to_draw=vars_shared,leg_labels=["prompt", "nonprompt","bckg_data"],fs=(15,10),alpha=0.3,fontsize=fontsize)

    colors=sns.color_palette(n_colors=len(set_names))
    for i in (0,2,4):
        print(set_names[i])
        prep.plot_hist(sets[i], vars_to_draw=vars[i],leg_labels=set_names[i], fs=(15,10) ,alpha=0.3,colors=colors[i],fontsize=fontsize)
    for i in (1,3,5):
        print(set_names[i])
        prep.plot_hist(sets[i], vars_to_draw=vars_shared+["fGenPt","fGenCt","fGenRadius"],leg_labels=set_names[i], fs=(15,10) ,alpha=0.3,colors=colors[i],fontsize=fontsize)

    if save_fig:
        pdf = PdfPages(pdf_filename)
        for fig_num in plt.get_fignums():
            fig = plt.figure(fig_num)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
        pdf.close()
    else:
        plt.show()


def plot_some_dist( sets:Sequence[TreeHandler], to_plot:Sequence[str],labels:Sequence[str], pdf_filename:str="some_Distributions.pdf",severalpages:Union[int,None]=None,no_bins:int=100,save_fig:bool=False, logy:bool=True, fontsize:int=20,fs:Tuple[float]=(15,10)):

    if severalpages:
        colors=sns.color_palette(n_colors=len(sets))
        if severalpages==2:
            prep.plot_hist(sets, vars_to_draw=to_plot,fs=fs,leg_labels=labels,alpha=0.3,colors=colors,no_bins=no_bins,logy=logy,fontsize=fontsize)
        for set,lab,col in zip(sets,labels,colors):
            prep.plot_hist(set, vars_to_draw=to_plot,fs=fs,leg_labels=lab,alpha=0.3,colors=col,no_bins=no_bins,logy=logy,fontsize=fontsize)
    else:
        prep.plot_hist(sets, vars_to_draw=to_plot,fs=fs,leg_labels=labels,alpha=0.3,no_bins=no_bins,logy=logy,fontsize=fontsize)
    
    if save_fig:
        pdf = PdfPages(pdf_filename)
        for fig_num in plt.get_fignums():
            fig = plt.figure(fig_num)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
        pdf.close()
    else:
        plt.show()

def plot_corr(pdf_filename:str="corr.pdf", already_saved:bool=True, save_fig:bool=False):

    allsets=get_sets(already_saved=already_saved)
    sets=list(allsets.values())
    names=list(allsets.keys())
    vars=[st.get_var_names() for st in sets]
    for st, variables, nm in zip(sets, vars, names):
        plot_utils.plot_corr([st], columns=variables, labels=[nm])
    if save_fig:
        pdf = PdfPages(pdf_filename)
        for fig_num in plt.get_fignums():
            fig = plt.figure(fig_num)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
        pdf.close()
    else:
        plt.show()


def plot_some_corr(pdf_filename:str, sets:Sequence[TreeHandler],  labels:Sequence[str], to_plot:Union[Sequence[str],None]=None,file_name:str="output_someCorr.pdf",save_fig:bool=False):

    if to_plot:
        plot_utils.plot_corr(sets,columns=to_plot,labels=labels)
    else:
        for st, lab in zip(sets, labels):
            cols=[var for var in st.get_var_names()]
            plot_utils.plot_corr([st],columns=cols,labels=[lab])
    if save_fig:
        pdf = PdfPages(pdf_filename)
        for fig_num in plt.get_fignums():
            fig = plt.figure(fig_num)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
        pdf.close()
    else:
        plt.show()

def plot_2dhist_numpy(sets:Sequence[TreeHandler], set_names:Sequence[str],varsx:Sequence[str], varsy:Sequence[str], pdf_filename:str="some_2dhist_numpy.pdf",binsx:int=10,binsy:int=10,cuts:Union[str,None]=None,save_pdf:bool=False,cmap:str="rainbow"):

    l1=varsx
    l2=varsy

    for (i,name) in zip(sets,set_names):
        fig,ax=plt.subplots(len(l1),len(l2),figsize=(4.7*len(l1)+2,4.7*(len(l2))))
        fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.3, hspace=0.3)        
        for ax_i, tu in zip(ax.T.flatten(),list(itertools.product(l1, l2))):
            cax=prep.plot_2dhist_numpy(i,tu[0],tu[1],ax=ax_i, binsx=binsx, binsy=binsy,cmap=cmap)
            ax_i.set(xlabel=tu[0],ylabel=tu[1])
        fig.suptitle(name+f", size of sample: {prep.get_sampleSize(i)}", fontsize=20)
        cbar = fig.colorbar(cax, ax=ax, orientation='vertical')
        cbar.set_label('Counts')
    if save_pdf:
        pdf = PdfPages(pdf_filename)    
        for fig_num in plt.get_fignums():
            fig = plt.figure(fig_num)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
        pdf.close()
    else:
        plt.show()

def plot_hist_root(save_directory:str, varsx:Sequence[str], data_names:Sequence[str],data:Union[Sequence[list],None]=None, files:Union[Sequence[str],None]=None, pdf_name:str="some_hist.pdf",binsx:int=100, save_pdf:bool=False, save_name_file:Union[str,None]=None,title:str="title",logy:bool=True,from_file:bool=True):

    dir_result=save_directory
    l1=varsx
    pdflist=[]
    if not from_file:
        for dat, tu, nm in zip(data,l1,data_names):
            hname=f"{nm}_{tu}"
            if not save_pdf:
                prep.plot_hist_root(var_name=tu, save_name_file=save_name_file, hist_name=hname, save_name_pdf=dir_result+f"{hname}.png",title=title, data=dat, no_bins=binsx,logy=logy,from_file=False)
                pdflist.append(dir_result+f"{hname}.png")
                display_root_canvas(image_path=dir_result+f"{hname}.png")
            else:
                prep.plot_hist_root(var_name=tu, save_name_file=save_name_file, hist_name=hname, save_name_pdf=dir_result+f"{hname}.png",title=title,data=dat, no_bins=binsx,logy=logy,from_file=False)
                pdflist.append(dir_result+f"{hname}.pdf")
    else:
        for file_data, tu ,nm in zip(files,l1,data_names):
            hname=f"{nm}_{tu}"
            if not save_pdf:
                prep.plot_hist_root(var_name=tu, save_name_file=save_name_file, hist_name=hname, save_name_pdf=dir_result+f"{hname}.png",title=title,file_name=file_data, no_bins=binsx,logy=logy,from_file=True)
                pdflist.append(dir_result+f"{hname}.png")
                display_root_canvas(image_path=dir_result+f"{hname}.png")
            else:
                prep.plot_hist_root(var_name=tu, save_name_file=save_name_file, hist_name=hname, save_name_pdf=dir_result+f"{hname}.png",title=title,file_name=file_data, no_bins=binsx,logy=logy,from_file=True)
                pdflist.append(dir_result+f"{hname}.pdf")
    if save_pdf:
        merge_pdfs(pdf_list=pdflist, output_path=dir_result+pdf_name)
    else: 
        for pdf in pdflist:
            os.remove(pdf)

def plot_2dhist_root(file_names:Sequence[str], set_names:Sequence[str], save_directory:str, varsx:Sequence[str], varsy:Sequence[str], pdf_name:str="some_2dhist.pdf",binsx:int=10,binsy:int=10,cuts:Union[str,None]=None,save_pdf:bool=False, save_file:bool=False, cmap=ROOT.kRainbow, logz:bool=True):

    dir_result=save_directory
    l1=varsx
    l2=varsy

    pdflist=[]

    for fname,sname in zip(file_names,set_names):
        root_file_hist=dir_result+sname+".root"
        for tu in list(itertools.product(l1, l2)):
            print(tu)
            hname=f"{tu[0]}_{tu[1]}"
            if not save_pdf:
                prep.plot_2dhist_root(file=fname, var1=tu[0], var2=tu[1], save_name_file=root_file_hist, hist_name=hname, save_name_pdf=dir_result+f"{sname+hname}.png",title=f"{sname}: {tu[0]} {tu[1]}",binsx=binsx, binsy=binsy,save_file=save_file, cuts=cuts,logz=logz)
                pdflist.append(dir_result+f"{sname+hname}.png")
                display_root_canvas(image_path=dir_result+f"{sname+hname}.png")
            else:
                prep.plot_2dhist_root(file=fname, var1=tu[0], var2=tu[1],  save_name_file=root_file_hist, hist_name=hname, save_name_pdf=dir_result+f"{sname+hname}.pdf",title=f"{sname}: {tu[0]} {tu[1]}",binsx=binsx, binsy=binsy, save_file=save_file,cuts=cuts,logz=logz)
                pdflist.append(dir_result+f"{sname+hname}.pdf")
    if save_pdf:
        merge_pdfs(pdf_list=pdflist, output_path=dir_result+pdf_name)
    else: 
        for pdf in pdflist:
            os.remove(pdf)

def plot_projection(set_name:str, hist_name:str,directory_hists:str,axis:int=0, pdfname:str="projection.pdf"):
    fname=directory_hists+set_name+".root"
    prep.add_projection(file_name=fname,hist_name=hist_name,save_name_pdf=directory_hists+pdfname,axis=axis)

def plot_3dhist_root(file_names:Sequence[str], set_names:Sequence[str], save_directory:str, varsx:Sequence[str], varsy:Sequence[str],varsz:Sequence[str],binsx:int=100, binsy:int=100, binsz:int=100,cuts:Union[str,None]=None, pdf_name:str="some_3dhist.pdf",save_pdf:bool=False, save_file:bool=False):

    dir_result=save_directory
    l1=varsx
    l2=varsy
    l3=varsz
    pdflist=[]

    for fname,sname in zip(file_names,set_names):
        root_file_hist=dir_result+sname+".root"
        for tu in list(itertools.product(l1, l2,l3)):
            print(tu)
            hname=f"{sname}_{tu[0]}_{tu[1]}_{tu[2]}"
            if not save_pdf:
                prep.plot_3dhist_root(file=fname, var1=tu[0], var2=tu[1], var3=tu[2], save_name_file=root_file_hist, hist_name=hname, save_name_pdf=dir_result+f"{hname}.png",title=f"{sname}: {tu[0]} {tu[1]} {tu[2]}",binsx=binsx, binsy=binsy, binsz=binsz,cuts=cuts, save_file=save_file)
                pdflist.append(dir_result+f"{hname}.png")
                display_root_canvas(image_path=dir_result+f"{hname}.png")
            else:
                prep.plot_3dhist_root(file=fname, var1=tu[0], var2=tu[1], var3=tu[2], save_name_file=root_file_hist, hist_name=hname, save_name_pdf=dir_result+f"{hname}.pdf",title=f"{sname}: {tu[0]} {tu[1]} {tu[2]}",binsx=binsx, binsy=binsy, binsz=binsz,cuts=cuts,save_file=save_file)
                pdflist.append(dir_result+f"{hname}.pdf")
    if save_pdf:
        merge_pdfs(pdf_list=pdflist, output_path=dir_result+pdf_name)
    else: 
        for pdf in pdflist:
            os.remove(pdf)


def scans(vals:Sequence[float],data:TreeHandler,var:str,vars_to_plot:Sequence[str],uppercuts:bool=True,base:float=None,filename:str="output_scan.pdf"):
    pdfs=[]
    if uppercuts:
        if base:
            for val in vals:
                data_cutted=prep.cut_data(data=data, var=var, upper_cut=val, lower_cut=base)
                plot_some_dist([data_cutted], to_plot=vars_to_plot,labels=[f"data cutted {var}: ({base},{val})"],file_name=f"cut_val{val}.pdf")
                pdfs.append(f"cut_val{val}.pdf")
            merge_pdfs(pdfs, output_path=filename)
        else:
            for val in vals:
                data_cutted=prep.cut_data(data=data, var=var, upper_cut=val)
                plot_some_dist([data_cutted], to_plot=vars_to_plot,labels=[f"data cutted {var}: (None,{val})"],file_name=f"cut_val{val}.pdf")
                pdfs.append(f"cut_val{val}.pdf")
            merge_pdfs(pdfs, output_path=filename)
    else:
        if base:
            for val in vals:
                data_cutted=prep.cut_data(data=data, var=var, upper_cut=base, lower_cut=val)
                plot_some_dist([data_cutted], to_plot=vars_to_plot,labels=[f"data cutted {var}: ({val},{base})"],file_name=f"cut_val{val}.pdf")
                pdfs.append(f"cut_val{val}.pdf")
            merge_pdfs(pdfs, output_path=filename)
        else:
            for val in vals:
                data_cutted=prep.cut_data(data=data, var=var, lower_cut=val)
                plot_some_dist([data_cutted], to_plot=vars_to_plot,labels=[f"data cutted {var}: ({val},None)"],file_name=f"cut_val{val}.pdf")
                pdfs.append(f"cut_val{val}.pdf")
            merge_pdfs(pdfs, output_path=filename)

def plot_scans_bdt(prompt:TreeHandler, nonprompt:TreeHandler, range:Sequence[float], no_points:int,var:str="trainBckgMC_class2"):
    values=np.arange(range[0],range[1], (range[1]-range[0])/no_points)
    myDict= prep.scans_bdt(vals=values[1:],prompt=prompt,nonprompt=nonprompt,base=range[0],uppercuts=True,var=var)
    fig, ax =plt.subplots()
    ax.scatter(values[1:], list(myDict.values()), marker="x")
    ax.set_xlabel("BDT score class 2 (prompt)")
    ax.set_ylabel("$\\frac{n(Nonprompt)<BDT}{n(Prompt)<BDT}$")
    ax.plot(range,len(range)*[1], color="black", linestyle="--")

    max_key = max(myDict, key=myDict.get)
    ax.plot([max_key,max_key],[0,max(list(myDict.values()))*1.1],color="red")
    arrowprops=dict(facecolor='black', shrink=0.05)
    ax.annotate(text=f"max at {max_key:.2f}",xytext=(0.3, 0.5),xy=(max_key,max(list(myDict.values()))*0.5),arrowprops=arrowprops)
    fig.suptitle("BDT-Scan:(Non-)prompt ratio")

    return max_key, max(list(myDict.values()))


def plot_comb_daughterPDG(var:Sequence[str], data:TreeHandler,save_dir_plots:str, pdf_filename:str="Dist_daughters.pdf"):
    pdfs=[]
    l1=list(set(prep.TreetoArray(data,var="fPDGCodeDauPos")))
    l2=list(set(prep.TreetoArray(data,var="fPDGCodeDauNeg")))
    
    for i,j in list(itertools.product(l1, l2)):
        print(i,j)
        st=prep.cut_data(data,var="fPDGCodeDauPos", upper_cut=i, lower_cut=i)
        plot_some_dist([st], to_plot=var, labels=[f"bckg_MC with PDGCodeDauPos={i} \n PDGCodeDauNeg={j}"],file_name=f"DistDau{i}_{j}.pdf")
        pdfs.append(save_dir_plots+f"DistDau{i}_{j}.pdf")
    merge_pdfs(pdfs,output_path=pdf_filename)

def get_allPDG(variable:str, data:TreeHandler):
    return set(prep.TreetoArray(data,var=variable))



def plot_chrystalball_fit(file_directory_data:str, x_min_fit:float=1.086,x_max_fit:float=1.14,x_min_data:float=1.05,x_max_data:float=1.16, savepdf:bool=False, var:str="fMass",save_name_root:str=f"data.root",nobins:int=200,cheb:bool=False,save_file:bool=False,logy:bool=True, fs:Tuple[float]=(8,5)):

    if cheb:
        pdfname=f"/home/katrin/Cern_summerProject/crystalball_fits/crystalball_bckg_cheb.png"
    else:
        pdfname=f"/home/katrin/Cern_summerProject/crystalball_fits/crystalball_bckg_poly.png"

    snf="/home/katrin/Cern_summerProject/crystalball_fits/"+save_name_root
    prep.fit_chrystalball(file_name=file_directory_data,save_name_file=snf,save_name_pdf=pdfname,x_max_data=x_max_data,x_min_data=x_min_data,x_max_fit=x_max_fit,x_min_fit=x_min_fit,var=var,no_bins=nobins,cheb=cheb,save_file=save_file,logy=logy)
    display_root_canvas(image_path=pdfname, fs=fs)
    if not savepdf:
        os.remove(pdfname)

def plot_histogram_fit(file_data:str, hist_data:str, file_model:str, hists_model:Sequence[str],title:str="title", png_name:str="hist_fit.png", savepng:bool=False, save_file:Union[str,None]=None,logy:bool=True, fs:Tuple[float]=(8,5)):

    prep.fit_histogram(file_data=file_data, hist_data=hist_data, file_model=file_model, hists_model=hists_model, title=title,save_name_file=save_file,save_name_pdf=png_name,logy=logy)
    display_root_canvas(image_path=png_name, fs=fs)
    if not savepng:
        os.remove(png_name)


def crystalball_fit_seperatedbins(file_directory_data:str,branch1:str="fMass", branch2:str="fPt", n_branch1:int=250, min_val_branch1:float=1.08, max_val_branch1:float=1.8, min_val_branch2:float=0.3,  max_val_branch2:float=4, binwidth_branch2:float=0.1, save_file:bool=False, hist2d_saved:bool=True, cheb:bool=False, logy:bool=True, save_pdf:bool=True, with_cut:bool=False):

    newbins=[0.5,1,1.5,2,2.5,3,4]

    if not hist2d_saved:
        if max_val_branch2==min_val_branch2:
            n=100
        else:
            n=int((max_val_branch2-min_val_branch2)/binwidth_branch2)
        print("n_bins= ", n)
        if with_cut:
            cut=str(input("Enter the cut: "))
            name_cut=str(input("Enter name of the cut: "))
            prep.plot_2dhist_root(file=file_directory_data, save_name_file=f"/home/katrin/Cern_summerProject/crystalball_fits/hist2d_{name_cut}.root", var2=branch2, var1=branch1, hist_name=f"2dhist_{branch1}_{branch2}", title=f"2dhist_{branch1}_{branch2}" ,binsx=n_branch1, binsy=n, miny=min_val_branch2, maxy=max_val_branch2, minx=1.07, maxx=1.18,cuts=cut, save_file=True)
            prep.new_bin_edges(file_name=f"/home/katrin/Cern_summerProject/crystalball_fits/hist2d_{name_cut}.root",hist_name=f"2dhist_{branch1}_{branch2}",new_bins=newbins, reb_y=True)
            prep.add_projection(file_name=f"/home/katrin/Cern_summerProject/crystalball_fits/hist2d_{name_cut}.root",hist_name=f"rebinned_hist",axis=0)
            prep.create_1d_histograms_from_2d(file_name=f"/home/katrin/Cern_summerProject/crystalball_fits/hist2d_{name_cut}.root",hist_name="rebinned_hist",already_saved=False )
        else:
            prep.plot_2dhist_root(file=file_directory_data, save_name_file=f"/home/katrin/Cern_summerProject/crystalball_fits/hist2d.root", var2=branch2, var1=branch1, hist_name=f"2dhist_{branch1}_{branch2}", title=f"2dhist_{branch1}_{branch2}" ,binsx=n_branch1, binsy=n, miny=min_val_branch2, maxy=max_val_branch2, minx=1.07, maxx=1.18,save_file=True)
            prep.new_bin_edges(file_name=f"/home/katrin/Cern_summerProject/crystalball_fits/hist2d.root",hist_name=f"2dhist_{branch1}_{branch2}",new_bins=newbins, reb_y=True)
            prep.add_projection(file_name=f"/home/katrin/Cern_summerProject/crystalball_fits/hist2d.root",hist_name=f"rebinned_hist",axis=0)
            prep.create_1d_histograms_from_2d(file_name=f"/home/katrin/Cern_summerProject/crystalball_fits/hist2d.root",hist_name="rebinned_hist",already_saved=False )
    else:
        if with_cut:
            name_cut=str(input("Enter name of the cut: "))        
    histograms=[]
    if with_cut:
        file = ROOT.TFile(f"/home/katrin/Cern_summerProject/crystalball_fits/hist2d_{name_cut}.root")
    else:
        file = ROOT.TFile(f"/home/katrin/Cern_summerProject/crystalball_fits/hist2d.root")
    for i in range(1,len(newbins)):
        hist = file.Get(f"hist_ybin_{i}")
        histograms.append(hist)
    pdflist=[]
    for i in range(len(histograms)):
        if save_pdf:
            if with_cut:
                pdfname=f"/home/katrin/Cern_summerProject/crystalball_fits/perPtbin/{branch2}Fit_{name_cut}_{branch1}bin_{i}.pdf"
                pdfname_manuel=f"/home/katrin/Cern_summerProject/crystalball_fits/perPtbin/{branch2}Fit_{name_cut}_{branch1}bin_{i}_manuel.pdf"
                pdflist.append(pdfname)
                pdfname_final=f"/home/katrin/Cern_summerProject/crystalball_fits/perPtbin/fits_perPtBin_{name_cut}.pdf"
            else:
                pdfname=f"/home/katrin/Cern_summerProject/crystalball_fits/perPtbin/{branch2}Fit_{branch1}bin_{i}.pdf"
                pdfname_manuel=f"/home/katrin/Cern_summerProject/crystalball_fits/perPtbin/{branch2}Fit_{branch1}bin_{i}_manuel.pdf"
                pdflist.append(pdfname)
                pdfname_final=f"/home/katrin/Cern_summerProject/crystalball_fits/perPtbin/fits_perPtBin.pdf"
        else:
            pdfname=f"/home/katrin/Cern_summerProject/crystalball_fits/perPtbin/{branch2}Fit_{branch1}bin_{i}.png"
            pdflist.append(pdfname)

        if with_cut:
            snf=f"/home/katrin/Cern_summerProject/crystalball_fits/perPtbin_{name_cut}/{branch2}Fit_{branch1}bin_{i}.root"
        else:
            snf=f"/home/katrin/Cern_summerProject/crystalball_fits/perPtbin/{branch2}Fit_{branch1}bin_{i}.root"
        prep.fit_chrystalball(file_name=file_directory_data, save_name_file=snf,  save_name_pdf=pdfname, hist_given=histograms[i], x_max_data=max_val_branch1,x_min_data=min_val_branch1,x_max_fit=1.14,x_min_fit=1.086,var=branch1,cheb=cheb,save_file=save_file,logy=logy, title=f"Fit for Pt bin [{newbins[i]},{newbins[i+1]}] GeV")
    if save_pdf:
        merge_pdfs(pdf_list=pdflist,output_path=pdfname_final)
    else:
        for pdf in pdflist:
            display_root_canvas(image_path=pdf)
            os.remove(pdf)


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


def display_root_canvas(image_path, fs:Tuple[float]=(8,5)):
    # Read the image
    img = mpimg.imread(image_path)
    
    # Display the image using imshow
    plt.figure(figsize=fs)
    plt.imshow(img)
    plt.axis('off')  # Hide the axis
    plt.show()

def apply_cuts(data:TreeHandler, var:str, lower_cut:Union[float,None]=None,upper_cut:Union[float,None]=None, inclusive:bool=True)->TreeHandler:
    return prep.cut_data(data=data, var=var,lower_cut=lower_cut,upper_cut=upper_cut,inclusive=inclusive)

def get_TreeHandler(file_name:str, tree_name:str, folder_name:str, is_largeFile:bool=False):
    
    if is_largeFile:
        data=TreeHandler()
        data.get_handler_from_large_file(file_name=file_name, tree_name=tree_name)
        return data
    else:
        data=TreeHandler(file_name, tree_name, folder_name=folder_name)
        return data

# Function to save the desired subsets
def save_sets(sets:Sequence[TreeHandler], set_names:Sequence[str], dir_tree:str, tree:str="tree"):
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
        prep.get_root_from_TreeHandler(treehdl=i, save_dir=dir_tree, output_name=f"{name}.root",treename=tree)

def prepare_hists_simFit(file_data:str, tree_data:str, model:ModelHandler, no_bins:int=50, fname:str="DF*", save_name_file:str="hists_simFit.root"):
    
    #minmass, maxmass=prep.get_minmax_of_tree(file_name=file_data, branch1="fMass")
    histmass = ROOT.TH1F("histmass", "Mass Data", no_bins, 1.086, 1.14)  # Adjust binning and range as necessary
    histbdt = ROOT.TH1F("histbdt", "Bdt Data", no_bins, 0 ,1)  # Adjust binning and range as necessary

    data=prep.cut_data(data=prep.get_rawdata(file_directory_data=file_data, tree_data=tree_data,folder_name=fname),var="fMass", lower_cut=1.086, upper_cut=1.14)
    df=data.get_data_frame()
    mass_array=df["fMass"]
    for mass in mass_array:
        histmass.Fill(mass)
    
    pred=model.predict(data, output_margin=False)
    pred_class2=[]

    for i in range(len(pred)):
        ls=pred[i]
        pred_class2.append(ls[2])
    
    for bdt in pred_class2:
        histbdt.Fill(bdt)

    output_file = ROOT.TFile(save_name_file, "RECREATE")
    histmass.Write()
    histbdt.Write()
    output_file.Close()

    

def plot_simFit(file_hist_data:str, hist_bdt:str, hist_mass:str, file_model:str, hists_model:Sequence[str],  png_name:str="simFit.png",fs:Tuple[float]=(10,6),no_bins:int=50,fname:str="DF*"):
    

    rfit.fit_simultanously(file_hist_data=file_hist_data, hist_bdt=hist_bdt,hist_mass=hist_mass, file_model=file_model, hists_model=hists_model,save_name_file="simFit.root")
    display_root_canvas(image_path="/home/katrin/Cern_summerProject/"+png_name, fs=fs)



#def analysis_cutted_subsets(already_saved:bool=True, onlynewMC:bool=False):
#
#    allsets=get_sets(already_saved=already_saved, onlynewMC=onlynewMC)
#    sets=list(allsets.values())
#    set_names=list(allsets.keys())
#    files={}
#    for nm in set_names:
#        files[nm]=directory_sets+nm+".root"
#    vars=[st.get_var_names() for st in sets]
#    vars_MC=list(allsets["bckg_MC"].get_var_names())
#    vars_shared=[var for var in vars[0] if all(var in sublist for sublist in vars)]
#

    #plot_some_dist([bckgMC_cutted_training_low, allsets["bckg_MC"]],to_plot=vars_shared+["trainBckgMC_class0","trainBckgMC_class1","trainBckgMC_class2"],labels=["lowcutted_bckg_MC","bckg_MC"],file_name="hist_bckgMC_trainlow.pdf",severalpages=True)
    #plot_some_dist([bckgMC_cutted_training_high, allsets["bckg_MC"]],to_plot=vars_shared+["trainBckgMC_class0","trainBckgMC_class1","trainBckgMC_class2"],labels=["upcutted_bckg_MC","bckg_MC"],file_name="hist_bckgMC_trainhigh.pdf",severalpages=True)
    #plot_some_dist([bckgdata_cutted_training_low, allsets["bckg_data"]],to_plot=vars_shared+["trainBckgdata_class0","trainBckgdata_class1","trainBckgdata_class2"],labels=["lowcutted_bckg_data","bckg_data"],file_name="hist_bckgdata_trainlow.pdf",severalpages=True)
    #plot_some_dist([bckgdata_cutted_training_high, allsets["bckg_data"]],to_plot=vars_shared+["trainBckgdata_class0","trainBckgdata_class1","trainBckgdata_class2"],labels=["upcutted_bckg_data","bckg_data"],file_name="hist_bckgdata_trainhigh.pdf",severalpages=True)

    #plot_some_dist([allsets["bckg_MC_cuttedRadius"],allsets["prompt_cuttedRadius"],allsets["nonprompt_cuttedRadius"]],to_plot=vars_shared+["trainBckgMC_cuttedRadius_class0","trainBckgMC_cuttedRadius_class1","trainBckgMC_cuttedRadius_class2"],labels=["bckg_MC_cuttedRadius","prompt_cuttedRadius","nonprompt_cuttedRadius"],file_name="hist_mlsetsMC_cuttedRadius.pdf",severalpages=True)
    #plot_some_dist([allsets["bckg_MC_cuttedMassRadius"],allsets["prompt_cuttedRadius"],allsets["nonprompt_cuttedRadius"]],to_plot=vars_shared+["trainBckgMC_cuttedMassRadius_class0","trainBckgMC_cuttedMassRadius_class1","trainBckgMC_cuttedMassRadius_class2"],labels=["bckg_MC_cuttedMassRadius","prompt_cuttedRadius","nonprompt_cuttedRadius"],file_name="hist_mlsetsMC_cuttedMassRadius.pdf",severalpages=True)
    #plot_some_dist([allsets["bckg_MC_cuttedMass"],allsets["prompt"],allsets["nonprompt"]],to_plot=vars_shared+["trainBckgMC_cuttedMass_class0","trainBckgMC_cuttedMass_class1","trainBckgMC_cuttedMass_class2"]+["trainBckgMC_cuttedMass_withCt_class0","trainBckgMC_cuttedMass_withCt_class1","trainBckgMC_cuttedMass_withCt_class2"],labels=["bckg_MC_cuttedMass","prompt","nonprompt"],file_name="hist_mlsetsMC_cuttedMass_withCt.pdf",severalpages=True)

    #for i in (0,1,2):
    #    prep.rename_column(file_name="bckgdata_cuttedRadius_high.root", file_dir=directory_sets, old_column_name=f"trainBckgdata_cutR_class{i}", new_column_name=f"trainBckgdata_cutR_high_class{i}")
    #    prep.rename_column(file_name="bckgMC_cuttedRadius_high.root", file_dir=directory_sets, old_column_name=f"trainBckgMC_cutR_class{i}", new_column_name=f"trainBckgMC_cutR_high_class{i}")

 
    #values=np.round(np.arange(start=-2,stop=1,step=0.1),2)
    #scans(vals=values,data=allsets["bckg_MC"], var="trainBckgMC_class0",vars_to_plot=["fMass","fDcaV0PV","fDcaPosPV","fDcaNegPV","fDcaPVProton","fDcaPVPion","fPDGCodeDauNeg","fPDGCodeDauPos","trainBckgMC_class0"])
    #plot_some_dist([allsets["bckg_MC_low"],allsets["bckg_MC"]],to_plot=["fMass","fDcaV0PV","fDcaPosPV","fDcaNegPV","fDcaPVProton","fDcaPVPion","fPDGCodeDauNeg","fPDGCodeDauPos","trainBckgMC_class0"],labels=["bckg_MC_lowBDT","bckg_MC_full"],severalpages=True)
    #plot_comb_daughterPDG(data=allsets["bckg_MC_low"], var="fMass")
    #st=prep.cut_data(data=prep.cut_data(allsets["bckg_MC"],var="fPDGCodeDauPos", lower_cut=2212,upper_cut=2212,inclusive=False),var="fPDGCodeDauPos",lower_cut=-13,upper_cut=-13,inclusive=False)
    #plot_some_dist(sets=[st,allsets["bckg_MC"]],to_plot=vars_shared+["trainBckgMC_class0"],labels=["",""],severalpages=True)
    #plot_2dhist_root(file_names=[files["bckg_MC"],files["prompt"]], set_names=["bckg_MC","prompt"],varsx=["fPDGCodeMotherDauNeg"],varsy=["fPDGCodeMotherDauPos"])

    #plot_some_dist(sets=[allsets["bckg_MC"],allsets["bckg_MC_filtered"]], labels=["bckg MC", "bckg MC w/ signal"],to_plot=vars_shared,file_name="dist_features_MC.pdf",no_bins=33)
    #plot_some_corr(sets=[allsets["matter_bckg_MC"],allsets["antimatter_bckg_MC"]], labels=["matter bckg MC", "antimatter bckg MC"],to_plot=list(allsets["matter_bckg_MC"].get_var_names()),file_name="corr_bckg_MC_matter.pdf")
    #plot_3dhist_root(file_names=[files["prompt"]], set_names=["prompt"],varsx=["fPDGCodeMotherDauNeg"],varsy=["fPDGCodeMotherDauPos"],varsz=["fMass"],pdf_name="3d_hist_prompt.pdf")

    #for name in ["matter_bckg_MC", "antimatter_bckg_MC"]:
    #    plot_2dhist_root(file_names=[files[name]], set_names=[name],varsx=["fPDGCodeMotherDauNeg"],varsy=["fPDGCodeMotherDauPos"],pdf_name=f"2d_hist_codedau_{name}.pdf",bins=400)
    #    plot_2dhist_root(file_names=[files[name]], set_names=[name],varsx=["fPDGCodeMotherDauNeg"],varsy=["fPDGMatchMotherSecondMother"],pdf_name=f"2d_hist_codedauneg_matchsecmoth_{name}.pdf",bins=100)
    #    plot_2dhist_root(file_names=[files[name]], set_names=[name],varsx=["fPDGCodeMotherDauPos"],varsy=["fPDGMatchMotherSecondMother"],pdf_name=f"2d_hist_codedaupos_matchsecmoth_{name}.pdf",bins=100)
#
    #    plot_2dhist_root(file_names=[files[name]], set_names=[name],varsx=["fPDGCodeMotherDauNeg","fPDGCodeMotherDauPos"],varsy=["fMass","fPDGMatchMotherSecondMother"],pdf_name=f"2d_hist_codedau_mass_{name}.pdf",bins=400)
    #    plot_3dhist_root(file_names=[files[name]], set_names=[name],varsx=["fPDGCodeMotherDauNeg"],varsy=["fPDGCodeMotherDauPos"],varsz=["fMass"],binsx=400,binsy=400, pdf_name=f"3d_hist_fMass_{name}.pdf")
    #    plot_3dhist_root(file_names=[files[name]], set_names=[name],varsx=["fPDGCodeMotherDauNeg"],varsy=["fPDGCodeMotherDauPos"],varsz=["fPDGMatchMotherSecondMother"],binsx=100,binsy=100, binsz=100,pdf_name=f"3d_hist_fMatchSecMoth_{name}.pdf")
#
#    
    #cutted_MC=prep.cut_data(data=allsets["bckg_MC"],var="fPDGMatchMotherSecondMother",lower_cut=-999,upper_cut=-999)
#    plot_some_dist(sets=[allsets["bckg_MC"],bckg_MC_filtered],to_plot=["fMass"],labels=["bckg_MC", "bckg_MC wo secMother Match"],file_name="mass_dist_bckg_MC.pdf")
#    plot_2dhist_root(file_names=["/home/katrin/Cern_summerProject/data/AO2D_data_new.root"],set_names=["data"], varsx=["fMass"],varsy=["fCosPA","fPt"],pdf_name="2dhistdata.pdf",bins=500,folders=True)
 #   plot_3dhist_root(file_names=["/home/katrin/Cern_summerProject/data/AO2D_data_new.root"],set_names=["data"], varsx=["fMass"],varsy=["fCosPA"],varsz=["fPt"],pdf_name="3dhistdata.pdf",binsx=100,binsy=100,binsz=100,folders=True)
    #plot_projection(set_name="data", hist_name="data_fMass_fCosPA;2",axis=0,pdfname="projection_fCosPA.pdf")
    #bad_bckg_MC=prep.cut_data(data=allsets["bckg_MC_masscut"],var="trainBckgMC_masscut_class0", upper_cut=0.8)
    #good_bckg_MC=prep.cut_data(data=allsets["bckg_MC_masscut"],var="trainBckgMC_masscut_class0", lower_cut=0.8)

    #plot_some_dist(sets=[bad_bckg_MC,good_bckg_MC], to_plot=vars_shared+["trainBckgMC_masscut_class0"], labels=["bad_bckg_MC","good_bckg_MC"],file_name="dist_bad_good_bkg.pdf",severalpages=True)
    #plot_some_dist(sets=[allsets["bckg_MC_masscut_radiuscut"],allsets["nonprompt_radiuscut"],allsets["prompt_radiuscut"]],to_plot=vars_shared,labels=["bkcg_MC_masscut_radiuscut","nonprompt_radiuscut","prompt_radiuscut"], file_name="distr_cuttedRadius.pdf")

    #mc=prep.get_rawMC_data(file_directory_mc=file_directory_mc,tree_mc=tree_mc,folder_name=fname)
    #data=prep.get_rawdata(file_directory_data=file_directory_data,tree_data=tree_data,folder_name=fname)

    #leftsideband_MC=prep.cut_data(data=data, var="fMass", upper_cut=1.1)
    #signal_MC=prep.cut_data(data=data, var="fMass", lower_cut=1.11, upper_cut=1.1208)
    #rightsideband_MC=prep.cut_data(data=data, var="fMass", lower_cut=1.13)
    #bump_MC=prep.cut_data(data=allsets["bckg_MC"], var="fMass", lower_cut=1.13,upper_cut=1.16)
    #prep.save_sets(sets=[leftsideband_MC,rightsideband_MC,signal_MC], set_names=["leftsideband_MC","rightsideband_MC","signal_MC"],dir_tree=directory_sets)
    #cut="fMass < 1.1 && fMass > 1.1208 && fQtAp < 0.12"

    #cut="fCosPA> 0.999"
    #plot_2dhist_root(file_names=[file_directory_data], set_names=["signal"],varsx=["fAlphaAP"],varsy=["fQtAP"],pdf_name=f"Armenteros_signal.pdf",binsx=200,binsy=200,cuts=cut)
    #for name in ["rightsideband_MC", "leftsideband_MC","signal_MC"]:
    #    files[name]=directory_sets+name+".root"
    #    plot_2dhist_root(file_names=[files[name]], set_names=[name],varsx=["fAlphaAP"],varsy=["fQtAP"],pdf_name=f"Armenteros_{name}.pdf",binsx=100, binsy=100)
    #plot_some_dist(sets=[allsets["bump_MC"]],to_plot=vars_MC, labels=["bump_MC"], file_name="dist_Massbump_MC.pdf")

    #exp_NOmassBump_data=prep.cut_data(data=data, var="fQtAP", upper_cut=0.12)
    #plot_some_dist(sets=[data,exp_NOmassBump_data], to_plot=["fMass"], labels=["data","data_ArmenterosNOK0_region"])
    #data=prep.get_rawdata(file_directory_data=file_directory_data, tree_data=tree_data, folder_name=fname)
    #plot_some_dist(sets=[allsets["bckg_MC"],data], to_plot="fMass",labels=["bckg_MC","data"],severalpages=1)
#analysis_cutted_subsets(already_saved=True, onlynewMC=True)
#get_sets(already_saved=True, onlynewMC=True)
#plot_bck_cuts() 
#analysis_cutted_subsets()

#plot_chrystalball_fit( savepdf=True,cheb=True,logy=True)
#plot_chrystalball_fit( savepdf=True,cheb=False,logy=True)
#crystalball_fit_seperatedbins(hist2d_saved=True,cheb=True,with_cut=True)
#plot_2dhist_root(file_names=[file_directory_data])
#plot_chrystalball_fit(file_directory_data="/home/katrin/Cern_summerProject/data/AO2D_data_woK0.root")


#crystalball_fit_seperatedbins(file_directory_data="/home/katrin/Cern_summerProject/data/AO2D_data_woK0.root", hist2d_saved=False, save_pdf=False, cheb=False, with_cut=False)