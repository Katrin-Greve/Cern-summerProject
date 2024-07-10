import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import preparing_data as prep
from hipe4ml import plot_utils
import itertools
from hipe4ml.tree_handler import TreeHandler
from typing import Union, Sequence, List
import seaborn as sns 
import os
import numpy as np

no_set=4
directory_sets=f"/home/katrin/Cern_summerProject/root_trees/set_{no_set}/"
directory_hists=f"/home/katrin/Cern_summerProject/root_histograms/set_{no_set}/"
save_dir_plots = f'/home/katrin/Cern_summerProject/imgs/set_{no_set}/'


if no_set==1:
    file_directory_data="/home/katrin/Cern_summerProject/data/AnalysisResults_treesML_data.root"
    tree_data="O2lambdatableml"
    file_directory_mc="/home/katrin/Cern_summerProject/data/AnalysisResults_treesML.root"
    tree_mc="O2mclambdatableml"
    fname=None

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

if no_set==4:
    file_directory_data="/home/katrin/Cern_summerProject/data/AO2D_data_new.root"
    tree_data="O2lambdatableml"
    file_directory_mc="/home/katrin/Cern_summerProject/data/AO2D_MC_mothdau.root"
    tree_mc="O2mclambdatableml"
    fname="DF*"

# function to get the usual needed data sets
def get_sets(already_saved:bool=True, onlynewMC:bool=False):
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
        matter, antimatter=prep.filter_posneg_Pt(data=bckg_MC)
        prep.save_sets(sets=[matter,antimatter],set_names=["matter_bckg_MC", "antimatter_bckg_MC"],dir_tree=directory_sets)
        prep.save_sets([bckg_MC,prompt,nonprompt],set_names=["bckg_MC","prompt","nonprompt"],dir_tree=directory_sets)
        print("MC Subsets saved!")


    elif already_saved==False:
        print("sets not saved yet")
        prompt, nonprompt, bckg_data, bckg_MC=prep.get_base_sets(file_directory_data=file_directory_data, file_directory_mc=file_directory_mc, tree_data=tree_data, tree_mc=tree_mc, fname=fname)
        
        #get Mass cutted Background
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

# function to visualize, how one gets the background out of the data by using cuts on the mass
def plot_bck_cuts():

    data=prep.get_rawdata(file_directory_data,tree_data, folder_name=fname)
    pdf_filename = save_dir_plots+'prep_MLdata_bckg.pdf'
    pdf = PdfPages(pdf_filename) 
    prep.plot_hist(data,"fMass",leg_labels="raw data", fs=(8,3),alpha=0.5)

    cu2=prep.fitplot_gauss_rec(data, var="fMass",p0=[300000,1.115,0.005,1000],fs=(8,3),sig_cut=13)[2]
    bckg=prep.get_bckg(file_directory_data, tree_data, cuts=cu2, folder_name=fname)

    prep.plot_hist(bckg,"fMass",leg_labels="bckg",fs=(8,3),alpha=0.5)

    for fig_num in plt.get_fignums():
        fig = plt.figure(fig_num)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
    pdf.close()

#function that shows the distribution of the MC data, the prompt and the nonprompt distribution
def plot_prompt_nonprompt():
    
    data=prep.get_rawMC_data(file_directory_mc, tree_mc ,folder_name=fname)
    pdf_filename =save_dir_plots+'prep_MLdata_prompt.pdf'
    pdf = PdfPages(pdf_filename)

    prompt= prep.get_prompt(file_directory_mc, tree_mc,folder_name=fname)
    nonprompt=prep.get_nonprompt(file_directory_mc, tree_mc,folder_name=fname)
    

    prep.plot_hist(data,"fMass",leg_labels="raw MC", fs=(8,3),alpha=0.5)
    prep.plot_hist(prompt,"fMass",leg_labels="prompt",fs=(8,3),alpha=0.5)
    prep.plot_hist(nonprompt,"fMass",leg_labels="nonprompt",fs=(8,3),alpha=0.5)

    for fig_num in plt.get_fignums():
        fig = plt.figure(fig_num)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    pdf.close()



def plot_allvar_dist(already_saved:bool=True):

    allsets=get_sets(already_saved=already_saved)
    sets=list(allsets.values())
    set_names=list(allsets.keys())
    
    
    vars=[st.get_var_names() for st in sets]
    vars_shared=[var for var in vars[0] if all(var in sublist for sublist in vars)]

    pdf_filename = save_dir_plots+'histograms.pdf'
    pdf = PdfPages(pdf_filename)
    prep.plot_hist([allsets["prompt"],allsets["nonprompt"],allsets["bckg_data"]], vars_to_draw=vars_shared,leg_labels=["prompt", "nonprompt","bckg_data"],fs=(15,10),alpha=0.3)

    colors=sns.color_palette(n_colors=len(set_names))
    for i in (0,2,4):
        print(set_names[i])
        prep.plot_hist(sets[i], vars_to_draw=vars[i],leg_labels=set_names[i], fs=(15,10) ,alpha=0.3,colors=colors[i])
    for i in (1,3,5):
        print(set_names[i])
        prep.plot_hist(sets[i], vars_to_draw=vars_shared+["fGenPt","fGenCt","fGenRadius"],leg_labels=set_names[i], fs=(15,10) ,alpha=0.3,colors=colors[i])

    for fig_num in plt.get_fignums():
        fig = plt.figure(fig_num)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
    pdf.close()

def plot_some_dist(sets:Sequence[TreeHandler], to_plot:Sequence[str],labels:Sequence[str], file_name:str="output_someDist.pdf",severalpages:bool=False,no_bins:int=100):

    pdf_filename = save_dir_plots+file_name
    pdf = PdfPages(pdf_filename)
    if severalpages:
        colors=sns.color_palette(n_colors=len(sets))
        prep.plot_hist(sets, vars_to_draw=to_plot,fs=(15,10),leg_labels=labels,alpha=0.3,colors=colors,no_bins=no_bins)
        for set,lab,col in zip(sets,labels,colors):
            prep.plot_hist(set, vars_to_draw=to_plot,fs=(15,10),leg_labels=lab,alpha=0.3,colors=col,no_bins=no_bins)
    else:
        prep.plot_hist(sets, vars_to_draw=to_plot,fs=(15,10),leg_labels=labels,alpha=0.3,no_bins=no_bins)

    for fig_num in plt.get_fignums():
        fig = plt.figure(fig_num)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
    pdf.close()

def plot_corr(already_saved:bool=True):

    allsets=get_sets(already_saved=already_saved)
    sets=list(allsets.values())
    names=list(allsets.keys())

    vars=[st.get_var_names() for st in sets]
    pdf_filename = save_dir_plots+"corr.pdf"
    pdf = PdfPages(pdf_filename)
    for st, variables, nm in zip(sets, vars, names):
        plot_utils.plot_corr([st], columns=variables, labels=[nm])
    for fig_num in plt.get_fignums():
        fig = plt.figure(fig_num)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
    pdf.close()


def plot_some_corr(sets:Sequence[TreeHandler],  labels:Sequence[str], to_plot:Union[Sequence[str],None]=None,file_name:str="output_someCorr.pdf"):

    
    pdf_filename = save_dir_plots+file_name
    pdf = PdfPages(pdf_filename)
    if to_plot:
        plot_utils.plot_corr(sets,columns=to_plot,labels=labels)
    else:
        for st, lab in zip(sets, labels):
            cols=[var for var in st.get_var_names()]
            plot_utils.plot_corr([st],columns=cols,labels=[lab])
    for fig_num in plt.get_fignums():
        fig = plt.figure(fig_num)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
    pdf.close()


def plot_2dhist_numpy(already_saved:bool=True):

    allsets=get_sets(already_saved=already_saved)
    sets=list(allsets.values())
    set_names=list(allsets.keys())

    l1=["fDcaV0PV", "fCosPA"]
    l2=["fCt","fPt"]

    pdf_filename = save_dir_plots+'histograms2d_numpy.pdf'
    pdf = PdfPages(pdf_filename)

    for (i,name) in zip(sets,set_names):
        fig,ax=plt.subplots(len(l1),len(l2),figsize=(10,10))
        for ax_i, tu in zip(ax.T.flatten(),list(itertools.product(l1, l2))):
            prep.plot_2dhist_numpy(i,tu[0],tu[1],ax=ax_i)
            ax_i.set(xlabel=tu[0],ylabel=tu[1])
            if tu[0]=="fCosPA":
                ax_i.set_xlim(0.995,1)
        ax[0][0].set_title(name+f", size of sample: {prep.get_sampleSize(i)}", fontsize=20)

    for fig_num in plt.get_fignums():
        fig = plt.figure(fig_num)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
    pdf.close()



def plot_2dhist_root(file_names:Sequence[str], set_names:Sequence[str], varsx:Sequence[str], varsy:Sequence[str], pdf_name:str="some_2dhist.pdf",bins:int=10,folders:bool=False):

    tree="tree"
    dir_result=directory_hists
    l1=varsx
    l2=varsy

    pdflist=[]

    for fname,sname in zip(file_names,set_names):
        save_dir=dir_result+sname+".root"
        for tu in list(itertools.product(l1, l2)):
            print(tu)
            hname=f"{tu[0]}_{tu[1]}"
            prep.plot_2dhist_root(file=fname, tree_name=tree, var1=tu[0], var2=tu[1], save_name_file=save_dir, hist_name=hname, save_name_pdf=dir_result+f"{hname}.pdf",title=f"{sname}: {tu[0]} {tu[1]}",bins=bins,folders=folders)
            pdflist.append(dir_result+f"{hname}.pdf")

    prep.merge_pdfs(pdf_list=pdflist,output_path=dir_result+pdf_name)

def plot_projection(set_name:str, hist_name:str,axis:int=0, pdfname:str="projection.pdf"):
    fname=directory_hists+set_name+".root"
    prep.add_projection(file_name=fname,hist_name=hist_name,save_name_pdf=directory_hists+pdfname,axis=axis)

def plot_3dhist_root(file_names:Sequence[str], set_names:Sequence[str], varsx:Sequence[str], varsy:Sequence[str],varsz:Sequence[str],binsx:int=100, binsy:int=100, binsz:int=100, pdf_name:str="some_2dhist.pdf",folders:bool=False):

    tree="tree"
    dir_result=directory_hists
    bins=100
    l1=varsx
    l2=varsy
    l3=varsz
    pdflist=[]

    for fname,sname in zip(file_names,set_names):
        save_dir=dir_result+sname+".root"
        for tu in list(itertools.product(l1, l2,l3)):
            print(tu)
            hname=f"{sname}_{tu[0]}_{tu[1]}_{tu[2]}"
            prep.plot_3dhist_root(file=fname, tree_name=tree, var1=tu[0], var2=tu[1], var3=tu[2], save_name_file=save_dir, hist_name=hname, save_name_pdf=dir_result+f"{hname}.pdf",title=f"{sname}: {tu[0]} {tu[1]} {tu[2]}",binsx=binsx, binsy=binsy, binsz=binsz,folders=folders)
            pdflist.append(dir_result+f"{hname}.pdf")

    prep.merge_pdfs(pdf_list=pdflist,output_path=dir_result+pdf_name)


def scans(vals:Sequence[float],data:TreeHandler,var:str,vars_to_plot:Sequence[str], uppercuts:bool=True,base:float=None,filename:str="output_scan.pdf"):
    pdfs=[]
    if uppercuts:
        if base:
            for val in vals:
                data_cutted=prep.cut_data(data=data, var=var, upper_cut=val, lower_cut=base)
                plot_some_dist([data_cutted], to_plot=vars_to_plot,labels=[f"data cutted {var}: ({base},{val})"],file_name=f"cut_val{val}.pdf")
                pdfs.append(save_dir_plots+f"cut_val{val}.pdf")
            prep.merge_pdfs(pdfs, output_path=save_dir_plots+filename)
        else:
            for val in vals:
                data_cutted=prep.cut_data(data=data, var=var, upper_cut=val)
                plot_some_dist([data_cutted], to_plot=vars_to_plot,labels=[f"data cutted {var}: (None,{val})"],file_name=f"cut_val{val}.pdf")
                pdfs.append(save_dir_plots+f"cut_val{val}.pdf")
            prep.merge_pdfs(pdfs, output_path=save_dir_plots+filename)
    else:
        if base:
            for val in vals:
                data_cutted=prep.cut_data(data=data, var=var, upper_cut=base, lower_cut=val)
                plot_some_dist([data_cutted], to_plot=vars_to_plot,labels=[f"data cutted {var}: ({val},{base})"],file_name=f"cut_val{val}.pdf")
                pdfs.append(save_dir_plots+f"cut_val{val}.pdf")
            prep.merge_pdfs(pdfs, output_path=save_dir_plots+filename)
        else:
            for val in vals:
                data_cutted=prep.cut_data(data=data, var=var, lower_cut=val)
                plot_some_dist([data_cutted], to_plot=vars_to_plot,labels=[f"data cutted {var}: ({val},None)"],file_name=f"cut_val{val}.pdf")
                pdfs.append(save_dir_plots+f"cut_val{val}.pdf")
            prep.merge_pdfs(pdfs, output_path=save_dir_plots+filename)

def plot_comb_daughterPDG(var:Sequence[str],data:TreeHandler):
    pdfs=[]
    l1=list(set(prep.TreetoArray(data,var="fPDGCodeDauPos")))
    l2=list(set(prep.TreetoArray(data,var="fPDGCodeDauNeg")))
    
    for i,j in list(itertools.product(l1, l2)):
        print(i,j)
        st=prep.cut_data(data,var="fPDGCodeDauPos", upper_cut=i, lower_cut=i)
        plot_some_dist([st], to_plot=var, labels=[f"bckg_MC with PDGCodeDauPos={i} \n PDGCodeDauNeg={j}"],file_name=f"DistDau{i}_{j}.pdf")
        pdfs.append(save_dir_plots+f"DistDau{i}_{j}.pdf")
    prep.merge_pdfs(pdfs,output_path=save_dir_plots+"Dist_daughterpairs.pdf")

def plot_chrystalball_fit(x_max:float=1.3,x_min:float=1,folders:bool=True,savepdf:bool=False, var:str="fMass",save_name_root:str=f"data_{no_set}.root"):

    if savepdf:
        pdfname=f"/home/katrin/Cern_summerProject/crystalball_fits/crystalball_set{no_set}.pdf"
    else: 
        pdfname=False
    snf="/home/katrin/Cern_summerProject/crystalball_fits/"+save_name_root
    prep.fit_chrystalball(file_name=file_directory_data,tree_name=tree_data,save_name_file=snf,save_name_pdf=pdfname,x_max=x_max,x_min=x_min,folders=folders,var=var)

def analysis_cutted_subsets(already_saved:bool=True, onlynewMC:bool=False):

    allsets=get_sets(already_saved=already_saved, onlynewMC=onlynewMC)
    sets=list(allsets.values())
    set_names=list(allsets.keys())
    files={}
    for nm in set_names:
        files[nm]=directory_sets+nm+".root"
    vars=[st.get_var_names() for st in sets]
    vars_shared=[var for var in vars[0] if all(var in sublist for sublist in vars)]


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

#analysis_cutted_subsets()
#plot_bck_cuts() 

plot_chrystalball_fit(x_max=1.18,x_min=1.05,savepdf=True)
