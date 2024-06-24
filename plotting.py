import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import preparing_data as prep
from hipe4ml import plot_utils
import itertools
from hipe4ml.tree_handler import TreeHandler
from typing import Union, Sequence, List
import seaborn as sns 
import os

no_set=3
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


# function to get the usual needed data sets
def get_sets(already_saved:bool=True):
    """
    Get the usual needed data sets

    Parameters:
        already_saved (bool): Decides whether the data sets have to be calculated or are already saved as a root file. Default to True
    """
    if already_saved==False:
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

    cu2=prep.fitplot_gauss_rec(data, var="fMass",p0=[300000,1.115,0.005,1000],fs=(8,3),sig_cut=11)[2]
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

def plot_some_dist(sets:Sequence[TreeHandler], to_plot:Sequence[str],labels:Sequence[str], file_name:str="output_someDist.pdf",severalpages:bool=False):

    pdf_filename = save_dir_plots+file_name
    pdf = PdfPages(pdf_filename)
    if severalpages:
        colors=sns.color_palette(n_colors=len(sets))
        prep.plot_hist(sets, vars_to_draw=to_plot,fs=(15,10),leg_labels=labels,alpha=0.3,colors=colors)
        for set,lab,col in zip(sets,labels,colors):
            prep.plot_hist(set, vars_to_draw=to_plot,fs=(15,10),leg_labels=lab,alpha=0.3,colors=col)
    else:
        prep.plot_hist(sets, vars_to_draw=to_plot,fs=(15,10),leg_labels=labels,alpha=0.3)

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



def plot_2dhist_root(file_names:Sequence[str], set_names:Sequence[str], varsx:Sequence[str], varsy:Sequence[str], pdf_name:str="some_2dhist.pdf", save_dir:str="output.root"):

    tree="tree"
    dir_result=directory_hists
    bins=100
    l1=varsx
    l2=varsy

    pdflist=[]

    for fname,sname in zip(file_names,set_names):
        save_dir=dir_result+sname+".root"
        for tu in list(itertools.product(l1, l2)):
            print(tu)
            hname=f"{sname}_{tu[0]}_{tu[1]}"
            prep.plot_2dhist_root(file=fname, tree_name=tree, var1=tu[0], var2=tu[1], save_name_file=save_dir, hist_name=hname, save_name_pdf=dir_result+f"{hname}.pdf",title=f"{sname}: {tu[0]} {tu[1]}",bins=bins)
            pdflist.append(dir_result+f"{hname}.pdf")

    prep.merge_pdfs(pdf_list=pdflist,output_path=dir_result+pdf_name)


def analysis_cutted_subsets(already_saved:bool=True):

    allsets=get_sets(already_saved=already_saved)
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
    plot_some_dist([allsets["bckg_MC_cuttedMass"],allsets["prompt"],allsets["nonprompt"]],to_plot=vars_shared+["trainBckgMC_cuttedMass_class0","trainBckgMC_cuttedMass_class1","trainBckgMC_cuttedMass_class2"]+["trainBckgMC_cuttedMass_withCt_class0","trainBckgMC_cuttedMass_withCt_class1","trainBckgMC_cuttedMass_withCt_class2"],labels=["bckg_MC_cuttedMass","prompt","nonprompt"],file_name="hist_mlsetsMC_cuttedMass_withCt.pdf",severalpages=True)

#for i in (0,1,2):
#    prep.rename_column(file_name="bckgdata_cuttedRadius_high.root", file_dir=directory_sets, old_column_name=f"trainBckgdata_cutR_class{i}", new_column_name=f"trainBckgdata_cutR_high_class{i}")
#    prep.rename_column(file_name="bckgMC_cuttedRadius_high.root", file_dir=directory_sets, old_column_name=f"trainBckgMC_cutR_class{i}", new_column_name=f"trainBckgMC_cutR_high_class{i}")


analysis_cutted_subsets()
