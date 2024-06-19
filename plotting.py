import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import preparing_data as prep
from hipe4ml import plot_utils
import itertools
from hipe4ml.tree_handler import TreeHandler
from typing import Union, Sequence, List
import seaborn as sns

no_set=2
directory_sets=f"/home/katrin/Cern_summerProject/root_trees/set_{no_set}/"
directory_hists=f"/home/katrin/Cern_summerProject/root_histograms/set_{no_set}/"



if no_set==1:
    file_directory_data="/home/katrin/Cern_summerProject/data/AnalysisResults_treesML_data.root"
    tree_data="O2lambdatableml"
    file_directory_mc="/home/katrin/Cern_summerProject/data/AnalysisResults_treesML.root"
    tree_mc="O2mclambdatableml"
    save_dir_plots = '/home/katrin/Cern_summerProject/imgs/first_data/'
    fname=None

if no_set==3:
    file_directory_data="/home/katrin/Cern_summerProject/data/AO2D_data.root"
    tree_data="O2lambdatableml"
    file_directory_mc="/home/katrin/Cern_summerProject/data/AO2D_MC.root"
    tree_mc="O2mclambdatableml"
    fname="DF*"
    save_dir_plots = '/home/katrin/Cern_summerProject/imgs/third_data/'


if no_set==2:
    file_directory_data="/home/katrin/Cern_summerProject/data/AnalysisResults_treesML_data_LHC22o_apass6_small.root"
    tree_data="O2lambdatableml"
    file_directory_mc="/home/katrin/Cern_summerProject/data/AnalysisResults_treesML_mc_LHC24b1b_small.root"
    tree_mc="O2mclambdatableml"
    fname=None
    save_dir_plots = '/home/katrin/Cern_summerProject/imgs/second_data/'


def get_sets(already_saved:bool=True):

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
        prep.save_sets(allsets,set_names=names,dir_tree=directory_sets, tree="tree")
        print("Subsets saved!")


        #names=prep.get_variable_names([prompt, nonprompt, bckg_MC,bckg_data,  bckg_cutted_Ct, bckg_cutted_Mass], locals())

    else:
        prompt=TreeHandler(directory_sets+f"prompt.root", "tree")
        nonprompt=TreeHandler(directory_sets+f"nonprompt.root", "tree")
        bckg_data=TreeHandler(directory_sets+f"bckg_data.root", "tree")
        bckg_MC=TreeHandler(directory_sets+f"bckg_MC.root", "tree")
        bckg_cutted_Ct=TreeHandler(directory_sets+f"bckg_cutted_Ct.root", "tree")
        bckg_cutted_Mass=TreeHandler(directory_sets+f"bckg_cutted_Mass.root", "tree")

        allsets=[prompt, nonprompt, bckg_MC, bckg_data,  bckg_cutted_Ct, bckg_cutted_Mass]
        names=prep.get_variable_names([prompt, nonprompt, bckg_MC, bckg_data,  bckg_cutted_Ct, bckg_cutted_Mass], locals())
    
    return allsets, names


def plot_bck_cuts():

    data=prep.get_rawdata(file_directory_data,tree_data, folder_name=fname)
    pdf_filename = save_dir_plots+'prep_MLdata_bckg.pdf'
    pdf = PdfPages(pdf_filename) 
    prep.plot_hist(data,"fMass",leg_labels="raw data", fs=(8,3),alpha=0.5)

    cu2=prep.fitplot_gauss_rec(data, var="fMass",p0=[300000,1.115,0.005,1000],fs=(8,3),sig_cut=11)[2]
    #cu2=prep.fit_gauss_rec(data, var="fMass",p0=[300000,1.115,0.005,1000],sig_cut=9)[2]
    bckg=prep.get_bckg(file_directory_data, tree_data, cuts=cu2, folder_name=fname)

    prep.plot_hist(bckg,"fMass",leg_labels="bckg",fs=(8,3),alpha=0.5)

    for fig_num in plt.get_fignums():
        fig = plt.figure(fig_num)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
    pdf.close()


def plot_prompt_cuts():
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

    sets,set_names=get_sets(already_saved=already_saved)
    
    vars=prep.var_draw_all(sets)
    pdf_filename = save_dir_plots+'histograms.pdf'
    pdf = PdfPages(pdf_filename)
    prep.plot_hist(sets[:2]+sets[3], vars_to_draw=vars,fs=(15,10),leg_labels=set_names[:2]+set_names[3],alpha=0.3)
    colors=sns.color_palette(n_colors=len(sets))
    for set,lab,col in zip(sets,set_names,colors):
        prep.plot_hist(set, vars_to_draw=vars,fs=(15,10),leg_labels=lab,alpha=0.3,colors=col)

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

    allsets,names=get_sets(already_saved=already_saved)

    vars=prep.var_draw_all(allsets)
    pdf_filename = save_dir_plots+"corr.pdf"
    pdf = PdfPages(pdf_filename)
    plot_utils.plot_corr(allsets,columns=vars,labels=names)
    for fig_num in plt.get_fignums():
        fig = plt.figure(fig_num)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
    pdf.close()

def plot_some_corr(sets:Sequence[TreeHandler], to_plot:Sequence[str],labels:Sequence[str], file_name:str="output_someCorr.pdf"):

    pdf_filename = save_dir_plots+file_name
    pdf = PdfPages(pdf_filename)
    plot_utils.plot_corr(sets,columns=to_plot,labels=labels)
    for fig_num in plt.get_fignums():
        fig = plt.figure(fig_num)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
    pdf.close()


def plot_2dhist_numpy(already_saved:bool=True):

    sets,set_names=get_sets(already_saved=already_saved)

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
    dir_result=f"/home/katrin/Cern_summerProject/root_histograms/set_{no_set}/"
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

    allsets,names=get_sets(already_saved=already_saved)




    
    #vars=["fCt","fL","fRadius","fMass"]
    #vars=prep.var_draw_all(allsets)
    #plot_some_dist(sets=allsets[3:],to_plot=vars,labels=prep.get_variable_names(allsets[3:],locals()),file_name="cuttedBckg_hist_sep.pdf",severalpages=True)
    #plot_some_dist(sets=[bckg,bckg_cutted_Ct],to_plot=vars,labels=["bckg","Ct cutted bckg"],file_name="Ct_cuttedBckg_hist.pdf")

    #plot_some_corr(sets=allsets[3:],to_plot=vars,labels=prep.get_variable_names(allsets[3:],locals()),file_name="cuttedBckg_corr.pdf")
    #plot_some_corr(sets=[bckg,bckg_cutted_Ct],to_plot=vars,labels=["bckg","Ct cutted bckg"],file_name="Ct_cuttedBckg_corr.pdf")
    files=[directory_sets+"prompt.root",directory_sets+"nonprompt.root"]
    plot_2dhist_root(file_names=files,set_names=["prompt","nonprompt"],varsx=["fRadius"],varsy=["fCt","fL"],pdf_name="2dhist_decaylength.pdf")
    #plot_2dhist_root(sets=[bckg_data,bckg_cutted_Mass,bckg_cutted_Ct],set_names=prep.get_variable_names([bckg_data,bckg_cutted_Mass, bckg_cutted_Ct],locals()),varsx=["fRadius"],varsy=["fCt","fL"],pdf_name="2dhist_Radius_Ct_L.pdf")
    #plot_2dhist_root(sets=[bckg_data,bckg_cutted_Mass,bckg_cutted_Ct],set_names=prep.get_variable_names([bckg_data,bckg_cutted_Mass, bckg_cutted_Ct],locals()),varsx=["fRadius"],varsy=["fCt","fL"],pdf_name="2dhist_Radius_Ct_L.pdf")

plot_corr(already_saved=False)
analysis_cutted_subsets(already_saved=True)

