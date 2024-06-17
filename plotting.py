import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import preparing_data as prep
from hipe4ml import plot_utils
import itertools
import uproot

no_set=1

if no_set==1:
    file_directory_data="/home/katrin/Cern_summerProject/data/AnalysisResults_treesML_data.root"
    tree_data="O2lambdatableml"
    file_directory_mc="/home/katrin/Cern_summerProject/data/AnalysisResults_treesML.root"
    tree_mc="O2mclambdatableml"
    save_dir = '/home/katrin/Cern_summerProject/imgs/first_data/'
    fname=None

if no_set==3:
    file_directory_data="/home/katrin/Cern_summerProject/data/AO2D_data.root"
    tree_data="O2lambdatableml"
    file_directory_mc="/home/katrin/Cern_summerProject/data/AO2D_MC.root"
    tree_mc="O2mclambdatableml"
    fname="DF*"
    save_dir = '/home/katrin/Cern_summerProject/imgs/third_data/'


if no_set==2:
    file_directory_data="/home/katrin/Cern_summerProject/data/AnalysisResults_treesML_data_LHC22o_apass6_small.root"
    tree_data="O2lambdatableml"
    file_directory_mc="/home/katrin/Cern_summerProject/data/AnalysisResults_treesML_mc_LHC24b1b_small.root"
    tree_mc="O2mclambdatableml"
    fname=None
    save_dir = '/home/katrin/Cern_summerProject/imgs/second_data/'



def plot_bck_cuts():

    data=prep.get_rawdata(file_directory_data,tree_data, folder_name=fname)
    pdf_filename = save_dir+'prep_MLdata_bckg.pdf'
    pdf = PdfPages(pdf_filename)
    
    prep.plot_hist(data,"fMass",leg_labels="raw data", fs=(8,3),alpha=0.5)
    
    cu2=prep.fitplot_gauss_rec(data, var="fMass",p0=[300000,1.115,0.005,1000],fs=(8,3),sig_cut=11)[2]
    #cu2=prep.fit_gauss_rec(data, var="fMass",p0=[300000,1.115,0.005,1000],sig_cut=9)[2]
    bckg=prep.get_bckg(file_directory_data, tree_data, cuts=cu2,folder_name=fname)

    prep.plot_hist(bckg,"fMass",leg_labels="bckg",fs=(8,3),alpha=0.5)

    for fig_num in plt.get_fignums():
        fig = plt.figure(fig_num)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


    pdf.close()

def plot_prompt_cuts():
    data=prep.get_rawMC_data(file_directory_mc, tree_mc ,folder_name=fname)
    pdf_filename =save_dir+'prep_MLdata_prompt.pdf'
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



def plot_allvar_dist():
        
    data=prep.get_rawdata(file_directory_data, tree_data,folder_name=fname)
    cu=prep.fit_gauss_rec(data, var="fMass",p0=[300000,1.115,0.005,1000])[2]

    nonprompt=prep.get_nonprompt(file_directory_mc, tree_mc,folder_name=fname)
    prompt=prep.get_prompt(file_directory_mc, tree_mc,folder_name=fname)
    bckg=prep.get_bckg(file_directory_data, tree_data, cu,folder_name=fname)
    bckg_MC=prep.get_MC_bckg(file_directory_mc, tree_mc,folder_name=fname)

    prompt_add=prep.proton_pion_division(prompt)
    nonprompt_add=prep.proton_pion_division(nonprompt)
    bckg_add=prep.proton_pion_division(bckg)
    bckg_MC_add=prep.proton_pion_division(bckg_MC)
    
    
    #vars=["fDcaPVProton","fDcaPVPion","fTpcNsigmaProton","fTpcNsigmaPion"]
    vars=prep.var_draw_all(prompt_add,bckg_add)
    #vars=["fPt"]
    pdf_filename = save_dir+'histograms.pdf'
    pdf = PdfPages(pdf_filename)
    prep.plot_hist([prompt_add,nonprompt_add], vars_to_draw=vars,fs=(15,10),leg_labels=["prompt","nonprompt"],alpha=0.3)
    prep.plot_hist([prompt_add, nonprompt_add,bckg_add] ,vars_to_draw=vars,fs=(15,10),leg_labels=["prompt", "nonprompt","bckg"],alpha=0.3)
    prep.plot_hist([prompt_add, nonprompt_add,bckg_MC_add] ,vars_to_draw=vars,fs=(15,10),leg_labels=["prompt", "nonprompt","bckg MC"],alpha=0.3)
    prep.plot_hist([bckg_add,bckg_MC_add], vars_to_draw=vars,fs=(15,10),leg_labels=["bckg data","bckg MC"],alpha=0.3)

    for fig_num in plt.get_fignums():
        fig = plt.figure(fig_num)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
    pdf.close()

def plot_corr():

    data=prep.get_rawdata(file_directory_data, tree_data,folder_name=fname)
    cu=prep.fit_gauss_rec(data, var="fMass",p0=[300000,1.115,0.005,1000])[2]

    nonprompt=prep.get_nonprompt(file_directory_mc, tree_mc,folder_name=fname)
    prompt=prep.get_prompt(file_directory_mc, tree_mc,folder_name=fname)
    bckg=prep.get_bckg(file_directory_data, tree_data, cu,folder_name=fname)
    bckg_MC=prep.get_MC_bckg(file_directory_mc, tree_mc,folder_name=fname)

    prompt_add=prep.proton_pion_division(prompt)
    nonprompt_add=prep.proton_pion_division(nonprompt)
    bckg_add=prep.proton_pion_division(bckg)
    bckg_MC_add=prep.proton_pion_division(bckg_MC)
    
    
    #vars=["fDcaPVProton","fDcaPVPion","fTpcNsigmaProton","fTpcNsigmaPion"]
    vars=prep.var_draw_all(prompt_add,bckg_add)


    pdf_filename = save_dir+"corr.pdf"
    pdf = PdfPages(pdf_filename)
    plot_utils.plot_corr([prompt_add,nonprompt_add,bckg_add,bckg_MC_add],columns=vars,labels=["prompt","nonprompt","bckg data","bckg MC"])
    for fig_num in plt.get_fignums():
        fig = plt.figure(fig_num)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
    pdf.close()

def plot_2dhist():
    data=prep.get_rawdata(file_directory_data, tree_data,folder_name=fname)
    cu=prep.fit_gauss_rec(data, var="fMass",p0=[300000,1.115,0.005,1000])[2]

    nonprompt=prep.get_nonprompt(file_directory_mc, tree_mc,folder_name=fname)
    prompt=prep.get_prompt(file_directory_mc, tree_mc,folder_name=fname)
    bckg=prep.get_bckg(file_directory_data, tree_data, cu,folder_name=fname)
    bckg_MC=prep.get_MC_bckg(file_directory_mc, tree_mc,folder_name=fname)

    prompt_add=prep.proton_pion_division(prompt)
    nonprompt_add=prep.proton_pion_division(nonprompt)
    bckg_add=prep.proton_pion_division(bckg)
    bckg_MC_add=prep.proton_pion_division(bckg_MC)

    sets=[prompt_add,nonprompt_add,bckg_add,bckg_MC_add]
    set_names=["prompt","nonprompt","bckg_data","bckg_MC"]

    l1=["fDcaV0PV", "fCosPA"]
    l2=["fCt","fPt"]

    pdf_filename = save_dir+'histograms2d.pdf'
    pdf = PdfPages(pdf_filename)

    for (i,name) in zip(sets,set_names):
        fig,ax=plt.subplots(len(l1),len(l2),figsize=(10,10))
        for ax_i, tu in zip(ax.T.flatten(),list(itertools.product(l1, l2))):
            prep.plot_2dhist(i,tu[0],tu[1],ax=ax_i)
            ax_i.set(xlabel=tu[0],ylabel=tu[1])
            if tu[0]=="fCosPA":
                ax_i.set_xlim(0.995,1)
        ax[0][0].set_title(name+f", size of sample: {prep.get_sampleSize(i)}", fontsize=20)

    for fig_num in plt.get_fignums():
        fig = plt.figure(fig_num)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
    pdf.close()


plot_bck_cuts()
plot_prompt_cuts()
plot_allvar_dist()
plot_2dhist()
plot_corr()