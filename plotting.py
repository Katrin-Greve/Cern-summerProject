import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import preparing_data as prep
import ROOT

file_directory_data="/home/katrin/Cern_summerProject/AnalysisResults_treesML_data.root"
tree_data="O2lambdatableml"
file_directory_mc="/home/katrin/Cern_summerProject/AnalysisResults_treesML.root"
tree_mc="O2mclambdatableml"

save_dir = '/home/katrin/Cern_summerProject/imgs/'


def plot_bck_cuts():

    data=ROOT.RDataFrame(tree_data,file_directory_data)
    params, unc, cu, range =prep.fit_gauss(data, var="fMass", fitting_range=[43,58],p0=[1100,1.115,0.005])
    pdf_filename = save_dir+'prep_MLdata_bckg.pdf'
    pdf = PdfPages(pdf_filename)

    bckg=prep.get_bckg(file_directory_data, tree_data, cuts=cu)

    prep.plot_dist(prep.get_rawdata(file_directory_data, tree_data),"fMass",leg_labels="raw data", fs=(8,3),alpha=0.5)
    prep.plot_gauss(data,par=params, cuts=cu, fitting_range=range,fs=(8,3))
    prep.plot_dist(bckg,"fMass",leg_labels="bckg",fs=(8,3),alpha=0.5)

    for fig_num in plt.get_fignums():
        fig = plt.figure(fig_num)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


    pdf.close()

def plot_prompt_cuts():
    data=prep.get_rawMC(file_directory_mc, tree_mc).get_data_frame()
    #data=ROOT.RDataFrame()
    params, unc, cu, range =prep.fit_gauss(data, var="fMass", fitting_range=[43,58],p0=[500,1.115,0.005])
    pdf_filename = save_dir+'prep_MLdata_prompt.pdf'
    pdf = PdfPages(pdf_filename)

    prompt= prep.get_prompt(file_directory_mc, tree_mc,cuts=cu)
    nonprompt=prep.get_nonprompt(file_directory_mc, tree_mc,cuts=cu)

    prep.plot_dist(prep.get_rawMC(file_directory_mc, tree_mc),"fMass",leg_labels="raw MC", fs=(8,3),alpha=0.5)
    prep.plot_gauss(data,par=params, cuts=cu, fitting_range=range,fs=(8,3))
    prep.plot_dist(prompt,"fMass",leg_labels="prompt",fs=(8,3),alpha=0.5)
    prep.plot_dist(nonprompt,"fMass",leg_labels="nonprompt",fs=(8,3),alpha=0.5)

    for fig_num in plt.get_fignums():
        fig = plt.figure(fig_num)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    pdf.close()

#plot_bck_cuts()
plot_prompt_cuts()