import ROOT
from ROOT import RooRealVar, RooDataHist, RooArgList, RooAddPdf,RooFit
from typing import Union, Sequence
import os

def fit_simultanously(file_hist_data:str, hist_bdt:str, hist_mass:str, hists_model:Sequence[str], file_model:str, save_name_file:Union[str,None]=None):
    
    ROOT.gROOT.SetBatch()
    ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.ERROR)
    ROOT.RooMsgService.instance().setSilentMode(ROOT.kTRUE)
    ROOT.gErrorIgnoreLevel = ROOT.kError
    # Create a ROOT application
     
    hist_file=ROOT.TFile(file_hist_data)
    histbdt = hist_file.Get(hist_bdt) # Adjust binning and range as necessary
    histmass = hist_file.Get(hist_mass) # Adjust binning and range as necessary

    if histmass is None:
        print("Error: 'hist_mass' not found in the file.")
    else:
        if isinstance(histmass, ROOT.TH1F):
            print("Mass Histogram successfully retrieved and is of type TH1F.")
        else:
            print("The retrieved mass object is not of type TH1F.")

    if isinstance(histbdt, ROOT.TH1F):
        print("bdt Histogram successfully retrieved and is of type TH1F.")
    else:
        print("The retrieved bdt object is not of type TH1F.")
    histbdt.SetName("histbdt")
    histmass.SetName("histmass")

    print(histmass.GetEntries())
    print(histbdt.GetEntries())

    xMass = ROOT.RooRealVar("xMass", "Mass", histmass.GetXaxis().GetXmin(), histmass.GetXaxis().GetXmax())  
    xBdt = ROOT.RooRealVar("xBdt", "Bdt class 2",histbdt.GetXaxis().GetXmin(),histbdt.GetXaxis().GetXmax())

    rdhBdt = ROOT.RooDataHist("rdhBdt", "RooDataHist for BDT", ROOT.RooArgList(xBdt), ROOT.RooFit.Import(histbdt))
    rdhMass = ROOT.RooDataHist("rdhMass", "RooDataHist for Mass", ROOT.RooArgList(xMass), ROOT.RooFit.Import(histmass))

    norm_signal = RooRealVar("norm_signal", "Signal Norm", 20000000,0,100000000000)
    norm_class0 = RooRealVar("norm_class0", "Bckg Norm", 3000,0,1000000000)    
    frac_p=RooRealVar("frac_p", "p-np ratio",0.8, 0, 1)

    print("rdhBbdtentries:", rdhBdt.sumEntries())
    print("rdhMass entries:", rdhMass.sumEntries())

    mean = RooRealVar("mean", "mean of gaussian", 1.1155,1.114,1.116)
    sigma = RooRealVar("sigmaLR", "width of gaussian", 0.0013,0,1)
    alphaL = RooRealVar("alphaL", "alphaL", 1.4178,0,2)
    nL = RooRealVar("nL", "nL", 6.7769,1,10)
    alphaR = RooRealVar("alphaR", "alphaR", 1.43941,10)
    nR = RooRealVar("nR", "nR", 10,1,11)
    p0= RooRealVar("p0", "p0", -0.1311,-10,1)
    p1= RooRealVar("p1", "p1", -0.1267,-10,0)

    crystalball = ROOT.RooCrystalBall("crystalball", "Crystal Ball PDF", xMass, mean, sigma, alphaL, nL,alphaR, nR)
    chebychev = ROOT.RooChebychev("chebychev", "chebychev polynomial", xMass, ROOT.RooArgList(p0,p1))

    model_mass = ROOT.RooAddPdf("model", "Crystal Ball + Chebychev", ROOT.RooArgList(crystalball, chebychev), ROOT.RooArgList(norm_signal, norm_class0))

    model_mass.fitTo(rdhMass)

    mean.setConstant(True)
    sigma.setConstant(True)
    alphaL.setConstant(True)
    nL.setConstant(True)
    alphaR.setConstant(True)
    nR.setConstant(True)


    filemodel = ROOT.TFile.Open(file_model)

    model_hist1= filemodel.Get(hists_model[0])
    model_hist2= filemodel.Get(hists_model[1])
    model_hist3= filemodel.Get(hists_model[2])

    # Convert histogram to RooDataHist
    model_rdh_class0 = RooDataHist("model_rdh1", "model_rdh1", RooArgList(xBdt), model_hist1)
    model_rdh_class1 = RooDataHist("model_rdh2", "model_rdh2", RooArgList(xBdt), model_hist2)
    model_rdh_class2 = RooDataHist("model_rdh3", "model_rdh3", RooArgList(xBdt), model_hist3)

    # Create RooHistPdf from each model RooDataHist
    model_pdf_class0 = ROOT.RooHistPdf("model_pdf_class0", "model_pdf1", ROOT.RooArgSet(xBdt), model_rdh_class0)
    model_pdf_class1 = ROOT.RooHistPdf("model_pdf2", "model_pdf2", ROOT.RooArgSet(xBdt), model_rdh_class1)
    model_pdf_class2 = ROOT.RooHistPdf("model_pdf3", "model_pdf3", ROOT.RooArgSet(xBdt), model_rdh_class2)

    # Define a normalization factor between prompt and non-prompt
    signal_model = RooAddPdf("signal_model", "Prompt+Nonprompt",  RooArgList(model_pdf_class2, model_pdf_class1), RooArgList(frac_p))
    model_dist_fit= RooAddPdf("total_pdf", "sum of extended pdfs", RooArgList(model_pdf_class0, signal_model), RooArgList(norm_class0, norm_signal))

    ## Define combined Data
    sample = ROOT.RooCategory("sample","sample")
    sample.defineType("bdt")
    sample.defineType("mass")
    combData = ROOT.RooDataHist("combData", "Combined data", ROOT.RooArgList(sample, xBdt, xMass))

    # Add entries to combined RooDataHist
    for i in range(1, histmass.GetNbinsX() + 1):
        xMass.setVal(histmass.GetXaxis().GetBinCenter(i))
        sample.setIndex(1)  # Set category index for mass
        combData.add(ROOT.RooArgSet(sample, xMass), histmass.GetBinContent(i))
#
    for j in range(1, histbdt.GetNbinsX() + 1):
        xBdt.setVal(histbdt.GetXaxis().GetBinCenter(j))
        sample.setIndex(0)  # Set category index for bdt
        combData.add(ROOT.RooArgSet(sample, xBdt), histbdt.GetBinContent(j))
    
    # Define simultaneous fit
    simPdf = ROOT.RooSimultaneous("simPdf","simPdf", sample)
    simPdf.addPdf(model_dist_fit, "bdt")
    simPdf.addPdf(model_mass, "mass")
    result_combined=simPdf.fitTo(combData,ROOT.RooFit.Save())
    print(result_combined.status())

    mass_pdf = simPdf.getPdf("mass")
    bdt_pdf = simPdf.getPdf("bdt")

    legend_mass = ROOT.TLegend(0.7, 0.7, 0.95, 0.9)  # (x1, y1, x2, y2) coordinates in normalized canvas units
    legend_mass.SetHeader("Legend")  # Optional legend title
    legend_mass.AddEntry(rdhMass, "Data", "p")  # Add data to legend
    legend_mass.AddEntry(mass_pdf, "Total", "L")  # Add data to legend
    legend_mass.AddEntry(crystalball, "Signal", "L")  # Add data to legend
    legend_mass.AddEntry(chebychev, "Background", "L")  # Add data to legend

    ## Match legend entries with their respective line styles and colors
    legend_mass.GetListOfPrimitives().At(1).SetMarkerStyle(ROOT.kDotted)  # Data marker style
    legend_mass.GetListOfPrimitives().At(2).SetLineColor(ROOT.kBlue)  # Data marker style
    legend_mass.GetListOfPrimitives().At(3).SetLineColor(ROOT.kGreen)  # Data marker style
    legend_mass.GetListOfPrimitives().At(3).SetLineStyle(ROOT.kDashed)  # Data marker style
    legend_mass.GetListOfPrimitives().At(4).SetLineColor(ROOT.kRed)  # Data marker style
    legend_mass.GetListOfPrimitives().At(4).SetLineStyle(ROOT.kDashed)  # Data marker style


    canvas = ROOT.TCanvas("canvas", "Fitted Distribution", 1000, 600)
    canvas.Divide(2,1)


    canvas.cd(1)
    ROOT.gPad.SetLogy(1)
    xframe = xMass.frame(RooFit.Title("invariant Mass"))
    rdhMass.plotOn(xframe)
    mass_pdf.plotOn(xframe)
    mass_pdf.plotOn(xframe, RooFit.Components("chebychev"), RooFit.LineStyle(ROOT.kDashed), RooFit.LineColor(ROOT.kRed))
    mass_pdf.plotOn(xframe, RooFit.Components("crystalball"), RooFit.LineStyle(ROOT.kDashed), RooFit.LineColor(ROOT.kGreen))

    xframe.Draw()
    legend_mass.Draw()

    canvas.cd(2)
    canvas.SetRightMargin(0.33)
    ROOT.gPad.SetLogy(1)
    fit_parameters = result_combined.floatParsFinal()
    notes = ROOT.TPaveText(0.15, 0.8, 0.7, 0.9, "NDC")  # (x1, y1, x2, y2) in normalized canvas units
    notes.AddText("Fit parameter:")
    for i in range(fit_parameters.getSize()):
        param = fit_parameters.at(i)
        if param.GetName()=="norm_signal" or param.GetName()=="norm_class0" or param.GetName()=="frac_p":
            notes.AddText(str(param.GetTitle())+f": {param.getVal():.2e} \pm {param.getError():.2e}")
            #notes.AddText(f"{param.getVal():.2e} \pm {param.getError():.2e}")
    notes.GetLineWith("Fit").SetTextFont(62)
    notes.SetTextAlign(12)  # 12 means left alignment and top vertical alignment
    notes.SetTextSize(0.03)  # Adjust the text size as needed
    notes.SetTextFont(42)
    xframe_bdt = xBdt.frame(RooFit.Title("BDT score class 2"))
    rdhBdt.plotOn(xframe_bdt)
    bdt_pdf.plotOn(xframe_bdt)
    bdt_pdf.plotOn(xframe_bdt, RooFit.Components("model_pdf_class0"), RooFit.LineStyle(ROOT.kDashed), RooFit.LineColor(ROOT.kRed))
    bdt_pdf.plotOn(xframe_bdt, RooFit.Components("signal_model"), RooFit.LineStyle(ROOT.kDashed), RooFit.LineColor(ROOT.kGreen))    
    xframe_bdt.Draw()
    notes.Draw()

    canvas.Update()
    canvas.SaveAs(save_name_file)

    Params=result_combined.floatParsFinal()

    print("Result:")
    for i in range(len(Params)):
        param = Params[i]
        print(f"Parameter: {param.GetName()} Value: {param.getVal()} Error: {param.getError()}")

    canvas.Close()
    if save_name_file:
        if os.path.exists(save_name_file):
            output_file = ROOT.TFile(save_name_file, "UPDATE")
        else:
            output_file = ROOT.TFile(save_name_file, "RECREATE")
        xframe.Write()
        output_file.Close()


def fit_BDThistogram(file_data:str, hist_data:str, file_model:str, hists_model:Sequence[str], save_name_file:Union[str,None]=None, title:str="title", logy:bool=True, save_name_png:str="distribution_fit.png"):
    
    #Open Files
    filedata = ROOT.TFile.Open(file_data)
    filemodel = ROOT.TFile.Open(file_model)

    # Get Histograms
    data_hist = filedata.Get(hist_data)
    model_hist1= filemodel.Get(hists_model[0])
    model_hist2= filemodel.Get(hists_model[1])
    model_hist3= filemodel.Get(hists_model[2])


    # Define the observable
    x = RooRealVar("x", "x", 0,1)

    if not isinstance(data_hist, ROOT.TH1F):
        raise TypeError(f"Expected a ROOT.TH1F histogram for data, but got {type(data_hist)}")
    # Convert histogram to RooDataHist
    data_rdh = RooDataHist("data_hist", "data_hist",  RooArgList(x), data_hist)
    model_rdh_class0 = RooDataHist("model_rdh1", "model_rdh1", RooArgList(x), model_hist1)
    model_rdh_class1 = RooDataHist("model_rdh2", "model_rdh2", RooArgList(x), model_hist2)
    model_rdh_class2 = RooDataHist("model_rdh3", "model_rdh3", RooArgList(x), model_hist3)

    # Create RooHistPdf from each model RooDataHist
    model_pdf_class0 = ROOT.RooHistPdf("model_pdf1", "model_pdf1", ROOT.RooArgSet(x), model_rdh_class0)
    model_pdf_class1 = ROOT.RooHistPdf("model_pdf2", "model_pdf2", ROOT.RooArgSet(x), model_rdh_class1)
    model_pdf_class2 = ROOT.RooHistPdf("model_pdf3", "model_pdf3", ROOT.RooArgSet(x), model_rdh_class2)

    # Define a normalization factor
    norm_class0 = RooRealVar("norm_class0", "normalization factor 1", 1.0, 0.0, 100000000.0)
    norm_class1 = RooRealVar("norm_class1", "normalization factor 2", 1.0, 0.0, 100000000.0)
    norm_class2 = RooRealVar("norm_class2", "normalization factor 3", 1.0, 0.0, 100000000.0)

    # Create extended PDFs by combining each model PDF with its normalization factor
    ext_pdf_class0 = ROOT.RooExtendPdf("ext_pdf_class0", "extended pdf 1", model_pdf_class0, norm_class0)
    ext_pdf_class1 = ROOT.RooExtendPdf("ext_pdf_class1", "extended pdf 2", model_pdf_class1, norm_class1)
    ext_pdf_class2 = ROOT.RooExtendPdf("ext_pdf_class2", "extended pdf 3", model_pdf_class2, norm_class2)

    total_pdf = RooAddPdf("total_pdf", "sum of extended pdfs", RooArgList(ext_pdf_class0, ext_pdf_class1, ext_pdf_class2))

    # Fit the extended model to the data
    result = total_pdf.fitTo(data_rdh, RooFit.Save())
#

    xframe = x.frame(RooFit.Title(title))
    data_rdh.plotOn(xframe)
    total_pdf.plotOn(xframe, RooFit.LineColor(ROOT.kBlue))
    total_pdf.plotOn(xframe, RooFit.Components("ext_pdf_class0"), RooFit.LineStyle(ROOT.kDashed), RooFit.LineColor(ROOT.kRed))
    total_pdf.plotOn(xframe, RooFit.Components("ext_pdf_class1"), RooFit.LineStyle(ROOT.kDashed), RooFit.LineColor(ROOT.kGreen))
    total_pdf.plotOn(xframe, RooFit.Components("ext_pdf_class2"), RooFit.LineStyle(ROOT.kDashed), RooFit.LineColor(ROOT.kOrange))
    legend = ROOT.TLegend(0.7, 0.7, 0.95, 0.9)  # (x1, y1, x2, y2) coordinates in normalized canvas units
    legend.SetHeader("Legend")  # Optional legend title
    legend.AddEntry(data_rdh, "Data", "p")  # Add data to legend
    legend.AddEntry(total_pdf, "Total", "L")  # Add data to legend
    legend.AddEntry(ext_pdf_class0, "Class0", "L")  # Add data to legend
    legend.AddEntry(ext_pdf_class1, "Class1", "L")  # Add data to legend
    legend.AddEntry(ext_pdf_class2, "Class2", "L")  # Add data to legend
    # Match legend entries with their respective line styles and colors
    legend.GetListOfPrimitives().At(2).SetMarkerStyle(ROOT.kDotted)  # Data marker style
    legend.GetListOfPrimitives().At(2).SetLineColor(ROOT.kBlue)  # Data marker style
    legend.GetListOfPrimitives().At(3).SetLineColor(ROOT.kRed)  # Data marker style
    legend.GetListOfPrimitives().At(4).SetLineColor(ROOT.kGreen)  # Data marker style
    legend.GetListOfPrimitives().At(5).SetLineColor(ROOT.kOrange)  # Data marker style
    fit_parameters = result.floatParsFinal()
    notes = ROOT.TPaveText(0.7, 0.05, 0.95, 0.65, "NDC")  # (x1, y1, x2, y2) in normalized canvas units
    notes.AddText("Fit parameter:")
    for i in range(fit_parameters.getSize()):
        param = fit_parameters.at(i)
        notes.AddText(f"{param.GetName()}:")
        notes.AddText(f"{param.getVal():.1e} \pm {param.getError():.1e}")
        notes.AddText("\n")
    notes.GetLineWith("Fit").SetTextFont(62)
    notes.SetTextAlign(12)  # 12 means left alignment and top vertical alignment
    notes.SetTextSize(0.03)  # Adjust the text size as needed
    notes.SetTextFont(42)
    canvas = ROOT.TCanvas("canvas", "Fitted Distribution", 800, 600)
    canvas.SetRightMargin(0.33)
    if logy:
        ROOT.gPad.SetLogy(1)
    xframe.Draw()
    legend.Draw()
    notes.Draw()
    canvas.SaveAs(save_name_png)
    if save_name_file:
        if os.path.exists(save_name_file):
            output_file = ROOT.TFile(save_name_file, "UPDATE")
        else:
            output_file = ROOT.TFile(save_name_file, "RECREATE")
        xframe.Write()
        output_file.Close()


    
def fit_chrystalball(hist_name:str, file_name:str, save_name_png:Union[str,None]=None, var:str="fMass", save_name_file:Union[str,None]=None, x_min_fit:float=1.086,x_max_fit:float=1.14, title:str="crystalball+background fit",cheb:bool=False, logy:bool=True, n_sig:float=4, fixed_cbParams:bool=False):

    # Create a ROOT application
    ROOT.gROOT.SetBatch()
    ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.ERROR)
    ROOT.RooMsgService.instance().setSilentMode(ROOT.kTRUE)
    ROOT.gErrorIgnoreLevel = ROOT.kError

    root_file = ROOT.TFile(file_name)    
    hist =root_file.Get(hist_name)  # Adjust binning and range as necessary

    x = RooRealVar(var, var, x_min_fit, x_max_fit)
    data = RooDataHist("data_hist", "RooDataHist from TH1", RooArgList(x), hist)

    fit_range = ROOT.RooFit.Range(x_min_fit, x_max_fit)
    print("min, max value for fitting: ", x_min_fit, x_max_fit)       

    if not fixed_cbParams:
        mean = RooRealVar("mean", "mean of gaussian", 1.1155, 1.115, 1.116)
        sigma = RooRealVar("sigmaLR", "width of gaussian", 0.001, 0.00001, 0.01)
        alphaL = RooRealVar("alphaL", "alphaL", 1.38, 0.1, 10)
        nL = RooRealVar("nL", "nL", 6, 0.01, 10)
        alphaR = RooRealVar("alphaR", "alphaR", 1.4, 0.1, 10)
        nR = RooRealVar("nR", "nR", 9, 0.01, 10)
        # Define the Crystal Ball function
        crystal_ball = ROOT.RooCrystalBall("crystal_ball", "Crystal Ball PDF", x, mean, sigma, alphaL, nL,alphaR, nR)

    else:
        #mean = RooRealVar("mean", "mean of gaussian", 1.1155, 1.115, 1.116)
        #sigma = RooRealVar("sigmaLR", "width of gaussian", 0.001, 0.00001, 0.01)
        #alphaL_fixed = RooRealVar("alphaL", "alphaL", 1.3197)
        #nL_fixed = RooRealVar("nL", "nL", 6.7180)
        #alphaR_fixed = RooRealVar("alphaR", "alphaR",  1.1630)
        #nR_fixed = RooRealVar("nR", "nR", 9.6692)
        #Define the Crystal Ball function
        #crystal_ball = ROOT.RooCrystalBall("crystal_ball", "Crystal Ball PDF", x, mean, sigma, alphaL, nL,alphaR, nR)
        mean, sigma, alphaL, nL, alphaR, nR = fit_chrystalball_manuel(file_name=file_name, hist_name=hist_name)
        crystalball = ROOT.RooCrystalBall("crystalball", "Crystal Ball PDF", x, mean, sigma, alphaL, nL,alphaR, nR)

    if not cheb:   
        coef1 = RooRealVar("coef1", "coefficient of chrystal ball",1000000,0,100000000)
        coef2 = RooRealVar("coef2", "coefficient of quadratic", 1000,0,1000000000)
        p0 = RooRealVar("p0", "coefficient of constant term", 100,-10000,10000)
        p1 = RooRealVar("p1", "coefficient of linear term", -50,-10000,10000)
        quadratic = ROOT.RooPolynomial("quadratic", "Quadratic function", x, RooArgList(p0,p1))
        #fit_range_help = ROOT.RooFit.Range(1.086, 1.1)
        #quadratic.fitTo(data, fit_range_help)
        fit_range_help2 = ROOT.RooFit.Range(1.13, 1.14)
        quadratic.fitTo(data, fit_range_help2)
        
        
    else:
        p0 = RooRealVar("p0", "coefficient of constant term", -0.115,-10000,10000)
        p1 = RooRealVar("p1", "coefficient of linear term", -0.158,-10000,1) 
        chebychev = ROOT.RooChebychev("chebychev", "chebychev polynomial",x, RooArgList(p0,p1))

        coef1 = RooRealVar("coef1", "coefficient of chrystal ball", 900000,0,100000000)
        coef2 = RooRealVar("coef2", "coefficient of chebychev", 9000,0,1000000000)
        #fit_range_help = ROOT.RooFit.Range(1.086, 1.1)
        #chebychev.fitTo(data, fit_range_help)
        #fit_range_help2 = ROOT.RooFit.Range(1.13, 1.14)
        #chebychev.fitTo(data, fit_range_help2)                

    # Combine the Crystal Ball function and the background function
    if not cheb:
        model = RooAddPdf("model", "Crystal Ball + Quadratic", RooArgList(crystal_ball, quadratic), RooArgList(coef1, coef2))
    else:
        model = RooAddPdf("model", "Crystal Ball + Chebychev", RooArgList(crystal_ball, chebychev), RooArgList(coef1, coef2))


    # Perform the fit
    result=model.fitTo(data, fit_range, RooFit.Save())

    chi2 = model.createChi2(data).getVal()
    mean_val = result.floatParsFinal().find("mean").getVal()
    sigma_val = abs(result.floatParsFinal().find("sigmaLR").getVal())
    mean_val=mean.getVal()
    sigma_val=sigma.getVal()
    print(coef1, coef2)
    x.setRange("integrationRange", mean_val-n_sig*sigma_val, mean_val+n_sig*sigma_val)
    components = model.getComponents()
    cb_component = components.find("crystal_ball")
    if not cheb:
        bckg_component= components.find("quadratic")
    else:
        bckg_component= components.find("chebychev")
    integral_bckg = bckg_component.createIntegral(ROOT.RooArgSet(x), RooFit.Range("integrationRange"),RooFit.NormSet(ROOT.RooArgSet(x)))
    integral_signal = cb_component.createIntegral(ROOT.RooArgSet(x), RooFit.Range("integrationRange"),RooFit.NormSet(ROOT.RooArgSet(x)))
    integral_bckg_val = integral_bckg.getVal()
    integral_signal_val = integral_signal.getVal()
    purity = integral_signal_val / (integral_bckg_val*(coef2.getVal()/coef1.getVal())+integral_signal_val)
    # Plot the results
    xframe = x.frame(RooFit.Title(title))
    data.plotOn(xframe)
    model.plotOn(xframe, RooFit.LineColor(ROOT.kBlue))
    model.plotOn(xframe, RooFit.Components("crystal_ball"), RooFit.LineStyle(ROOT.kDashed), RooFit.LineColor(ROOT.kRed))
    if not cheb:
        model.plotOn(xframe,  RooFit.Components("quadratic"), RooFit.LineStyle(ROOT.kDotted), RooFit.LineColor(ROOT.kGreen))
    else:
        model.plotOn(xframe, RooFit.Components("chebychev"), RooFit.LineStyle(ROOT.kDotted), RooFit.LineColor(ROOT.kGreen))
    line1 = ROOT.TLine(mean_val+n_sig*sigma_val, xframe.GetMinimum(), mean_val+n_sig*sigma_val, xframe.GetMaximum())
    line1.SetLineColor(ROOT.kBlack)
    line1.SetLineWidth(2)
    xframe.addObject(line1)
    line2 = ROOT.TLine(mean_val-n_sig*sigma_val, xframe.GetMinimum(), mean_val-n_sig*sigma_val, xframe.GetMaximum())
    line2.SetLineColor(ROOT.kBlack)
    line2.SetLineWidth(2)
    xframe.addObject(line2)
    legend = ROOT.TLegend(0.7, 0.7, 0.95, 0.9)  # (x1, y1, x2, y2) coordinates in normalized canvas units
    legend.SetHeader("Legend")  # Optional legend title
    legend.AddEntry(data, "Data", "p")  # Add data to legend
    legend.AddEntry(model, "Model", "l")
    legend.AddEntry(crystalball, "Crystal Ball", "l")
    if not cheb:
        legend.AddEntry(quadratic, "Quadratic", "l")
    else:
        legend.AddEntry(chebychev, "Chebychev", "l")
    # Match legend entries with their respective line styles and colors
    legend.GetListOfPrimitives().At(1).SetMarkerStyle(ROOT.kFullCircle)  # Data marker style
    legend.GetListOfPrimitives().At(2).SetLineColor(ROOT.kBlue)  # Model line color
    legend.GetListOfPrimitives().At(3).SetLineColor(ROOT.kRed)    # Crystal Ball line color
    legend.GetListOfPrimitives().At(3).SetLineStyle(ROOT.kDashed)  # Crystal Ball line style
    legend.GetListOfPrimitives().At(4).SetLineColor(ROOT.kGreen)  # Quadratic line color
    legend.GetListOfPrimitives().At(4).SetLineStyle(ROOT.kDotted)  # Quadratic line style
    notes = ROOT.TPaveText(0.7, 0.05, 0.95, 0.65, "NDC")  # (x1, y1, x2, y2) in normalized canvas units
    notes.AddText("Fit parameter:")
    #print("Fit parameters:")
    pdf_list = model.pdfList()
    for pdf_index in range(pdf_list.getSize()):
        notes.AddText("\n")
        pdf = pdf_list.at(pdf_index)
        notes.AddText(f"{pdf.GetName()}:")
        params = pdf.getParameters(data)     
        iterator = params.createIterator()
        param = iterator.Next()
        while param:
            notes.AddText(f"{param.GetName()}: {param.getVal():.4f} \pm {param.getError():.4f}")
            param = iterator.Next()
    notes.AddText("\n")
    notes.AddText(f"Chi2: {chi2:.5f}")
    notes.AddText("\n")
    notes.AddText(f"Chi2/bin: {chi2/data.get_n:.5f}")
    notes.AddText("\n")
    notes.AddText(f"Purity in {n_sig}-sig.:")
    notes.AddText(f"={integral_signal_val:.5f} / {integral_bckg_val*(coef2.getVal()/coef1.getVal())+integral_signal_val:.5f}")
    notes.AddText(f"={purity:.5f}")
    notes.SetTextAlign(12)  # 12 means left alignment and top vertical alignment
    notes.SetTextSize(0.03)  # Adjust the text size as needed
    notes.SetTextFont(42)
    notes.GetLineWith("Fit").SetTextFont(62)
    notes.GetLineWith("crys").SetTextFont(62)
    if not cheb:
        notes.GetLineWith("qua").SetTextFont(62)
    else:
        notes.GetLineWith("cheb").SetTextFont(62)
    notes.GetLineWith("Pur").SetTextFont(62)
    notes.GetLineWith("Chi2:").SetTextFont(62)
    canvas = ROOT.TCanvas("canvas", "Crystal Ball Fit", 800, 600)
    canvas.SetRightMargin(0.33)
    if logy:
        ROOT.gPad.SetLogy(1)
    xframe.Draw()
    legend.Draw()
    notes.Draw()
    canvas.SaveAs(save_name_png)
    if save_name_file:
        if os.path.exists(save_name_file):
            output_file = ROOT.TFile(save_name_file, "UPDATE")
        else:
            output_file = ROOT.TFile(save_name_file, "RECREATE")
        xframe.Write()
        output_file.Close()



def crystalball_plus_quadratic(x, params):
    # Crystal Ball parameters
    alpha_l = params[0]  # Alpha for the left side
    n_l = params[1]      # n for the left side
    alpha_r = params[2]  # Alpha for the right side
    n_r = params[3]      # n for the right side
    mu = params[4]       # Mean
    sigma = params[5]    # Standard deviation
    norm_cb = params[6]

    # Quadratic background parameters
    a = params[7]
    b = params[8]

    # Crystal Ball function
    t = (x[0] - mu) / sigma

    if t < -alpha_l:
        a_l = ROOT.TMath.Power(n_l / abs(alpha_l), n_l) * ROOT.TMath.Exp(-0.5 * alpha_l * alpha_l)
        b_l = n_l / abs(alpha_l) - abs(alpha_l)
        cb= norm_cb * a_l / ROOT.TMath.Power(b_l - t, n_l)
    elif t > alpha_r:
        a_r = ROOT.TMath.Power(n_r / abs(alpha_r), n_r) * ROOT.TMath.Exp(-0.5 * alpha_r * alpha_r)
        b_r = n_r / abs(alpha_r) - abs(alpha_r)
        cb= norm_cb * a_r / ROOT.TMath.Power(b_r + t, n_r)
    else:
        cb=norm_cb * ROOT.TMath.Exp(-0.5 * t * t)
    
    # Quadratic background function
    quad = a * x[0]**2 + b * x[0] 

    return cb + quad



def fit_chrystalball_manuel(hist_name:str, file_name:str,save_name_file:Union[str,None]=None,save_name_pdf:Union[str,None]=None,var:str="fMass",x_min:float=1.086,x_max:float=1.14,folders:bool=None,hist_given:Union[ROOT.TH1F,None]=None, logy:bool=True,save_file:bool=False):


    root_file = ROOT.TFile(file_name)
    hist=root_file.Get(hist_name)
    canvas = ROOT.TCanvas("canvas", "Fit Canvas", 800, 600)
 
    n_params = 9
    combined_function = ROOT.TF1("combined_function", crystalball_plus_quadratic, 1.086, 1.14, n_params)
    combined_function.SetParameters(2, 1.5, 2, 1.2, 1.1155, 0.002, 400000, -1.0, 5.0)
    combined_function.SetParNames("alphaL", "nL","alphaR", "nR", "mean", "sigmaLR", "norm_cb", "a", "b")


    hist.Fit(combined_function, "R")

    combined_function.Draw("SAME")
    #consfunc.Draw("Same")
    fit_results = combined_function.GetParameters()
    fit_errors = combined_function.GetParErrors()

    #    Print the results
    for i in range(n_params):
        print(f"Parameter {i}: {fit_results[i]} ± {fit_errors[i]}")
    if save_name_pdf:
        canvas.SaveAs(save_name_pdf)

    mean = RooRealVar("mean", "mean of gaussian", fit_results[4], fit_results[4]-fit_results[5],fit_results[4]+fit_results[5])
    sigma = RooRealVar("sigmaLR", "width of gaussian", fit_results[5],0,1)
    alphaL = RooRealVar("alphaL", "alphaL", fit_results[0])
    nL = RooRealVar("nL", "nL", fit_results[1])
    alphaR = RooRealVar("alphaR", "alphaR", fit_results[2])
    nR = RooRealVar("nR", "nR", fit_results[3])
    p0 = RooRealVar("p0", "coefficient of linear term", fit_results[8]*10000/fit_results[6])
    p1 = RooRealVar("p1", "coefficient of quadratic term", fit_results[7]*10000/fit_results[6])



    if save_file:
        if os.path.exists(save_name_file):
            output_file = ROOT.TFile(save_name_file, "UPDATE")
        else:
            output_file = ROOT.TFile(save_name_file, "RECREATE")
        hist.Write()
        output_file.Close()
    
    return mean, sigma, alphaL, nL, alphaR, nR