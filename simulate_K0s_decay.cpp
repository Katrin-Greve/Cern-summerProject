#include "TGenPhaseSpace.h"
#include "TLorentzVector.h"
#include "TH1F.h"
#include "TCanvas.h"
#include "TFile.h"
#include <iostream>
#include <vector>

void simulate_K0s_decay() {
    // Define the masses
    double massK0s = 0.497611; // Mass of the K0s in GeV/c^2
    double massP = 0.938272;   // Mass of the proton in GeV/c^2
    double massPi = 0.139570;  // Mass of the pion in GeV/c^2

    TH1F *hist_lambda_mass = new TH1F("hist_lambda_mass", "Invariant Mass; M [GeV/c^2]; Events", 1000,0,0);

    const char* filename ="/home/katrin/Cern_summerProject/root_trees/set_3/bckg_MC.root";
    const char* treeName= "tree";
    const char* branchName ="fPt";


    TFile *file = new TFile(filename, "READ");
    if (!file || file->IsZombie()) {
        std::cerr << "Error: Cannot open file!" << std::endl;
        return ;
    }

    TTree *tree = (TTree*)file->Get(treeName);
    if (!tree) {
        std::cerr << "Error: Cannot find tree '" << treeName << "'!" << std::endl;
        file->Close();
        return ;
    }

    double branchValue;
    tree->SetBranchAddress(branchName, &branchValue);
    Long64_t nEntries = tree->GetEntries();

    for (Long64_t iEntry = 0; iEntry < nEntries; ++iEntry) {
        tree->GetEntry(iEntry);
    

    // Initialize the K0s particle with its mass and zero momentum (at rest)
        TLorentzVector pK0s(0, 0, branchValue, sqrt(massK0s*massK0s+branchValue*branchValue));

    // Initialize the phase space generator
        TGenPhaseSpace event;
        double masses[2] = {massPi, massPi}; // Masses for the pion+ and pion-
        if (!event.SetDecay(pK0s, 2, masses)) {
            std::cerr << "Error: The phase space decay setup failed." << std::endl;
            return;
        }
        int nEvents = 1000;

    // Loop over the events
        for (int i = 0; i < nEvents; i++) {
            // Generate the event
            double weight = event.Generate();

            // Get the daughter particles' momenta
            TLorentzVector *pPip = event.GetDecay(0);  // Pion+
            TLorentzVector *pPim = event.GetDecay(1); // Pion-

            // Treating the decay as if it were from a Lambda particle
            // Assign mass hypotheses: proton for pion+ and pion- remains pion-
            TLorentzVector pProton = *pPip;
            pProton.SetE(sqrt(pProton.P()*pProton.P() + massP*massP)); // Recalculate energy with proton mass

            // Compute the invariant mass of the Lambda candidate
            TLorentzVector lambdaCandidate = pProton + *pPim;
            double massLambda = lambdaCandidate.M();

            // Fill the histogram
            hist_lambda_mass->Fill(massLambda, weight);
    
        }
    }

    int maxBin = hist_lambda_mass->GetMaximumBin();
    double binCenter = hist_lambda_mass->GetXaxis()->GetBinCenter(maxBin);
    double binLowerEdge= hist_lambda_mass->GetXaxis()->GetBinLowEdge(maxBin);
    //double *minx =hist_lambda_mass->GetXaxis();

    //double binHigherEdge = hist_lambda_mass->GetXaxis()->GetBinHighEdge(maxBin);
    double width =hist_lambda_mass->GetBinWidth(maxBin);

    // Create a canvas to draw the histogram
    TCanvas *canvas = new TCanvas("canvas", "Invariant Mass Distribution of the Lambda Candidate", 800, 600);

    // Draw the histogram
    hist_lambda_mass->Draw("HIST");
    

    // Create a text entry for the legend with the maximum value
    TText *textmax = new TText(0, 0, Form("Max Value: %.7f", binCenter));

    TLatex text;
    text.SetTextAlign(12); // Text alignment (12 for left aligned)
    text.SetTextSize(0.03);
    text.SetTextColor(kBlack);
    text.DrawLatexNDC(0.15, 0.85, Form("#splitline{Max Value: %.4f #pm %.4f}{ Width: %.4f}", binCenter,width/2., width)); // Draw at normali
    
    // Save the canvas as an image file
    canvas->SaveAs("lambda_candidate_mass.png");

    // Optionally, save the histogram to a ROOT file
    TFile *outputFile = new TFile("lambda_candidate_mass.root", "RECREATE");
    hist_lambda_mass->Write();
}

int main() {
    simulate_K0s_decay();
    return 0;
}
