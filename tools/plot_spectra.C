// plot_spectra.C -- luminosity spectra from FEEBL's per-pair native output.
//
// Builds electron-energy, positron-energy and collision-energy (sqrt(s))
// histograms for a baseline and a monochromatized run, overlaid, for as many
// machines as you hand it.
//
//   root -l -b -q 'plot_spectra.C'                       // all three machines
//   root -l -b -q 'plot_spectra.C("ILC250")'             // just one
//   root -l -b -q 'plot_spectra.C("", "/path/to/spectra_raw")'
//
// Extract spectra_raw.tar.gz next to the macro and the default path works.
//
// Outputs, per machine: a PDF, a PNG, and a .root holding the six TH1F objects
// so the plots can be restyled without re-reading the data.
//
// -----------------------------------------------------------------------------
// INPUT FORMAT.  studies/spectra_raw/<tag>/lumi.ee.out.fne, written by FEEBL
// under store_native=1.  Comment lines start with '#'.  The columns are declared
// in the file's own #COLUMNS line:
//
//   e1 e2 x y z t vx1 vy1 vz1 vx2 vy2 vz2 sx1 sy1 sz1 sx2 sy2 sz2 index
//
// Energies in GeV, x/y in nm, z in um, directions as slopes.  Each row is one
// stored colliding pair, sampled so that every row carries equal luminosity
// weight -- so an unweighted histogram of the rows IS the luminosity spectrum.
//
// A WARNING THE FILE CARRIES ITSELF, worth repeating here: the momenta are in
// the HEAD-ON frame.  FEEBL applies the crossing angle as a per-slice position
// shear only, so every direction in the file is short by the half crossing
// angle.  Energy-domain quantities -- which is all this macro uses -- are valid;
// angular ones are not.
//
// -----------------------------------------------------------------------------
// WINDOWING.  Each panel spans -7 sigma to +2.5 sigma about the BASELINE peak,
// where sigma is the Gaussian-equivalent core width (FWHM/2.3548) of the
// baseline's dL/dW.  Both curves therefore share an axis, and the narrowing
// reads as a narrower peak inside a fixed frame rather than as a rescaled axis.
// The asymmetry keeps a slice of the beamstrahlung tail in view without letting
// it dominate.
//
// For the single-beam panels sigma_E = sigma_W/sqrt(2): W = E1+E2 for a head-on
// pair and the two beams are independent, so that is exact rather than a fit.
// -----------------------------------------------------------------------------

#include <TCanvas.h>
#include <TFile.h>
#include <TH1F.h>
#include <TLatex.h>
#include <TLegend.h>
#include <TLine.h>
#include <TStyle.h>

#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

const double kMe = 0.000510999;  // GeV
const int    kNBins = 200;
const double kLoSig = -7.0, kHiSig = 2.5;

struct Machine {
  const char* name;
  double E0, W0;        // nominal beam energy and sqrt(s), GeV
  double coreBase;      // core sigma_W of the baseline, GeV
  double coreMono;
  const char* dirBase;
  const char* dirMono;
};

// Core widths are the values quoted in studies/FINDINGS.md, from the 2 MeV
// dL/dW spectra. Do not re-fit them from a coarse histogram: at FCC-ee's ~7k
// pairs that is noisy, and a second slightly different value for the same
// quantity is how confusion starts.
const Machine kMachines[] = {
  {"ILC250",   125.0, 250.0, 0.3463, 0.1821,
   "ilc250_base", "ilc250_mono"},
  {"ILC_Z",     45.6,  91.2, 0.2036, 0.0991,
   "ilcz_base",   "ilcz_mono"},
  {"FCCee_Z",   45.6,  91.2, 0.1668, 0.1595,
   "fccz_base",   "fccz_mono"},
};

// Exact invariant mass. |p| = sqrt((E-m)(E+m)) rather than sqrt(E^2-m^2): the
// latter loses the difference to cancellation near threshold.
double InvariantMass(double e1, double e2,
                     double vx1, double vy1, double vz1,
                     double vx2, double vy2, double vz2) {
  const double p1 = (e1 > kMe) ? std::sqrt((e1 - kMe) * (e1 + kMe)) : 0.0;
  const double p2 = (e2 > kMe) ? std::sqrt((e2 - kMe) * (e2 + kMe)) : 0.0;
  double n1 = std::sqrt(vx1*vx1 + vy1*vy1 + vz1*vz1); if (n1 == 0) n1 = 1;
  double n2 = std::sqrt(vx2*vx2 + vy2*vy2 + vz2*vz2); if (n2 == 0) n2 = 1;
  const double dot = (vx1*vx2 + vy1*vy2 + vz1*vz2) / (n1 * n2);
  const double w2 = 2*kMe*kMe + 2*(e1*e2 - p1*p2*dot);
  return (w2 > 0) ? std::sqrt(w2) : 0.0;
}

// Fills the three histograms from one run directory. Returns false and says so
// if the file is missing or empty -- a plot drawn from nothing must not look
// like a plot drawn from data.
bool Fill(const std::string& dir, TH1F* hE1, TH1F* hE2, TH1F* hW) {
  const std::string path = dir + "/lumi.ee.out.fne";
  std::ifstream in(path.c_str());
  if (!in) {
    std::cerr << "  ERROR: cannot open " << path << std::endl;
    return false;
  }
  std::string line;
  long n = 0;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream ss(line);
    double e1, e2, x, y, z, t, vx1, vy1, vz1, vx2, vy2, vz2;
    if (!(ss >> e1 >> e2 >> x >> y >> z >> t
             >> vx1 >> vy1 >> vz1 >> vx2 >> vy2 >> vz2)) continue;
    hE1->Fill(e1);
    hE2->Fill(e2);
    hW->Fill(InvariantMass(e1, e2, vx1, vy1, vz1, vx2, vy2, vz2));
    ++n;
  }
  if (n == 0) {
    std::cerr << "  ERROR: " << path << " held no usable rows" << std::endl;
    return false;
  }
  std::cout << "  " << dir << ": " << n << " pairs" << std::endl;
  return true;
}

// Peak position, from a coarse pass. Only the LOCATION is taken from here; the
// width comes from the table above.
double PeakOf(const std::string& dir, int col, double nominal) {
  TH1F h("hpk", "", 500, nominal * 0.96, nominal * 1.01);
  std::ifstream in((dir + "/lumi.ee.out.fne").c_str());
  if (!in) return nominal;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream ss(line);
    double e1, e2, x, y, z, t, vx1, vy1, vz1, vx2, vy2, vz2;
    if (!(ss >> e1 >> e2 >> x >> y >> z >> t
             >> vx1 >> vy1 >> vz1 >> vx2 >> vy2 >> vz2)) continue;
    if (col == 0) h.Fill(e1);
    else if (col == 1) h.Fill(e2);
    else h.Fill(InvariantMass(e1, e2, vx1, vy1, vz1, vx2, vy2, vz2));
  }
  if (h.GetEntries() == 0) return nominal;
  return h.GetBinCenter(h.GetMaximumBin());
}

void StyleOne(TH1F* h, int colour, int width, int fillStyle) {
  h->SetLineColor(colour);
  h->SetLineWidth(width);
  h->SetFillColorAlpha(colour, 0.13);
  h->SetFillStyle(fillStyle);
  h->SetStats(0);
  h->GetXaxis()->SetTitleSize(0.045);
  h->GetYaxis()->SetTitleSize(0.045);
  h->GetXaxis()->SetLabelSize(0.040);
  h->GetYaxis()->SetLabelSize(0.040);
  h->GetYaxis()->SetTitle("pairs / bin");
}

void DrawPanel(TH1F* base, TH1F* mono, double peak, const char* xtitle,
               int colour) {
  StyleOne(base, kGray + 2, 2, 1001);
  StyleOne(mono, colour, 3, 1001);
  base->GetXaxis()->SetTitle(xtitle);

  const double top = 1.15 * std::max(base->GetMaximum(), mono->GetMaximum());
  base->SetMaximum(top);
  base->SetMinimum(0.0);          // linear counts, zero-based
  base->Draw("HIST");
  mono->Draw("HIST SAME");

  TLine* l = new TLine(peak, 0.0, peak, top);
  l->SetLineStyle(2);
  l->SetLineColor(kBlack);
  l->Draw();
}

}  // namespace

void plot_spectra(const char* only = "",
                  const char* dataRoot = "spectra_raw") {
  gStyle->SetOptStat(0);
  gStyle->SetPadTickX(1);
  gStyle->SetPadTickY(1);
  gStyle->SetFrameLineWidth(2);

  const int nMach = sizeof(kMachines) / sizeof(kMachines[0]);
  for (int im = 0; im < nMach; ++im) {
    const Machine& M = kMachines[im];
    if (only && only[0] && std::string(only) != M.name) continue;

    std::cout << "=== " << M.name << " ===" << std::endl;

    const double sW = M.coreBase;
    const double sE = sW / std::sqrt(2.0);

    const std::string dB = std::string(dataRoot) + "/" + M.dirBase;
    const std::string dM = std::string(dataRoot) + "/" + M.dirMono;

    const double pE1 = PeakOf(dB, 0, M.E0);
    const double pE2 = PeakOf(dB, 1, M.E0);
    const double pW  = PeakOf(dB, 2, M.W0);

    const double e1lo = pE1 + kLoSig * sE, e1hi = pE1 + kHiSig * sE;
    const double e2lo = pE2 + kLoSig * sE, e2hi = pE2 + kHiSig * sE;
    const double wlo  = pW  + kLoSig * sW, whi  = pW  + kHiSig * sW;

    printf("  window E- %.3f..%.3f   E+ %.3f..%.3f   sqrt(s) %.3f..%.3f GeV\n",
           e1lo, e1hi, e2lo, e2hi, wlo, whi);

    TH1F* bE1 = new TH1F(Form("%s_base_Em", M.name), ";E_{-} [GeV];", kNBins, e1lo, e1hi);
    TH1F* bE2 = new TH1F(Form("%s_base_Ep", M.name), ";E_{+} [GeV];", kNBins, e2lo, e2hi);
    TH1F* bW  = new TH1F(Form("%s_base_W",  M.name), ";#sqrt{s} [GeV];", kNBins, wlo, whi);
    TH1F* mE1 = new TH1F(Form("%s_mono_Em", M.name), ";E_{-} [GeV];", kNBins, e1lo, e1hi);
    TH1F* mE2 = new TH1F(Form("%s_mono_Ep", M.name), ";E_{+} [GeV];", kNBins, e2lo, e2hi);
    TH1F* mW  = new TH1F(Form("%s_mono_W",  M.name), ";#sqrt{s} [GeV];", kNBins, wlo, whi);

    const bool okB = Fill(dB, bE1, bE2, bW);
    const bool okM = Fill(dM, mE1, mE2, mW);
    if (!okB || !okM) {
      std::cerr << "  SKIPPING " << M.name
                << ": input missing. Nothing is drawn rather than an empty plot."
                << std::endl;
      continue;
    }

    const int colour = (im == 0) ? kOrange + 8 : (im == 1 ? kTeal + 2 : kRed + 1);

    TCanvas* c = new TCanvas(Form("c_%s", M.name),
                             Form("%s luminosity spectra", M.name), 1500, 480);
    c->Divide(3, 1);
    c->cd(1); gPad->SetLeftMargin(0.15); gPad->SetBottomMargin(0.14);
    DrawPanel(bE1, mE1, pE1, "E_{-}  [GeV]", colour);
    c->cd(2); gPad->SetLeftMargin(0.15); gPad->SetBottomMargin(0.14);
    DrawPanel(bE2, mE2, pE2, "E_{+}  [GeV]", colour);
    c->cd(3); gPad->SetLeftMargin(0.15); gPad->SetBottomMargin(0.14);
    DrawPanel(bW, mW, pW, "#sqrt{s}  [GeV]", colour);

    TLegend* leg = new TLegend(0.18, 0.72, 0.62, 0.88);
    leg->SetBorderSize(0);
    leg->SetFillStyle(0);
    leg->AddEntry(bW, "no dispersion", "l");
    leg->AddEntry(mW, "co-optimised monochromatization", "l");
    leg->Draw();

    TLatex tt;
    tt.SetNDC();
    tt.SetTextSize(0.045);
    tt.DrawLatex(0.18, 0.66,
                 Form("core #sigma_{W}: %.4f #rightarrow %.4f GeV",
                      M.coreBase, M.coreMono));

    c->cd(0);
    TLatex title;
    title.SetNDC();
    title.SetTextSize(0.045);
    title.DrawLatex(0.01, 0.965, Form("%s   (window: -7#sigma to +2.5#sigma "
                                      "about the baseline peak)", M.name));

    c->SaveAs(Form("spectra_%s.pdf", M.name));
    c->SaveAs(Form("spectra_%s.png", M.name));

    TFile out(Form("spectra_%s.root", M.name), "RECREATE");
    bE1->Write(); bE2->Write(); bW->Write();
    mE1->Write(); mE2->Write(); mW->Write();
    out.Close();
    std::cout << "  wrote spectra_" << M.name << ".{pdf,png,root}" << std::endl;
  }
}
