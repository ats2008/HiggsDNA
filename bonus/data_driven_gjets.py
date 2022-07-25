import awkward
import argparse
import json
import numpy
import os

from yahist import Hist1D
from yahist.utils import plot_stack
import matplotlib.pyplot as plt

from higgs_dna.utils.logger_utils import setup_logger

def parse_arguments():
    parser = argparse.ArgumentParser(
            description="Assess HiggsDNA parquet output files")

    parser.add_argument(
        "--input_dir",
        required=True,
        type=str,
        help="path to directory with HiggsDNA parquet output files")

    parser.add_argument(
        "--output_dir",
        required=False,
        default="output",
        type=str,
        help="output dir to store tables and plots in")

    parser.add_argument(
        "--pho_id_cut",
        required=True,
        type=float,
        help="sideband cut value for pho id")

    return parser.parse_args()


def fit(array, weights, N):
    """
    Fit polynominal of degree `N` to values in `array` with weight `weights`
    """
    bins = numpy.linspace(-0.9,1,50)
    h = Hist1D(array, weights = weights, bins = bins)
    h = h.normalize()

    x = (bins[1:] + bins[:-1]) / 2.

    coeffs = numpy.polyfit(x, h.counts, N)
    func = numpy.poly1d(coeffs)

    fig, (ax1,ax2) = plt.subplots(2, sharex=True, figsize=(12,9), gridspec_kw=dict(height_ratios=[3, 1]))
    plt.grid()

    h.plot(ax=ax1,color="black", errors=False)
    ax1.plot(x, func(x), color="blue", linestyle="-", label="fit")
    ax1.legend()

    ratio = func(x) / h.counts
    ax2.plot(x, ratio, color = "blue", linestyle="-")
    ax2.set_ylim([0.8, 1.2])
    ax2.set_ylabel("Ratio")

    plt.savefig("fake_id_fit.pdf")
   
    return coeffs, func



def poly_to_pdf(poly, x_min, x_max):
    """
    Convert a polynomial function to a binned pdf in range [x_min, x_max]
    """
    x = numpy.linspace(x_min, x_max, 10000)
    pdf = poly(x) / numpy.sum(poly(x))
    return x, pdf


def sample(poly, x_min, x_max, N):
    x, pdf = poly_to_pdf(poly, x_min, x_max)
    return numpy.random.choice(x, N, p=pdf)


def find_nearest(array,value):
    val = numpy.ones_like(array)*value
    idx = (numpy.abs(array-val)).argmin()
    return array[idx], idx


def integrate(x, pdf, B, A):
    """
    Integrate pdf from [B, A].
    A is assumed to be an array
    """
    x = awkward.Array(x)
    pdf = awkward.Array(pdf)
    idx_min = awkward.argmin(abs(x - (awkward.ones_like(x) * B)))
    idx_min = awkward.ones_like(A,dtype=int) * idx_min

    print(idx_min)

    idx_max = awkward.argmin(
            abs(awkward.unflatten(x, 1) - awkward.Array([A])),
            axis=0
    )

    print(idx_max)

    integral = x[idx_max] - x[idx_min]
    return integral


def main(args):
    # 1. Load inputs
    logger = setup_logger("DEBUG")
    
    events = awkward.from_parquet(args.input_dir + "/merged_nominal.parquet")
    events = events[(events.LeadPhoton_mvaID > -0.9) & (events.SubleadPhoton_mvaID > -0.9)]


    # 1.1 Add min/max photon id vars
    photons = awkward.concatenate(
            [awkward.unflatten(events.LeadPhoton_mvaID, 1), awkward.unflatten(events.SubleadPhoton_mvaID, 1)],
            axis=1
    )

    events["Max_mvaID"] = awkward.ravel(awkward.max(photons, axis=1))
    events["Min_mvaID"] = awkward.ravel(awkward.min(photons, axis=1))

    print(events["Max_mvaID"].type)

    with open(args.input_dir + "/summary.json", "r") as f_in:
        process_map = json.load(f_in)["sample_id_map"]

    os.system("mkdir -p %s" % args.output_dir)

    # 1.2 Check eff on signal
    sig_cut = awkward.zeros_like(events.weight_central, dtype=bool)
    for proc, proc_id in process_map.items():
        if not "ttHH" in  proc:
            continue
        sig_cut = sig_cut | (events.process_id == proc_id)  
    sig_events = events[sig_cut]

    presel_sig_events = sig_events[(sig_events.LeadPhoton_mvaID > args.pho_id_cut) & (sig_events.SubleadPhoton_mvaID > args.pho_id_cut)]
    print("Signal eff. of %.2f with a presel cut of %.1f" % (float(len(presel_sig_events)) / float(len(sig_events)), args.pho_id_cut))

    # 2. Get fake photons from GJets
    gjets_cut = awkward.zeros_like(events.weight_central, dtype=bool) # all false
    for proc, proc_id in process_map.items():
        if not "GJets" in  proc:
            continue
        gjets_cut = gjets_cut | (events.process_id == proc_id)

    gjets_events = events[gjets_cut]

    fake_lead_photons = gjets_events[gjets_events.LeadPhoton_genPartFlav == 0]
    fake_sublead_photons = gjets_events[gjets_events.SubleadPhoton_genPartFlav == 0]

    fake_photons_id = awkward.concatenate([fake_lead_photons.LeadPhoton_mvaID, fake_sublead_photons.SubleadPhoton_mvaID], axis=0)
    fake_photons_weight = awkward.concatenate([fake_lead_photons.weight_central, fake_sublead_photons.weight_central], axis=0)

    coeffs, poly = fit(fake_photons_id, fake_photons_weight, 9)

    x, pdf_full = poly_to_pdf(poly, -0.9, 1.0)
    x, pdf_trim = poly_to_pdf(poly, args.pho_id_cut, 1.0)


    # 3. Get sideband data events
    sideband_cut = (events.process_id == process_map["Data"]) & (events.Max_mvaID > args.pho_id_cut) & (events.Min_mvaID < args.pho_id_cut)
    sdbd_events = events[sideband_cut]
    presel_events_data = events[(events.process_id == process_map["Data"]) & (events.Max_mvaID > args.pho_id_cut) & (events.Min_mvaID > args.pho_id_cut)]

    print("With a presel cut of %.1f, %d events in sideband and %d events in presel." % (args.pho_id_cut, len(sdbd_events), len(presel_events_data)))

    # 3.1 Generate new photon ID value
    sdbd_events["Min_mvaID"] = numpy.random.choice(x, len(sdbd_events), p=pdf_trim)

    # 3.2 Reweight by per-event weight
    omega = integrate(x, pdf_trim, args.pho_id_cut, sdbd_events.Max_mvaID)
    sdbd_events["weight_central"] = sdbd_events.weight_central * omega

    for i in range(25):
        print(sdbd_events.Max_mvaID[i], omega[i])

    # 4. Derive overall normalization scaling
    presel_events = events[(events.Max_mvaID > args.pho_id_cut) & (events.Min_mvaID > args.pho_id_cut)]

    nrbs = ["DiPhoton", "TTGG", "TTGamma", "TTJets", "WGamma", "ZGamma"]
    nrb_cut = awkward.zeros_like(presel_events.weight_central, dtype=bool) # all false
    for proc, proc_id in process_map.items():
        if proc not in nrbs:
            continue
        nrb_cut = nrb_cut | (presel_events.process_id == proc_id)
    nrb_events = presel_events[nrb_cut]

    gjets_target_norm = awkward.sum(presel_events_data.weight_central) - awkward.sum(nrb_events.weight_central)
    gjets_scaling = gjets_target_norm / awkward.sum(sdbd_events.weight_central)
    sdbd_events["weight_central"] = sdbd_events.weight_central * gjets_scaling
    sdbd_events["process_id"] = awkward.ones_like(sdbd_events.process_id) * 99

    presel_events = awkward.concatenate([presel_events, sdbd_events], axis=0)
    presel_events = presel_events[presel_events.Min_mvaID >= -0.7]

    print("Coeffs")
    for x in coeffs:
        print("%.6f," % x)
    print("GJets scaling", gjets_scaling)

    awkward.to_parquet(presel_events, args.output_dir + "/merged_nominal.parquet")

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
