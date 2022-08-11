import awkward
import vector
import xgboost
import numpy

vector.register_awkward()

import logging
logger = logging.getLogger(__name__)

from higgs_dna.taggers.tagger import Tagger, NOMINAL_TAG
from higgs_dna.utils import awkward_utils, misc_utils

DEFAULT_OPTIONS = {
    "bdt_file" : "data/xyh_ggbb_bdt_1Aug2022.xgb", # if running on condor, this file needs to be placed somewhere under higgs_dna/ so that it is included in the tar file. We probably want to think of a better long term solution for this.
    "bdt_features" : [
        "dijet_lead_pt", "dijet_lead_eta", "dijet_lead_btagDeepFlavB",
        "dijet_sublead_pt", "dijet_sublead_eta", "dijet_sublead_btagDeepFlavB",
        "dijet_pt", "dijet_eta", "dijet_mass", "dijet_dR",
        "xcand_pt", "xcand_eta", "xcand_mass",
        ("Diphoton", "pt_mgg"), ("Diphoton", "eta"),
        ("LeadPhoton", "pt_mgg"), ("LeadPhoton", "eta"), ("LeadPhoton", "mvaID"),
        ("SubleadPhoton", "pt_mgg"), ("SubleadPhoton", "eta"), ("SubleadPhoton", "mvaID")
    ],
    "bdt_cuts" : [0.9898, 0.882222, 0.0]
}

class XYH_ggbb_SRTagger(Tagger):
    """
    Signal region tagger for the non-resonant HH->ggTauTau analysis.
    """
    def __init__(self, name, options = {}, is_data = None, year = None):
        super(XYH_ggbb_SRTagger, self).__init__(name, options, is_data, year)

        if not options:
            self.options = DEFAULT_OPTIONS
        else:
            self.options = misc_utils.update_dict(
                    original = DEFAULT_OPTIONS,
                    new = options
            )


    def calculate_selection(self, events):
        #####################################
        ### HH->ggTauTau Non-resonant SRs ###
        #####################################

        # Initialize BDT 
        bdt = xgboost.Booster()
        bdt.load_model(misc_utils.expand_path(self.options["bdt_file"]))

        # Convert events to proper format for xgb
        events_bdt = awkward.values_astype(events, numpy.float64)

        bdt_features = []
        for x in self.options["bdt_features"]:
            if isinstance(x, tuple):
                name_flat = "_".join(x)
                events_bdt[name_flat] = events_bdt[x]
                bdt_features.append(name_flat)
            else:
                bdt_features.append(x)
 
        features_bdt = awkward.to_numpy(
                events_bdt[bdt_features]
        )
        features_bdt = xgboost.DMatrix(
                features_bdt.view((float, len(features_bdt.dtype.names)))
        )

        # Calculate BDT score
        events["bdt_score"] = bdt.predict(features_bdt)
        

        # Calculate SR cuts
        n_signal_regions = len(self.options["bdt_cuts"])
        sr_cuts = []
        for i in range(n_signal_regions):
            cut_sr = events.bdt_score >= self.options["bdt_cuts"][i]
            for j in range(len(sr_cuts)):
                cut_sr = cut_sr & ~(sr_cuts[j])

            # record which SR each event enters
            events["pass_sr_%d" % i] = cut_sr
            sr_cuts.append(cut_sr)


        # Calculate OR of all BDT cuts
        presel_cut = events.run < 0 # dummy all False
        for cut in sr_cuts:
            presel_cut = presel_cut | cut

        return presel_cut, events
