import awkward
import vector

vector.register_awkward()

import logging
logger = logging.getLogger(__name__)

from higgs_dna.taggers.tagger import Tagger, NOMINAL_TAG
from higgs_dna.selections import object_selections, lepton_selections, jet_selections, tau_selections, physics_utils, gen_selections
from higgs_dna.utils import awkward_utils, misc_utils

DUMMY_VALUE = -9.
GEN_WEIGHT_BAD_VAL = -99999.
DEFAULT_OPTIONS = {
    "electrons" : {
        "pt" : 10.0,
        "eta" : 2.5,
        "dxy" : 0.045,
        "dz" : 0.2,
        "id" : "WP90",
        "dr_photons" : 0.2,
        "veto_transition" : True,
    },
    "muons" : {
        "pt" : 15.0,
        "eta" : 2.4,
        "dxy" : 0.045,
        "dz" : 0.2,
        "id" : "medium",
        "pfRelIso03_all" : 0.3,
        "dr_photons" : 0.2
    },
    "jets" : {
        "pt" : 25.0,
        "eta" : 2.4,
        "looseID" : True,
        "dr_photons" : 0.4,
        "dr_electrons" : 0.4,
        "dr_muons" : 0.4,
        "dr_taus" : 0.4,
        "dr_iso_tracks" : 0.4,
        "bjet_thresh" : {
            "2016UL_postVFP" : 0.3093,
            "2016UL_preVFP": 0.3093,
            "2017" : 0.3033,
            "2018" : 0.2770
        }
    },
    "photon_mvaID" : -0.7
}

class XYHggbbTagger(Tagger):
    """
    Preselection Tagger for X->YH with Y->gg, H->bb 
    """
    def __init__(self, name = "xyh_ggbb_tagger", options = {}, is_data = None, year = None):
        super(XYHggbbTagger, self).__init__(name, options, is_data, year)

        if not options:
            self.options = DEFAULT_OPTIONS 
        else:
            self.options = misc_utils.update_dict(
                    original = DEFAULT_OPTIONS,
                    new = options
            )


    def calculate_selection(self, events):
        #################################
        ### HH->ggTauTau Preselection ###
        #################################

        ### Presel step 1 : select objects ###
        
        # Electrons
        electron_cut = lepton_selections.select_electrons(
                electrons = events.Electron,
                options = self.options["electrons"],
                clean = {
                    "photons" : {
                        "objects" : events.Diphoton.Photon,
                        "min_dr" : self.options["electrons"]["dr_photons"]
                    }
                },
                name = "SelectedElectron",
                tagger = self
        )

        electrons = awkward_utils.add_field(
                events = events,
                name = "SelectedElectron",
                data = events.Electron[electron_cut]
        )

        # Muons
        muon_cut = lepton_selections.select_muons(
                muons = events.Muon,
                options = self.options["muons"],
                clean = {
                    "photons" : {
                        "objects" : events.Diphoton.Photon,
                        "min_dr" : self.options["muons"]["dr_photons"]
                    }
                },
                name = "SelectedMuon",
                tagger = self
        )

        muons = awkward_utils.add_field(
                events = events,
                name = "SelectedMuon",
                data = events.Muon[muon_cut]
        )

        # Jets
        jet_cut = jet_selections.select_jets(
                jets = events.Jet,
                options = self.options["jets"],
                clean = {
                    "photons" : {
                        "objects" : events.Diphoton.Photon,
                        "min_dr" : self.options["jets"]["dr_photons"]
                    }
                },
                name = "SelectedJet",
                tagger = self
        )

        jets = awkward_utils.add_field(
                events = events,
                name = "SelectedJet",
                data = events.Jet[jet_cut]
        )

        bjets = jets[awkward.argsort(jets.btagDeepFlavB, axis = 1, ascending = False)]
        awkward_utils.add_object_fields(
                events = events,
                name = "b_jet",
                objects = bjets,
                n_objects = 2,
                fields = ["btagDeepFlavB"],
                dummy_value = DUMMY_VALUE
        )

        pho_id = (events.LeadPhoton.mvaID > self.options["photon_mvaID"]) & (events.SubleadPhoton.mvaID > self.options["photon_mvaID"])

        lepton_veto = awkward.num(electrons) + awkward.num(muons) == 0
        n_jet_cut = awkward.num(jets) >= 2

        medium_bjets = bjets[bjets.btagDeepFlavB > self.options["jets"]["bjet_thresh"][self.year]]
        b_jet_cut = awkward.num(medium_bjets) >= 0 # no cut for now

        # Gen info
        if not self.is_data:
            gen_xyh = gen_selections.select_x_to_yz(events.GenPart, 45, 25, 35)
            gen_ygg = gen_selections.select_x_to_yz(events.GenPart, 35, 22, 22)
            gen_hbb = gen_selections.select_x_to_yz(events.GenPart, 25, 5, 5)

            awkward_utils.add_object_fields(events, "GenX", gen_xyh.GenParent, n_objects=1)
            awkward_utils.add_object_fields(events, "GenY", gen_ygg.GenParent, n_objects=1)
            awkward_utils.add_object_fields(events, "GenHiggs", gen_hbb.GenParent, n_objects=1)

            awkward_utils.add_object_fields(events, "GenBFromHiggs", awkward.concatenate([gen_hbb.LeadGenChild,gen_hbb.SubleadGenChild], axis=1), n_objects=2)
            awkward_utils.add_object_fields(events, "GenGFromHiggs", awkward.concatenate([gen_ygg.LeadGenChild,gen_ygg.SubleadGenChild], axis=1), n_objects=2)

            jets["gen_match"] = object_selections.delta_R(
                jets,
                awkward.concatenate([gen_hbb.LeadGenChild,gen_hbb.SubleadGenChild], axis=1),
                dr = 0.4,
                mode = "max"
            )

            events["n_gen_matched_jets"] = awkward.num(jets[jets.gen_match == True])

        # bb candidates
        jets = awkward.Array(jets, with_name = "Momentum4D")
        events["n_jets"] = awkward.num(jets)
        
        dijet_pairs = awkward.combinations(jets, 2, fields = ["LeadJet", "SubleadJet"])

        if awkward.any(awkward.num(dijet_pairs) >= 2):
            dijet_pairs = dijet_pairs[awkward.argsort(dijet_pairs.LeadJet.btagDeepFlavB + dijet_pairs.SubleadJet.btagDeepFlavB, axis=1, ascending=False)]

        dijet_pairs["dijet"] = dijet_pairs.LeadJet + dijet_pairs.SubleadJet
        dijet_pairs[("dijet", "dR")] = dijet_pairs.LeadJet.deltaR(dijet_pairs.SubleadJet)
        dijet_pairs = awkward.firsts(dijet_pairs)

        for field in ["pt", "eta", "phi", "btagDeepFlavB", "mass", "gen_match"]:
            if self.is_data and field == "gen_match":
                continue
            awkward_utils.add_field(
                    events,
                    "dijet_lead_%s" % field,
                    awkward.fill_none(dijet_pairs.LeadJet[field], DUMMY_VALUE)
            )
            awkward_utils.add_field(
                    events,
                    "dijet_sublead_%s" % field,
                    awkward.fill_none(dijet_pairs.SubleadJet[field], DUMMY_VALUE)
            )

        if not self.is_data:
            events["n_gen_matched_in_dijet"] = awkward.where(dijet_pairs.LeadJet.gen_match == True, awkward.ones_like(events.run), awkward.zeros_like(events.run)) + awkward.where(dijet_pairs.SubleadJet.gen_match == True, awkward.ones_like(events.run), awkward.zeros_like(events.run)) 

        awkward_utils.add_field(events, "dijet_pt", awkward.fill_none(dijet_pairs.dijet.pt, DUMMY_VALUE))
        awkward_utils.add_field(events, "dijet_eta", awkward.fill_none(dijet_pairs.dijet.eta, DUMMY_VALUE))
        awkward_utils.add_field(events, "dijet_phi", awkward.fill_none(dijet_pairs.dijet.phi, DUMMY_VALUE))
        awkward_utils.add_field(events, "dijet_mass", awkward.fill_none(dijet_pairs.dijet.mass, DUMMY_VALUE))
        awkward_utils.add_field(events, "dijet_dR", awkward.fill_none(dijet_pairs.dijet.dR, DUMMY_VALUE))

        # X->YH candidates
        x_cands = events.Diphoton + dijet_pairs.dijet

        awkward_utils.add_field(events, "xcand_pt", awkward.fill_none(x_cands.pt, DUMMY_VALUE))
        awkward_utils.add_field(events, "xcand_eta", awkward.fill_none(x_cands.eta, DUMMY_VALUE))
        awkward_utils.add_field(events, "xcand_phi", awkward.fill_none(x_cands.phi, DUMMY_VALUE))
        awkward_utils.add_field(events, "xcand_mass", awkward.fill_none(x_cands.mass, DUMMY_VALUE))

        dijet_mass_cut = events.dijet_mass >= 50. 

        # Add some dipho/pho variables
        events[("Diphoton", "pt_mgg")] = events.Diphoton.pt / events.Diphoton.mass
        events[("LeadPhoton", "pt_mgg")] = events.LeadPhoton.pt / events.Diphoton.mass
        events[("SubleadPhoton", "pt_mgg")] = events.SubleadPhoton.pt / events.Diphoton.mass

        presel_cut = pho_id & lepton_veto & n_jet_cut & b_jet_cut & dijet_mass_cut
        self.register_cuts(
            names = ["photon id", "lepton veto", "n_jets >= 2", "1 medium b", "m_jj > 50 GeV", "all cuts"],
            results = [pho_id, lepton_veto, n_jet_cut, b_jet_cut, dijet_mass_cut, presel_cut]
        )

        return presel_cut, events 
