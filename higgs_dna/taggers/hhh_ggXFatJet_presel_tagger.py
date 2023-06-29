import awkward
import vector

vector.register_awkward()

import logging
logger = logging.getLogger(__name__)

from higgs_dna.taggers.tagger import Tagger, NOMINAL_TAG
from higgs_dna.selections import object_selections, jet_selections,  physics_utils
from higgs_dna.utils import awkward_utils, misc_utils
from higgs_dna.selections import gen_selections

DUMMY_VALUE = -9.

DEFAULT_OPTIONS = {

    "gen_info" : { "calculate" : False },
    "jets" : {
        "pt" : 25.0,
        "eta" : 2.4,
        "looseID" : True,
        "dr_photons" : 0.4,
        "bjet_thresh" : {
            "2016UL_postVFP" : 0.3093,
            "2016UL_preVFP": 0.3093,
            "2017" : 0.3033,
            "2018" : 0.2770
        }
    },
    "fatjets" : {
        "pt" : 180.0,
        "eta" : 2.4,
        "looseID" : True,
        "dr_photons" : 0.8,
    },

    "photon_mvaID" : -0.7
}

class HHHgg4bFatJetTagger(Tagger):
    """
    Preselection Tagger for the HH->ggTauTau analysis.
    """
    def __init__(self, name = "hhh_ggXFatJet_presel_tagger", options = {}, is_data = None, year = None):
        super(HHHgg4bFatJetTagger, self).__init__(name, options, is_data, year)

        if not options:
            self.options = DEFAULT_OPTIONS 
        else:
            self.options = misc_utils.update_dict(
                    original = DEFAULT_OPTIONS,
                    new = options
            )
        
        print(self.options)

    def calculate_selection(self, events):
        
        #################################
        ### HH->ggTauTau Preselection ###
        #################################

        ### Presel step 1 : select objects ###
        print(events.Diphoton.Photon)
        print(awkward.count(events.Diphoton.Photon.pt,axis=None))
        #print(awkward.count(events.Jet,axis=1))
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
                n_objects = 8,
                fields = ["btagDeepFlavB"],
                dummy_value = DUMMY_VALUE
        )

        
        fatjet_cut = jet_selections.select_jets(
                jets = events.FatJet,
                options = self.options["fatjets"],
                clean = {
                    "photons" : {
                        "objects" : events.Diphoton.Photon,
                        "min_dr" : self.options["fatjets"]["dr_photons"]
                    }
                },
                name = "SelectedFatJet",
                tagger = self
        )
        fatjets=events.FatJet[fatjet_cut]



        # Add object fields to events array
        for objects, name in zip([fatjets], ["fatjet"]):
            awkward_utils.add_object_fields(
                    events = events,
                    name = name,
                    objects = objects,
                    n_objects = 6,
                    dummy_value = DUMMY_VALUE
            )
        
        for objects, name in zip([jets], ["jet"]):
            awkward_utils.add_object_fields(
                    events = events,
                    name = name,
                    objects = objects,
                    n_objects = 8,
                    dummy_value = DUMMY_VALUE
            )
        
        n_jets = awkward.num(jets)
        awkward_utils.add_field(events, "n_jets", n_jets, overwrite=True)

        #n_bjets = awkward.num(bjets[bjets.btagDeepFlavB > self.options["jets"]["bjet_thresh"][self.year]]) 
        n_bjets = awkward.num(bjets) 
        awkward_utils.add_field(events, "n_bjets", n_bjets, overwrite=True)
        
        n_fatjets = awkward.num(fatjets) 
        awkward_utils.add_field(events, "n_fatjets", n_fatjets, overwrite=True)



        ## 3.5 Select only the highest priority di-tau candidate(s) in each event
        #ditau_pairs = ditau_pairs[ditau_pairs.priority == awkward.min(abs(ditau_pairs.priority), axis = 1)]

        ## 3.6 If still more than one ditau candidate in an event, take the one with m_vis closest to mH = 125 GeV
        #ditau_pairs["ditau"] = ditau_pairs.LeadTauCand + ditau_pairs.SubleadTauCand
        #ditau_pairs[("ditau", "dR")] = ditau_pairs.LeadTauCand.deltaR(ditau_pairs.SubleadTauCand)

        #if awkward.any(awkward.num(ditau_pairs) >= 2):
        #    ditau_pairs = ditau_pairs[awkward.argsort(abs(ditau_pairs.ditau.mass - 125), axis = 1)] # if so, take the one with m_vis closest to mH
        #ditau_pairs = awkward.firsts(ditau_pairs)

        # Add ditau-related fields to array
        # for field in ["pt", "eta", "phi", "mass", "charge", "id"]:
        #     if not field in ["charge", "id"]:
        #         awkward_utils.add_field(
        #                 events,
        #                 "ditau_%s" % field,
        #                 awkward.fill_none(getattr(ditau_pairs.ditau, field), DUMMY_VALUE)
        #         )
        #     awkward_utils.add_field(
        #             events,
        #             "ditau_lead_lepton_%s" % field,
        #             awkward.fill_none(ditau_pairs.LeadTauCand[field], DUMMY_VALUE)
        #     )
        #     awkward_utils.add_field(
        #             events,
        #             "ditau_sublead_lepton_%s" % field,
        #             awkward.fill_none(ditau_pairs.SubleadTauCand[field], DUMMY_VALUE)
        #     )
        # awkward_utils.add_field(
        #         events,
        #         "ditau_dR",
        #         awkward.fill_none(ditau_pairs.ditau.dR, DUMMY_VALUE)
        # )
        # awkward_utils.add_field(
        #         events,
        #         "ditau_dphi",
        #         awkward.fill_none(ditau_pairs.LeadTauCand.deltaphi(ditau_pairs.SubleadTauCand), DUMMY_VALUE)
        # )
        # awkward_utils.add_field(
        #         events,
        #         "ditau_deta",
        #         awkward.fill_none(ditau_pairs.LeadTauCand.deltaeta(ditau_pairs.SubleadTauCand), DUMMY_VALUE)
        # )
 
        # Photon ID cut
        pho_id = (events.LeadPhoton.mvaID > self.options["photon_mvaID"]) & (events.SubleadPhoton.mvaID > self.options["photon_mvaID"])

        #awkward_utils.add_field(events, "dilep_leadpho_mass", awkward.fill_none(dilep_lead_photon.mass, DUMMY_VALUE)) 
        #awkward_utils.add_field(events, "dilep_subleadpho_mass", awkward.fill_none(dilep_sublead_photon.mass, DUMMY_VALUE)) 

        # Veto event if there are at least 2 OSSF leptons and they have m_llg (for either lead or sublead photon) in the z mass window
        #m_llg_veto = ~(((n_muons >= 2) | (n_electrons >= 2)) & (m_llg_veto_lead | m_llg_veto_sublead)) 

        # Add pho/dipho variables
        events[("Diphoton", "pt_mgg")] = events.Diphoton.pt / events.Diphoton.mass
        events[("LeadPhoton", "pt_mgg")] = events.LeadPhoton.pt / events.Diphoton.mass
        events[("SubleadPhoton", "pt_mgg")] = events.SubleadPhoton.pt / events.Diphoton.mass 
        events[("Diphoton", "max_mvaID")] = awkward.where(
                events.LeadPhoton.mvaID > events.SubleadPhoton.mvaID,
                events.LeadPhoton.mvaID,
                events.SubleadPhoton.mvaID
        )
        events[("Diphoton", "min_mvaID")] = awkward.where(
                events.LeadPhoton.mvaID > events.SubleadPhoton.mvaID,
                events.SubleadPhoton.mvaID,
                events.LeadPhoton.mvaID
        )
        events[("Diphoton", "dPhi")] = events.LeadPhoton.deltaphi(events.SubleadPhoton)
        events[("Diphoton", "max_pt_mgg")] = events.LeadPhoton.pt_mgg
        events[("Diphoton", "min_pt_mgg")] = events.SubleadPhoton.pt_mgg
        events[("Diphoton", "helicity")] = physics_utils.abs_cos_theta_parentCM(events.LeadPhoton, events.SubleadPhoton)

        met_p4 = vector.awk({
            "pt" : events.MET_pt,
            "eta" : awkward.zeros_like(events.MET_pt),
            "phi" : events.MET_phi,
            "mass" : awkward.zeros_like(events.MET_pt)
        })

        awkward_utils.add_field(
                events,
                "diphoton_met_dPhi",
                events.Diphoton.deltaphi(met_p4)
        )

        # Fill lead/sublead lepton kinematics as:
        #    if there is a ditau candidate
        #        lead/sublead pt/eta = pt/eta of leading/subleading lepton in ditau candidate
        #    if there is not a ditau candidate
        #        lead pt/eta = pt/eta of leading lepton in event
        #        sublead pt/eta = DUMMY_VALUE
        #for field in ["pt", "eta", "phi", "mass", "charge", "id"]:
        #    awkward_utils.add_field(events, "lead_lepton_%s" % field, awkward.ones_like(events.ditau_pt) * DUMMY_VALUE)
        #    awkward_utils.add_field(events, "sublead_lepton_%s" % field, awkward.ones_like(events.ditau_pt) * DUMMY_VALUE)
        #    events["lead_lepton_%s" % field] = events["tau_candidate_1_%s" % field]
        #    events["lead_lepton_%s" % field] = awkward.where(
        #            events["ditau_pt"] > 0,
        #            events["ditau_lead_lepton_%s" % field],
        #            events["lead_lepton_%s" % field]
        #    )
        #    events["sublead_lepton_%s" % field] = awkward.where(
        #            events["ditau_pt"] > 0,
        #            events["ditau_sublead_lepton_%s" % field],
        #            events["sublead_lepton_%s" % field]
        #    )

        if "weight" not in events.fields:
            events["weight"] = awkward.ones_like(fatjets)

        presel_cut = pho_id 
        self.register_cuts(
            names = [ "photon ID MVA",  "all cuts"],
            results = [ pho_id, presel_cut]
        )
        if not self.is_data and self.options["gen_info"]["calculate"]:
            print("Calculating the GEN vars  : ")
            events = self.calculateHbb_gen_info(events, self.options["gen_info"])

        return presel_cut, events 


    
    def calculateHbb_gen_info(self, events, options=None):
        """
        Calculate gen info, adding the following fields to the events array:
            GenH1[2]bbHiggs : [pt, eta, phi, mass, dR]
            GenH1[2]LeadB : [pt, eta, phi, mass, dR, pt_diff]
            GenH1[2]SubleadB : [pt, eta, phi, mass, dR, pt_diff]

        Perform both matching of
            - closest gen photons from Higgs to reco lead/sublead photons from diphoton candidate
            - closest reco photons to gen photons from Higgs

        If no match is found for a given reco/gen photon, it will be given values of -999. 
        """
        gen_hbb = gen_selections.select_x_to_yz(events.GenPart, 25, 5, -5)
        awkward_utils.add_object_fields(
                events = events,
                name = "GenHbbHiggs",
                objects = gen_hbb.GenParent,
                n_objects = 2
        )

        awkward_utils.add_object_fields(
                events = events,
                name = "GenHbbLeadB",
                objects = gen_hbb.LeadGenChild,
                n_objects = 2
        )

        awkward_utils.add_object_fields(
                events = events,
                name = "GenHbbSubleadB",
                objects = gen_hbb.SubleadGenChild,
                n_objects = 2
        )

        return events 
        


