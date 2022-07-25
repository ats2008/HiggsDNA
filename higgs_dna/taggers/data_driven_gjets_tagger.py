import awkward
import vector
import numpy

vector.register_awkward()

import logging
logger = logging.getLogger(__name__)

from higgs_dna.taggers.tagger import Tagger
from higgs_dna.constants import CENTRAL_WEIGHT, NOMINAL_TAG, DATA_DRIVEN_GJETS_PROCESS_ID
from higgs_dna.utils import awkward_utils, misc_utils

DEFAULT_OPTIONS = {
    "sideband_cut" : None,
    "pdf_coeffs" : None,
    "norm_factor" : 1
}

def poly_to_pdf(poly, x_min, x_max):
    """
    Convert a polynomial function to a binned pdf in range [x_min, x_max]
    """
    x = numpy.linspace(x_min, x_max, 10000)
    pdf = poly(x) / numpy.sum(poly(x))
    return x, pdf


def sample(poly, x_min, x_max, N):
    """
    Sample N times from a pdf with shape given by `poly` in the range [`x_min`, `x_max`].
    """
    x, pdf = poly_to_pdf(poly, x_min, x_max)
    return numpy.random.choice(x, N, p=pdf)


def integrate(x, pdf, B, A):
    """
    Integrate pdf from [B, A].
    A is assumed to be an array
    """
    x = awkward.Array(x)
    pdf = awkward.Array(pdf)
    idx_min = awkward.argmin(abs(x - (awkward.ones_like(x) * B)))
    idx_min = awkward.ones_like(A,dtype=int) * idx_min

    idx_max = awkward.argmin(
            abs(awkward.unflatten(x, 1) - awkward.Array([A])),
            axis=0
    )

    integral = x[idx_max] - x[idx_min]
    return integral


class DataDrivenGJetsTagger(Tagger):
    """
    Tagger to modify data events according to the procedure for creating a data-driven QCD/GJets sample from a low photon ID sideband.
    """
    def __init__(self, name = "dd_gjets_tagger", options = {}, is_data = None, year = None):
        super(DataDrivenGJetsTagger, self).__init__(name, options, is_data, year)

        if not options:
            self.options = DEFAULT_OPTIONS
        else:
            self.options = misc_utils.update_dict(
                    original = DEFAULT_OPTIONS,
                    new = options
            )

    def calculate_selection(self, events):
        for opt, val in self.options.items():
            if val is None:
                message = "[DataDrivenGJetsTagger : calculate_selection] Option '%s' must be explicitly specified." % opt
                logger.exception(message)
                raise ValueError(message)

        # If this is not data, return dummy cut of all True
        if not self.is_data:
            presel_cut = awkward.ones_like(events.event, dtype=bool)
            return presel_cut, events

        
        events["sideband_event"] = awkward.where(
            (events.Diphoton.max_mvaID > self.options["sideband_cut"]) & (events.Diphoton.min_mvaID < self.options["sideband_cut"]),
            awkward.ones_like(events.event, dtype=bool),
            awkward.zeros_like(events.event, dtype=bool)
        )

        # Initialize pdf
        poly = numpy.poly1d(self.options["pdf_coeffs"])
        x, pdf = poly_to_pdf(poly, self.options["sideband_cut"], 1.0)

        # Generate new min mvaID values
        events[("Diphoton", "min_mvaID")] = awkward.where(
            events.sideband_event,
            awkward.Array(numpy.random.choice(x, len(events), p=pdf)),
            events.Diphoton.min_mvaID
        )

        # Update the relevant lead/sublead value accordingly
        events[("LeadPhoton", "mvaID")] = awkward.where(
            (events.LeadPhoton.mvaID < self.options["sideband_cut"]) & (events.SubleadPhoton.mvaID > self.options["sideband_cut"]),
            events.Diphoton.min_mvaID,
            events.LeadPhoton.mvaID
        )
            
        events[("SubleadPhoton", "mvaID")] = awkward.where(
            (events.LeadPhoton.mvaID > self.options["sideband_cut"]) & (events.SubleadPhoton.mvaID < self.options["sideband_cut"]),
            events.Diphoton.min_mvaID,
            events.SubleadPhoton.mvaID
        )

        # Calculate per-event weight
        omega = integrate(x, pdf, self.options["sideband_cut"], events.Diphoton.max_mvaID)

        # Update weights
        events[CENTRAL_WEIGHT] = awkward.where(
            events.sideband_event,
            events[CENTRAL_WEIGHT] * omega * self.options["norm_factor"],
            events[CENTRAL_WEIGHT]
        )

        # In rare cases, the newly generated min_mvaID can be greater than max_mvaID. Switch them here:
        max_id = events.Diphoton.max_mvaID
        min_id = events.Diphoton.min_mvaID

        events[("Diphoton", "max_mvaID")] = awkward.where(events.sideband_event & (min_id > max_id), min_id, max_id)
        events[("Diphoton", "min_mvaID")] = awkward.where(events.sideband_event & (min_id > max_id), max_id, min_id)

        # Update process_id
        events["process_id"] = awkward.where(
            events.sideband_event,
            numpy.ones(len(events)) * DATA_DRIVEN_GJETS_PROCESS_ID,
            events.process_id
        )

        cut = events.Diphoton.min_mvaID > self.options["sideband_cut"]
        self.register_cuts(
                names = ["sideband cut"],
                results = [cut]
        )

        return cut, events
