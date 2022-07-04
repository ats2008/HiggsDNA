import awkward

import logging
logger = logging.getLogger(__name__)

def data_driven_gjets(events, central_only, year):
    required_fields = [
        ("LeadPhoton", "mvaID"), ("SubleadPhoton", "mvaID")
    ]

    missing_fields = awkward_utils.missing_fields(events, required_fields)

    if missing_fields:
        message = "[photon_systematics : photon_preselection_sf] The events array is missing the following fields: %s which are needed as inputs." % (str(missing_fields))
        logger.exception(message)
        raise ValueError(message)

    variations = {
            "central" : awkward.ones_like(events.event),
            "up" : awkward.ones_like(events.event),
            "down" : awkward.ones_like(events.event)
    }

    return variations
