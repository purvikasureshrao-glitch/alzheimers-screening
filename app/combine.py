"""The only cross-modality step: a stated rule, not a model. 2 or more of 4 flagged -> recommend clinical follow-up.

No weighting, no averaging, no fused score, and no accuracy claim (the four datasets share no participants, so none can exist)."""
from .schema import ORDER


def combine(results):
    by = {r.modality: r for r in results}
    run = [m for m in ORDER if m in by and by[m].status != "not_run" and by[m].counts]
    flagged = [m for m in run if by[m].status == "flagged"]
    excluded = [m for m in ORDER if m in by and by[m].status != "not_run" and not by[m].counts]
    follow_up = len(flagged) >= 2
    if follow_up:
        text = (f"{len(flagged)} of {len(run)} completed screens were flagged ({', '.join(flagged)}). "
                "Under the stated rule (2 or more flagged), clinical follow-up is recommended.")
    else:
        text = (f"{len(flagged)} of {len(run)} completed screens were flagged. Under the stated rule "
                "(2 or more flagged) follow-up is not triggered.")
    if excluded:
        text += f" Shown but not counted (experimental, unvalidated): {', '.join(excluded)}."
    text += " This is a screening rule, not a diagnosis, and it carries no accuracy claim."
    return {"flagged": flagged, "completed": run, "excluded": excluded, "follow_up": follow_up, "text": text}
