"""States each modality's own flag and how many of the four were flagged. No weighting, no averaging, no fused score, no accuracy claim.

Display wording was simplified at the user's request (2026-10-19): the rule text and the "experimental / not counted" notes are gone from the page.
`follow_up` (2 or more flagged) is kept only so the page can colour the box; it is not written out as a recommendation."""
from .schema import NAMES, ORDER


def combine(results):
    by = {r.modality: r for r in results}
    run = [m for m in ORDER if m in by and by[m].status != "not_run"]
    flagged = [m for m in run if by[m].status == "flagged"]
    parts = []
    for m in ORDER:
        r = by.get(m)
        parts.append(f"{NAMES[m]}: " + ("not completed" if r is None or r.status == "not_run" else "flagged" if r.status == "flagged" else "not flagged"))
    text = ". ".join(parts) + f". {len(flagged)} of {len(run)} completed screens flagged."
    return {"flagged": flagged, "completed": run, "excluded": [], "follow_up": len(flagged) >= 2, "text": text}
