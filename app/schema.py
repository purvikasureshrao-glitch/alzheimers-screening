"""Result schema for the four independent screening modalities (display order: Clock, Motor, Eye, Speech).

Each modality reports a flag from its own method and its own evidence. Nothing is combined numerically (PLAN rule 2)."""
from dataclasses import asdict, dataclass, field

ORDER = ("clock", "motor", "eye", "speech")
STATUSES = ("flagged", "not_flagged", "not_run")


@dataclass
class Result:
    modality: str
    status: str                       # flagged | not_flagged | not_run
    label: str                        # wording shown to the user; never a diagnosis or stage
    measured: dict = field(default_factory=dict)   # what was captured, e.g. {"taps per second": 5.1}
    evidence: str = ""                # the modality's own published/measured evidence, with CI, n and protocol
    caveat: str = ""                  # known limits (domain shift, pilot, unvalidated)
    counts: bool = True               # False = shown but excluded from the 2-of-4 rule (experimental, e.g. webcam eye)

    def __post_init__(self):
        if self.modality not in ORDER:
            raise ValueError(f"unknown modality {self.modality!r}")
        if self.status not in STATUSES:
            raise ValueError(f"unknown status {self.status!r}")

    def to_dict(self):
        return asdict(self)
