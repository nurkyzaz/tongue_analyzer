"""Structured Stage-1 output — the clean JSON interface consumed by Stage 2 (interpretation).

Keeping this schema stable is what lets Stage 1 and Stage 2 evolve independently.
"""
from dataclasses import dataclass, field, asdict
from typing import Optional
import json


@dataclass
class Characteristic:
    value: str                 # predicted class label, e.g. "greasy_thick"
    confidence: float          # softmax prob of the predicted class
    description: str = ""       # human-readable name of the characteristic


@dataclass
class Stage1Output:
    sid: Optional[str] = None
    key_characteristics: dict = field(default_factory=dict)   # char -> Characteristic
    quality: dict = field(default_factory=dict)               # mask coverage, accepted, reasons
    mask_path: Optional[str] = None
    model_version: str = "tih-stage1-v0.1"

    def to_json(self, indent=2) -> str:
        d = asdict(self)
        return json.dumps(d, indent=indent, ensure_ascii=False)

    def summary_lines(self):
        """Compact human-readable lines (the 'Quantitative Summary' block for the app)."""
        out = []
        for ch, c in self.key_characteristics.items():
            out.append(f"{c['description'] or ch}: {c['value']} ({c['confidence']*100:.0f}%)")
        return out
