from dataclasses import dataclass
from typing import Optional, List, Dict
import torch.nn as nn


def summarize_model(m) -> None:
    """prints a summary of the Gurobi MILP model structure"""
    print("\n=== Model Summary ===")
    print(f"Variables:   {m.NumVars}")
    print(f"  Continuous: {m.NumVars - m.NumBinVars}")
    print(f"  Binary:     {m.NumBinVars}")
    print(f"Constraints: {m.NumConstrs}")
    print(f"Objective sense: {'Minimize' if m.ModelSense == 1 else 'Maximize'}")
    print("======================\n")


def make_network() -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(3, 48),
        nn.ReLU(),
        nn.Linear(48, 48),
        nn.ReLU(),
        nn.Linear(48, 48),
        nn.ReLU(),
        nn.Linear(48, 1)
    )


@dataclass
class VerificationResult:
    safe: bool  # True = INFEASIBLE, False = OPTIMAL
    counterexample: Optional[List[float]]
    counterexample_next: Optional[List[float]]
    counterexample_normalized: Optional[List[float]]


@dataclass
class Bound:
    lb: float
    ub: float


def merge_bounding_boxes(bboxes: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Computes the absolute min/max bounds across a list of bounding box dictionaries.
    """
    if not bboxes:
        return {}

    return {
        'vx_lower': max(min(box['vx_lower'] for box in bboxes)-0.001, -200.0),
        'vx_upper': min(max(box['vx_upper'] for box in bboxes)+0.001, 200.0),
        'vy_lower': max(min(box['vy_lower'] for box in bboxes)-0.001, -200.0),
        'vy_upper': min(max(box['vy_upper'] for box in bboxes)+0.001, 200.0)
    }

