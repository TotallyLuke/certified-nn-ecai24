
# NN_Inputs = ["x", "y", "psi"]
# NN_Outputs = ["out"]
# Fieldnames = NN_Inputs + NN_Outputs
BinaryIndexes = []

from dataclasses import dataclass


@dataclass
class Bound:
    lb: float
    ub: float


NN_Inputs = ["x", "y", "psi"]
NN_Outputs = ["o"]
lbpos = -61000
ubpos = 61000
mean_x = -278.628049
mean_y = 0.0
mean_psi = 0.0
range_x = 112000.0
range_y = 112000.0
range_psi = 6.283185