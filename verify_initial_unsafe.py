# verify_motion_property.py
import math
from types import SimpleNamespace

import torch
import torch.nn as nn
import gurobipy as gp
from gurobipy import GRB
from acasxu_smtverifier_helper.milp import milp_encoding
from acasxu_smtverifier_helper.utils import *
import time
from acasxu_smtverifier_helper.conf import NN_Inputs, NN_Outputs
lbpos = -10000
ubpos = 10000

input_bounds = {"x": Bound(0.0, ubpos),"y": Bound(-61000.0, ubpos),"psi": Bound(-math.pi, math.pi)}


def verify_unsafe(model: nn.Module, input_bounds, verbose=False) -> Optional[List[float]]:
    m = milp_encoding(model, K=1, input_bounds=[input_bounds], NN_Inputs=NN_Inputs, NN_Outputs=NN_Outputs)
    x = m.getVarByName("x0_0")
    y = m.getVarByName("x0_1")
    intercept = 2 * math.sqrt((10000**2)/2)
    m.addConstr(-x + y <=  intercept, name="ineq1")
    m.addConstr( x + y <=  intercept, name="ineq2")
    m.addConstr(-x + y >= -intercept, name="ineq3")
    m.addConstr( x + y >= -intercept, name="ineq4")
    m.addConstr( x >= -10000, name="ineq5")
    m.addConstr( x <= 10000, name="ineq6")
    # m.addConstr( x**2 + y**2 <= 10000, name="ineq6")
    m.addConstr( y >= -10000, name="ineq7")
    m.addConstr( y <= 10000, name="ineq8")

    out = m.getVarByName("y0_0")
    m.addConstr(out <= 0.0, name="output_negative")


    if verbose:
        print("== Optimizing ==")

    start_time = time.time()
    m.optimize()

    if verbose:
        print(f"\nSolver runtime: {m.Runtime:.3f} seconds")
        print(f"Elapsed wall time: {time.time() - start_time:.3f} seconds")
        print(f"Nodes explored: {m.NodeCount}")
        print(f"MIP gap: {m.MIPGap}")

    if m.status == GRB.Status.INFEASIBLE:
        print("Verified: No unsafe state leads to output ≤ 0 (property holds).")
    elif m.status in [GRB.Status.OPTIMAL, GRB.Status.TIME_LIMIT]:
        print(f"Property violated (output = {out.X:.6f})")
        x_cex = m.getVarByName("x0_0").X
        y_cex = m.getVarByName("x0_1").X
        p_cex = m.getVarByName("x0_2").X
        print(f"({x_cex:.2f}, {y_cex:.2f}, {p_cex:.2f})")

        return [x_cex, y_cex, p_cex]
    else:
        print(f"Solver returned unexpected status: {m.status}")

    m.dispose()
    return None



def verify_initial(model: nn.Module, input_bounds, verbose=False) -> Optional[List[float]]:


    m = milp_encoding(model, K=1, input_bounds=[input_bounds], NN_Inputs=NN_Inputs, NN_Outputs=NN_Outputs)

    out = m.getVarByName("y0_0")
    m.addConstr(out >= 0.0, name="output_negative")

    if verbose:
        print("== Optimizing ==")

    start_time = time.time()
    m.optimize()

    if verbose:
        print(f"\nSolver runtime: {m.Runtime:.3f} seconds")
        print(f"Elapsed wall time: {time.time() - start_time:.3f} seconds")
        print(f"Nodes explored: {m.NodeCount}")
        print(f"MIP gap: {m.MIPGap}")
        print(model)

    if m.status == GRB.Status.INFEASIBLE:
        print("Verified: No initial state leads to output > 0 (property holds).")
    elif m.status in [GRB.Status.OPTIMAL, GRB.Status.TIME_LIMIT]:
        print(f"Property violated (output = {out.X:.6f})")
        x_cex = m.getVarByName("x0_0").X
        y_cex = m.getVarByName("x0_1").X
        p_cex = m.getVarByName("x0_2").X
        print(f"({x_cex:.2f}, {y_cex:.2f}, {p_cex:.2f})")
        return [x_cex, y_cex, p_cex]
    else:
        print(f"Solver returned unexpected status: {m.status}")

    m.dispose()
    return None

import torch
import torch.nn as nn
import onnx
import onnx2torch

def load_trained_model(path: str) -> nn.Module:
    """load an onnx model and convert it to a torch module"""
    onnx_model = onnx.load(path)
    torch_model = onnx2torch.convert(onnx_model)
    torch_model.eval()
    return torch_model

def graphmodule_to_sequential(graph_module: torch.fx.GraphModule) -> nn.Sequential:
    """extract linear and relu layers from a graphmodule into a sequential model"""
    layers = []
    for _, module in graph_module.named_modules():
        if isinstance(module, (nn.Linear, nn.ReLU)):
            layers.append(module)
    return nn.Sequential(*layers)

INIT_BOUNDS = SimpleNamespace(
    X_MIN=58000,
    X_MAX=61000,
    Y_MIN=1000,
    Y_MAX=3000,
    PSI_MIN=-0.2,
    PSI_MAX=0.2
)
if __name__ == "__main__":
    # model_path = "/home/lucav/Downloads/acasxu_barrier_cegis_rho_theta_psi/model_outputs/trained_barrier.onnx"
    # model_path = "/home/lucav/git_workspace/acasxu_barrier_cegis/model_outputs/trained_barrier.onnx"
    model_path = "/home/lucav/git_workspace/acasxu_barrier_cegis/model_outputs/trained_barrier19.onnx"

    graph_model = load_trained_model(model_path)
    model = graphmodule_to_sequential(graph_model)
    # print(model)
    cex_unsafe = verify_unsafe(model, input_bounds)
    input_bounds = {"x": Bound(INIT_BOUNDS.X_MIN, INIT_BOUNDS.X_MAX),"y": Bound(INIT_BOUNDS.Y_MIN, INIT_BOUNDS.Y_MAX),
                    "psi": Bound(INIT_BOUNDS.PSI_MIN, INIT_BOUNDS.PSI_MAX)}
    cex_initial = verify_initial(model, input_bounds)

