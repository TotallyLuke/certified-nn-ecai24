from typing import List, Dict
from acasxu_smtverifier_helper.utils import Bound
import gurobipy as gp
from gurobipy import GRB, Model, quicksum

import torch
from torch.nn import Linear, ReLU

from acasxu_smtverifier_helper.conf import *

# Initialize Gurobi environment
env = gp.Env(empty=True)
env.setParam('OutputFlag', 0)
env.start()
import gurobipy
from gurobipy import *
import torch

env = gurobipy.Env(empty=True)
env.setParam('OutputFlag', 0)
env.start()


def milp_encoding(
        model,
        K=1,
        input_bounds: List[Dict[str, Bound]] = None,
        NN_Inputs=NN_Inputs,
        NN_Outputs=NN_Outputs,
        var_offset=0,
        gurobi_model=None,
    ):
    """
    MILP encoding of a feedforward ReLU neural network.

    Args:
        :param model: a torch.nn.Sequential model (Linear/ReLU alternating)
        :param K: number of copies (for multi-step encodings)
        :param input_bounds: list of dicts mapping input names to pairs of bounds
    """

    # allow injecting constraints into an existing model
    if gurobi_model is None:
        global env
        m = Model("neural network", env=env)
    else:
        m = gurobi_model

    # dictionaries for (neuron) variables
    m._x = dict()  # input
    m._h = dict()  # hidden
    m._y = dict()  # output

    Layers = list(model._modules.values())
    sum(len(dic) for dic in input_bounds)
    assert isinstance(Layers[0], torch.nn.Linear)

    assert K * len(NN_Inputs) == sum(len(d) for d in input_bounds), (
        f"input_bounds mismatch: expected {K * len(NN_Inputs)} bounds (K × inputs) "
        f"but got {sum(len(d) for d in input_bounds)}. Provided keys={ [list(d.keys()) for d in input_bounds] }"
    )

    for k in range(K):
        kk = var_offset + k

        # input layer
        for i in range(len(NN_Inputs)):
            vtype = GRB.BINARY if i in BinaryIndexes else GRB.CONTINUOUS
            domain = input_bounds[k][NN_Inputs[i]]
            m._x[kk, i] = m.addVar(lb=domain.lb, ub=domain.ub, vtype=vtype, name=f"x{kk}_{i}")

        w = model.state_dict()['0.weight'].tolist()
        b = model.state_dict()['0.bias'].tolist()
        for i in range(Layers[0].out_features):
            m._h[kk, 0, i] = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS,
                                     name=f"h{kk}_0_{i}")
            m.update()
            m.addConstr(m._h[kk, 0, i] == b[i] + quicksum(w[i][i1] * m._x[kk, i1] for i1 in range(Layers[0].in_features)))

        # hidden vars
        last = Layers[0]
        j = 1
        for lay in Layers[1:-1]:
            if isinstance(lay, torch.nn.Linear):
                w = model.state_dict()['{}.weight'.format(j)].tolist()
                b = model.state_dict()['{}.bias'.format(j)].tolist()
                for i in range(lay.out_features):
                    m._h[kk, j, i] = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS,
                                             name=f"h{kk}_{j}_{i}")
                    m.update()
                    m.addConstr(m._h[kk, j, i] == b[i] + quicksum(
                        w[i][i1] * m._h[kk, j - 1, i1] for i1 in range(lay.in_features)))

            elif isinstance(lay, torch.nn.ReLU):
                for i in range(last.out_features):
                    m._h[kk, j, i] = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS,
                                             name=f"h{kk}_{j}_{i}")
                    m.update()
                    m.addConstr(m._h[kk, j, i] == max_(m._h[kk, j - 1, i], constant=0),
                                name=f"ReLU{kk}_{j}_{i}")
            else:
                assert False, "unexpected type of layer"

            j += 1
            last = lay

        lay = Layers[-1]
        j = len(Layers) - 1
        assert isinstance(lay, torch.nn.Linear)
        w = model.state_dict()['{}.weight'.format(j)].tolist()
        b = model.state_dict()['{}.bias'.format(j)].tolist()
        for i in range(len(NN_Outputs)):
            m._y[kk, i] = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="y{}_{}".format(kk, i))
            m.update()
            m.addConstr(m._y[kk, i] == b[i] + quicksum(w[i][i1] * m._h[kk, j - 1, i1] for i1 in range(lay.in_features)))

        m.update()
    m.write("myfile.lp")  # for debug

    return m
