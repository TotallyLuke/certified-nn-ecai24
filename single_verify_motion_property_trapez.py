import math
import time
from typing import List, Dict

from gurobipy import GRB
from torch import nn

from acasxu_smtverifier_helper.arc_bbox import get_arc_bounding_box
from acasxu_smtverifier_helper.milp import milp_encoding
from acasxu_smtverifier_helper.utils import Bound, summarize_model
from acasxu_smtverifier_helper.conf import ubpos, lbpos, NN_Inputs, NN_Outputs, mean_x, range_x, mean_y, range_y, \
    mean_psi, range_psi
from acasxu_smtverifier_helper.add_argmax_output_constraints import add_argmax_output_constraints
from acasxu_smtverifier_helper.trapezoid import secant_line_coeffs, tangent_line_coeffs, psi_to_xy
from acasxu_smtverifier_helper.utils import VerificationResult



def single_verify_motion_property(model: nn.Module, turn_index, controller, psi_deg_init, interv_width = 3.0, verbose=False):
    ACTIONS = [math.radians(a) for a in [0.0, 1.5, -1.5, 3.0, -3.0]]
    turn = ACTIONS[turn_index]

    underflow = math.radians(psi_deg_init + 3.0) - turn <= -math.pi + 0.001
    overflow = math.radians(psi_deg_init) - turn >= math.pi - 0.001
    assert not (underflow and overflow)
    current_state_bounds = {
        "x": Bound(0.0, ubpos),
        "y": Bound(-61000.0, ubpos),
        "psi": Bound(math.radians(psi_deg_init), math.radians(psi_deg_init + 3.0))
    }
    next_state_bounds = {
        "x": Bound(lbpos, ubpos),
        "y": Bound(lbpos, ubpos),
        "psi": Bound(current_state_bounds["psi"].lb - turn + (underflow*2*math.pi) - (overflow*2*math.pi),
                     current_state_bounds["psi"].ub - turn + (underflow*2*math.pi) - (overflow*2*math.pi))
    }
    input_bounds: List[Dict[str, Bound]] = [current_state_bounds, next_state_bounds]

    assert turn in [math.radians(a) for a in [0.0, 1.5, -1.5, 3.0, -3.0]]



    m = milp_encoding(model, K=2, input_bounds=input_bounds, NN_Inputs=NN_Inputs, NN_Outputs=NN_Outputs)
    if verbose:
        summarize_model(m)

    # turn and its sin&cos are fixed
    delta_psi = -turn
    cos_dp = math.cos(turn)
    sin_dp = math.sin(turn)

    # main state variables
    x = m.getVarByName("x0_0")
    y = m.getVarByName("x0_1")
    psi = m.getVarByName("x0_2")

    bbox = get_arc_bounding_box(input_bounds[0]["psi"].lb, input_bounds[0]["psi"].ub, speed=200.0)

    vx_start = bbox['vx_lower']
    vy_start = bbox['vy_lower']
    vx_end = bbox['vx_upper']
    vy_end = bbox['vy_upper']


    # main vx, vy
    vx = m.addVar(lb=max(min(vx_start, vx_end)-0.001, -200.0), ub=min(max(vx_start, vx_end)+0.001, 200.0), name="vx")
    vy = m.addVar(lb=max(min(vy_start, vy_end)-0.001, -200.0), ub=min(max(vy_start, vy_end)+0.001, 200.0), name="vy")

    x_intermediate = x + vx
    y_intermediate = y + vy
    psi_interval = interv_width
    A1, B1, C1 = tangent_line_coeffs(*psi_to_xy(math.radians(psi_deg_init), 200.0))
    A2, B2, C2 = tangent_line_coeffs(*psi_to_xy(math.radians(psi_deg_init + psi_interval / 2.0), 200.0))
    A3, B3, C3 = tangent_line_coeffs(*psi_to_xy(math.radians(psi_deg_init + psi_interval), 200.0))
    A4, B4, C4 = secant_line_coeffs(*psi_to_xy(math.radians(psi_deg_init), 200.0),
                                    *psi_to_xy(math.radians(psi_deg_init + psi_interval), 200.0))

    m.addConstr(A1 * vx + B1 * vy + C1 <= 0, name="trapezoid_1")
    m.addConstr(A2 * vx + B2 * vy + C2 <= 0, name="trapezoid_2")
    m.addConstr(A3 * vx + B3 * vy + C3 <= 0, name="trapezoid_3")
    m.addConstr(A4 * vx + B4 * vy + C4 >= 0, name="trapezoid_4")

    x_next = m.getVarByName("x1_0")
    y_next = m.getVarByName("x1_1")
    psi_next = m.getVarByName("x1_2")

    m.addConstr(
        x_next == cos_dp * x_intermediate + sin_dp * y_intermediate - 200,
        name="state_update_x"
    )

    m.addConstr(
        y_next == -sin_dp * x_intermediate + cos_dp * y_intermediate,
        name="state_update_y"
    )

    # add_angle_wrapping_constraint(m, psi, delta_psi, psi_next, "psi")
    m.addConstr(
        psi_next == psi + delta_psi + (underflow*2*math.pi) - (overflow*2*math.pi),
        name="state_update_psi"
    )

    controller_input_bounds = [{"x": Bound(-0.5, +0.5), "y": Bound(-0.5, +0.5), "psi": Bound(-0.5, +0.5)}]
    controller_Outputs = ["o1", "o2", "o3", "o4", "o5"]
    kk=2
    m = milp_encoding(controller, K=1, input_bounds=controller_input_bounds, NN_Inputs=NN_Inputs,
                      NN_Outputs=controller_Outputs, var_offset=kk, gurobi_model=m)


    xnorm = m.getVarByName("x2_0")
    ynorm = m.getVarByName("x2_1")
    psinorm = m.getVarByName("x2_2")


    m.addConstr(xnorm == ((x - mean_x) / range_x), name="normalize_x")
    m.addConstr(ynorm == ((y - mean_y) / range_y), name="normalize_y")
    m.addConstr(psinorm == ((psi - mean_psi) / range_psi), name="normalize_psi")

    m.write("motion_model.lp")
    output = m.getVarByName(f"y0_0")
    output_next = m.getVarByName(f"y1_0")
    outputs_controller = [m.getVarByName(f"y{kk}_{i}") for i in range(5)]
    # check existence of two inputs producing outputs of opposite sign
    m.addConstr(output <= 0.0, name="output_nonpositive")
    m.addConstr(output_next >= 0.05, name="output_next_positive")
    add_argmax_output_constraints(m, outputs_controller, 4)

    start_time = time.time()
    m.optimize()
    end_time = time.time()
    elapsed_time = end_time - start_time
    if m.Status == GRB.OPTIMAL:

        x0 = [m.getVarByName(f"x0_{i}").X for i in range(len(NN_Inputs))]
        x1 = [m.getVarByName(f"x1_{i}").X for i in range(len(NN_Inputs))]

        vx = m.getVarByName("vx").X
        vy = m.getVarByName("vy").X

        result = VerificationResult(
            safe=False,
            counterexample=x0,
            counterexample_next=x1,
            counterexample_normalized=[xnorm.X, ynorm.X, psinorm.X]
        )
    elif m.Status == GRB.INFEASIBLE:
        result = VerificationResult(True, None, None, None)
    else:
        print(f"Model status: {m.Status}")
        result = VerificationResult(False, None, None, None)
    # print output based on verbose flag
    if verbose:
        status = "✓ SAFE" if result.safe else "✗ UNSAFE"
        print(f"Turn {turn_index} | ψ={psi_deg_init:+7.2f}° | {status} | {elapsed_time:6.2f}s")

    m.dispose()
    return result
