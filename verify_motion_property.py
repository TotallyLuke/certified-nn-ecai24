# verify_motion_property.py
import math
import time
from typing import List, Dict

import gurobipy

from acasxu_smtverifier_helper.add_argmax_output_constraints import add_argmax_output_constraints
from acasxu_smtverifier_helper.arc_bbox import get_arc_bounding_box
from acasxu_smtverifier_helper.milp import milp_encoding
from acasxu_smtverifier_helper.single_verify_motion_property_trapez import single_verify_motion_property
from acasxu_smtverifier_helper.NNet.python.nnet import NNet
from acasxu_smtverifier_helper.test_nnet import FeedForwardNet
from acasxu_smtverifier_helper.utils import Bound, VerificationResult, summarize_model, merge_bounding_boxes
from acasxu_smtverifier_helper.conf import ubpos, lbpos, NN_Inputs, NN_Outputs, mean_x, range_x, mean_y, range_y, \
    mean_psi, range_psi
from acasxu_smtverifier_helper.trapezoid import secant_line_coeffs, tangent_line_coeffs, psi_to_xy

import torch
import onnx
import onnx2torch
from gurobipy import GRB
from acasxu_smtverifier_helper.barrier_loader import load_onnx_as_sequential


def multiple_verify_motion_property(model: torch.nn.Module, turn_index, controller, psi_deg_init, n_intervals=3,
                                    interv_width=3.0, verbose=False):
    ACTIONS = [math.radians(a) for a in [0.0, 1.5, -1.5, 3.0, -3.0]]
    turn = ACTIONS[turn_index]

    psi_init = math.radians(psi_deg_init)
    underflow = math.radians(psi_init + 3.0) - turn <= -math.pi + 0.001
    overflow = math.radians(psi_init) - turn >= math.pi - 0.001
    assert not (underflow and overflow)

    current_state_bounds = {
        "x": Bound(0.0, ubpos),
        "y": Bound(-61000.0, ubpos),
        "psi": Bound(math.radians(psi_deg_init), math.radians(psi_init + n_intervals * interv_width))
    }
    next_state_bounds = {
        "x": Bound(lbpos, ubpos),
        "y": Bound(lbpos, ubpos),
        "psi": Bound(current_state_bounds["psi"].lb - turn + (underflow * 2 * math.pi) - (overflow * 2 * math.pi),
                     current_state_bounds["psi"].ub - turn + (underflow * 2 * math.pi) - (overflow * 2 * math.pi))
    }
    input_bounds: List[Dict[str, Bound]] = [current_state_bounds, next_state_bounds]

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

    vx_start = math.cos(math.radians(psi_deg_init)) * 200.0
    vy_start = math.sin(math.radians(psi_deg_init)) * 200.0
    vx_end = math.cos(math.radians(psi_deg_init + interv_width * n_intervals)) * 200.0
    vy_end = math.sin(math.radians(psi_deg_init + interv_width * n_intervals)) * 200.0

    # main vx, vy
    vx = m.addVar(lb=min(vx_start, vx_end) - 0.001, ub=max(vx_start, vx_end) + 0.001, name="vx")
    vy = m.addVar(lb=min(vy_start, vy_end) - 0.001, ub=max(vy_start, vy_end) + 0.001, name="vy")

    b = [m.addVar(vtype=GRB.BINARY, name=f"bbox_sel_{i}") for i in range(n_intervals)]
    m.addConstr(gurobipy.quicksum(b) == 1, name="one_bbox_active")
    # per-bbox candidate variables and indicator constraints

    psi_interval = 3.0

    for i in range(n_intervals):
        r = i * interv_width
        s = (i + 1) * interv_width

        A1, B1, C1 = tangent_line_coeffs(*psi_to_xy(math.radians(psi_deg_init + r), 200.0))
        A2, B2, C2 = tangent_line_coeffs(*psi_to_xy(math.radians(psi_deg_init + ((r + s) / 2.0)), 200.0))
        A3, B3, C3 = tangent_line_coeffs(*psi_to_xy(math.radians(psi_deg_init + s), 200.0))
        A4, B4, C4 = secant_line_coeffs(*psi_to_xy(math.radians(psi_deg_init + r), 200.0),
                                        *psi_to_xy(math.radians(psi_deg_init + s), 200.0))
        if verbose:
            print(f"in interval ",
                  A1 * vx_start + B1 * vy_start + C1 <= 0 and A2 * vx_start + B2 * vy_start + C2 <= 0
                  and A3 * vx_start + B3 * vy_start + C3 <= 0 <= A4 * vx_start + B4 * vy_start + C4)

            print(A1 * vx_end + B1 * vy_end + C1 <= 0 and A2 * vx_end + B2 * vy_end + C2 <= 0
                  and A3 * vx_end + B3 * vy_end + C3 <= 0 <= A4 * vx_end + B4 * vy_end + C4)

        m.addConstr(A1 * vx + B1 * vy + C1 <= 0, name=f"trapezoid_1")
        m.addConstr(A2 * vx + B2 * vy + C2 <= 0, name=f"trapezoid_2")
        m.addConstr(A3 * vx + B3 * vy + C3 <= 0, name=f"trapezoid_3")
        m.addGenConstrIndicator(b[i], True, A4 * vx + B4 * vy + C4 >= 0, name=f"trapezoid_4_{i}")

    x_intermediate = x + vx
    y_intermediate = y + vy

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
        psi_next == psi + delta_psi + (underflow * 2 * math.pi) - (overflow * 2 * math.pi),
        name="state_update_psi"
    )

    controller_input_bounds = [{"x": Bound(-0.5, +0.5), "y": Bound(-0.5, +0.5), "psi": Bound(-0.5, +0.5)}]
    controller_Outputs = ["o1", "o2", "o3", "o4", "o5"]

    kk = 2
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
    add_argmax_output_constraints(m, outputs_controller, turn_index)

    start_time = time.time()
    m.optimize()
    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"Optimization time: {elapsed_time:.3f} seconds")
    if m.Status == GRB.OPTIMAL:

        x0 = [m.getVarByName(f"x0_{i}").X for i in range(len(NN_Inputs))]
        x1 = [m.getVarByName(f"x1_{i}").X for i in range(len(NN_Inputs))]

        print("Optimal solution found\n")

        result = VerificationResult(
            safe=False,
            counterexample=x0,
            counterexample_next=x0,
            counterexample_normalized=[xnorm.X, ynorm.X, psinorm.X]
        )
    elif m.Status == GRB.INFEASIBLE:
        result = VerificationResult(True, None, None, None)
    else:
        print(f"Model status: {m.Status}")
        result = VerificationResult(False, None, None, None)
    m.dispose()
    return result


def sequential_multi_verification(psis, turn, n_intervals, model, controller, interv_width = 3.0, verbose = False):
    to_return = []
    for psi in psis:
        if verbose:
            print(f"turn {turn}, psi {psi} + ({int(n_intervals)}*{int(interv_width)})")

        result: VerificationResult = multiple_verify_motion_property(model, turn, controller, psi,
                                                                     n_intervals=n_intervals, interv_width=interv_width,
                                                                     verbose=False)
        if not result.safe:
            pair = [result.counterexample, result.counterexample_next]
            # x_cex, y_cex, psi_cex = result.counterexample
            # x_cex_nxt, y_cex_nxt, psi_cex_nxt = result.counterexample_next
            # res5 = nn0.evaluate_network(result.counterexample_normalized)
            # print(f"Counterexample found at [{x_cex:.1f}, {y_cex:.1f}, {psi_cex:.2f}]")
            to_return.append(pair)
    return to_return


if __name__ == "__main__":
    model_path = "/home/lucav/git_workspace/acasxu_barrier_cegis/model_outputs/trained_barrier19.onnx"
    model = load_onnx_as_sequential(model_path)
    # nn0 = NNet("/home/lucav/thesis/ACASXu-20251018T095242Z-1-001/ACASXu/rectangular-coordinates/networks/medium/HCAS_rect_v6_pra0_tau00_25HU_02042.nnet")
    # nn0 = NNet("/home/lucav/thesis/ACASXu-20251018T095242Z-1-001/ACASXu/rectangular-coordinates/networks/mediumbig/HCAS_rect_v6_pra0_tau00_22HU_03565.nnet")
    nn0 = NNet(
        "/home/lucav/thesis/ACASXu-20251018T095242Z-1-001/ACASXu/rectangular-coordinates/networks/medium23/HCAS_rect_v6_pra0_tau00_23HU_03169.nnet")
    seq = FeedForwardNet(nn0)
    counterexamples = []
    TURN = math.radians(-3.0)
    TURN = 4

    sequential_multi_verification([-180, -171, -162, -153, -144], TURN, 3)
    sequential_multi_verification([-135, -126, -117, -108, -99], TURN, 3)
    sequential_multi_verification([-90, -45, 0, 45], TURN, 15)
    sequential_multi_verification([90, 99, 108, 117, 126, 135, 144, 153, 162], TURN, 3)
    sequential_multi_verification([171], TURN, 2)
    for psi in [177]:
        result: VerificationResult = single_verify_motion_property(model, TURN, seq.net, psi, verbose=False)
        if not result.safe:
            x_cex, y_cex, psi_cex = result.counterexample
            res5 = nn0.evaluate_network(result.counterexample_normalized)
            print(f"Counterexample found at [{x_cex:.1f}, {y_cex:.1f}, {psi_cex:.2f}]")

    print("===-=== SINGULAR ===-===")
    count_safe = 0
    count_unsafe = 0
    for psi_deg in range(-180, 180, 3):
        print("psi: ", psi_deg)
        result: VerificationResult = single_verify_motion_property(model, TURN, seq.net, psi_deg, verbose=False)
        if not result.safe:
            x_cex, y_cex, psi_cex = result.counterexample
            res5 = nn0.evaluate_network(result.counterexample_normalized)
            print(f"Counterexample found at [{x_cex:.1f}, {y_cex:.1f}, {psi_cex:.2f}]")
            print("res5: ", res5)
            count_unsafe += 1
        else:
            count_safe += 1

    TURN = 2

    sequential_multi_verification([-171], TURN, 2)
    sequential_multi_verification([-162, -153, -144, -135, -126, -117, -108, -99], TURN, 3)
    sequential_multi_verification([-90, -45, 0, 45], TURN, 15)
    sequential_multi_verification([90, 99, 108, 117, 126, 135, 144, 153, 162, 171], TURN, 3)
    for psi in [-177]:
        result: VerificationResult = single_verify_motion_property(model, TURN, seq.net, psi, verbose=False)
        if not result.safe:
            x_cex, y_cex, psi_cex = result.counterexample
            res5 = nn0.evaluate_network(result.counterexample_normalized)
            print(f"Counterexample found at [{x_cex:.1f}, {y_cex:.1f}, {psi_cex:.2f}]")

    print("===-=== SINGULAR ===-===")
    count_safe = 0
    count_unsafe = 0
    for psi_deg in range(-180, 180, 3):
        print("psi: ", psi_deg)
        result: VerificationResult = single_verify_motion_property(model, TURN, seq.net, psi_deg, verbose=False)
        if not result.safe:
            x_cex, y_cex, psi_cex = result.counterexample
            res5 = nn0.evaluate_network(result.counterexample_normalized)
            print(f"Counterexample found at [{x_cex:.1f}, {y_cex:.1f}, {psi_cex:.2f}]")
            print("res5: ", res5)
            count_unsafe += 1
        else:
            count_safe += 1

    print("safe: ", count_safe)
    print("unsafe: ", count_unsafe)
