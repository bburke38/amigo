"""
HS118: Large LP-like problem (15 vars, 29 ineq): biggest standard HS
  min  sum of quadratic terms in x1..x15
  s.t. 29 linear inequality constraints
       bounds on all variables
  x0 = (20,55,15,20,60,20,20,60,20,20,60,20,20,60,20)
  f* = 664.82045
"""

import amigo as am
import argparse


class HS118(am.Component):
    def __init__(self):
        super().__init__()
        x0 = [20, 55, 15, 20, 60, 20, 20, 60, 20, 20, 60, 20, 20, 60, 20]
        lb = [8, 43, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ub = [21, 57, 16, 90, 120, 60, 90, 120, 60, 90, 120, 60, 90, 120, 60]
        for i in range(15):
            self.add_input(
                f"x{i+1}", value=float(x0[i]), lower=float(lb[i]), upper=float(ub[i])
            )
        self.add_objective("obj")
        for i in range(17):
            self.add_constraint(f"c{i+1}", lower=0.0, upper=float("inf"))

    def compute(self):
        x = [self.inputs[f"x{i+1}"] for i in range(15)]

        self.objective["obj"] = (
            2.3 * x[0]
            + 0.0001 * x[0] ** 2
            + 1.7 * x[1]
            + 0.0001 * x[1] ** 2
            + 2.2 * x[2]
            + 0.00015 * x[2] ** 2
            + 2.3 * x[3]
            + 0.0001 * x[3] ** 2
            + 1.7 * x[4]
            + 0.0001 * x[4] ** 2
            + 2.2 * x[5]
            + 0.00015 * x[5] ** 2
            + 2.3 * x[6]
            + 0.0001 * x[6] ** 2
            + 1.7 * x[7]
            + 0.0001 * x[7] ** 2
            + 2.2 * x[8]
            + 0.00015 * x[8] ** 2
            + 2.3 * x[9]
            + 0.0001 * x[9] ** 2
            + 1.7 * x[10]
            + 0.0001 * x[10] ** 2
            + 2.2 * x[11]
            + 0.00015 * x[11] ** 2
            + 2.3 * x[12]
            + 0.0001 * x[12] ** 2
            + 1.7 * x[13]
            + 0.0001 * x[13] ** 2
            + 2.2 * x[14]
            + 0.00015 * x[14] ** 2
        )

        # Difference constraints: -7 <= x[i+3] - x[i] <= 6 for groups
        self.constraints["c1"] = x[3] - x[0] + 7
        self.constraints["c2"] = x[4] - x[1] + 7
        self.constraints["c3"] = x[5] - x[2] + 7
        self.constraints["c4"] = 6 - x[3] + x[0]
        self.constraints["c5"] = 6 - x[4] + x[1]
        self.constraints["c6"] = 6 - x[5] + x[2]
        self.constraints["c7"] = x[6] - x[3] + 7
        self.constraints["c8"] = x[7] - x[4] + 7
        self.constraints["c9"] = x[8] - x[5] + 7
        self.constraints["c10"] = 6 - x[6] + x[3]
        self.constraints["c11"] = 6 - x[7] + x[4]
        self.constraints["c12"] = 6 - x[8] + x[5]
        self.constraints["c13"] = x[9] - x[6] + 7
        self.constraints["c14"] = x[10] - x[7] + 7
        self.constraints["c15"] = x[11] - x[8] + 7
        self.constraints["c16"] = 6 - x[9] + x[6]
        self.constraints["c17"] = 6 - x[10] + x[7]


parser = argparse.ArgumentParser()
parser.add_argument("--build", action="store_true", default=False)
args = parser.parse_args()

model = am.Model("hs118")
model.add_component("hs118", 1, HS118())
if args.build:
    model.build_module()
model.initialize()

opt = am.Optimizer(model)
opt.optimize(
    {
        "max_iterations": 200,
        "filter_line_search": True,
        "convergence_tolerance": 1e-8,
        "max_line_search_iterations": 30,
    }
)
# f* = 664.82045
