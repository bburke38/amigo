"""
HS116: Large mixed constraints (13 vars, 15 ineq): one of the hardest HS problems
  min  x11 + x12 + x13
  s.t. 15 nonlinear inequalities
       0.1 <= x1 <= 1, 0.1 <= x2 <= 1, etc.
  x0 specified below, f* = 97.588409
"""

import amigo as am
import argparse


class HS116(am.Component):
    def __init__(self):
        super().__init__()
        # Bounds from the problem
        lb = [0.1, 0.1, 0.1, 0.001, 0.1, 0.1, 0.1, 0.1, 500, 0.1, 1, 0.0001, 0.0001]
        ub = [1, 1, 1, 0.1, 0.9, 0.9, 1000, 1000, 1000, 500, 150, 150, 150]
        x0 = [0.5, 0.8, 0.9, 0.01, 0.5, 0.5, 150, 150, 500, 250, 50, 50, 50]
        for i in range(13):
            self.add_input(f"x{i+1}", value=x0[i], lower=lb[i], upper=ub[i])
        self.add_objective("obj")
        for i in range(15):
            self.add_constraint(f"c{i+1}", lower=0.0, upper=float("inf"))

    def compute(self):
        x = [self.inputs[f"x{i+1}"] for i in range(13)]
        a = 0.002
        b = 1.262626
        c = 1.231059
        d = 0.03475
        e = 0.975
        f = 0.00975

        self.objective["obj"] = x[10] + x[11] + x[12]

        self.constraints["c1"] = x[2] - x[1]
        self.constraints["c2"] = x[1] - x[0]
        self.constraints["c3"] = 1 - a * x[6] / x[3]
        self.constraints["c4"] = 1 - a * x[7] / x[4]
        self.constraints["c5"] = x[10] + x[11] + x[12] - 50
        self.constraints["c6"] = x[10] - b * x[0] * x[5]
        self.constraints["c7"] = x[11] - c * x[1] * x[6] + c * x[2] * x[6]
        self.constraints["c8"] = x[12] + d * x[8] - e * x[2] * x[8]
        self.constraints["c9"] = x[10] + f * x[8] - e * x[0] * x[8]
        self.constraints["c10"] = 500 * x[0] - x[10] * x[5]
        self.constraints["c11"] = 500 * x[1] - x[11] * x[6] + x[11] * x[6]
        self.constraints["c12"] = 500 * x[2] - x[12] * x[8]
        self.constraints["c13"] = x[3] * (x[10] + x[11])
        self.constraints["c14"] = x[4] * (x[11] + x[12])
        self.constraints["c15"] = x[5] * x[10] + x[6] * x[11] + x[8] * x[12]


parser = argparse.ArgumentParser()
parser.add_argument("--build", action="store_true", default=False)
args = parser.parse_args()

model = am.Model("hs116")
model.add_component("hs116", 1, HS116())
if args.build:
    model.build_module()
model.initialize()

opt = am.Optimizer(model)
opt.optimize(
    {
        "max_iterations": 300,
        "filter_line_search": True,
        "convergence_tolerance": 1e-8,
        "max_line_search_iterations": 30,
    }
)
# f* = 97.588409
