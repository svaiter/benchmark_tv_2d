from benchopt import BaseObjective
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from benchmark_utils.shared import huber
    from benchmark_utils.matrix_op import grad


class Objective(BaseObjective):
    min_benchopt_version = "1.3"
    name = "TV2D"

    parameters = {'reg': [0.02],
                  'delta': [0.9],
                  'isotropy': ["anisotropic", "isotropic", "split"],
                  'data_fit': ["lsq", "huber"]}

    def __init__(self, reg=0.02, delta=0.1,
                 isotropy="anisotropic", data_fit="lsq"):
        self.reg = reg
        self.delta = delta
        self.isotropy = isotropy
        self.data_fit = data_fit

    def set_data(self, A, y):
        self.A = A
        self.y = y
        self.reg = self.reg

    def compute(self, u):
        # R means "residual"
        R = self.y - self.A @ u
        if self.data_fit == "lsq":
            loss = .5 * np.linalg.norm(R) ** 2
        else:
            loss = huber(R, self.delta)
        if self.isotropy == "isotropic":
            penality = self.isotropic_tv_value(u)
        else:
            penality = self.anisotropic_tv_value(u)
        return loss + self.reg * penality

    def get_one_solution(self):
        return np.zeros(self.y.shape)

    def get_objective(self):
        return dict(A=self.A,
                    reg=self.reg,
                    delta=self.delta,
                    data_fit=self.data_fit,
                    y=self.y,
                    isotropy=self.isotropy)

    def isotropic_tv_value(self, u):
        gh, gv = grad(u)
        return (np.sqrt(gh ** 2 + gv ** 2)).sum()

    def anisotropic_tv_value(self, u):
        gh, gv = grad(u)
        return (np.abs(gh) + np.abs(gv)).sum()
