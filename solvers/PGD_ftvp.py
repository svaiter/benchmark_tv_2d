from benchopt import BaseSolver
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    import pyftvp
    from benchmark_utils.shared import grad_F
    from benchmark_utils.shared import get_l2norm


class Solver(BaseSolver):
    """Primal forward-backward for anisoTV using prox-tv."""

    name = 'Primal PGD ftvp'

    # install_cmd = 'conda'
    # We need blas devel to get the include file for BLAS/LAPACK operations
    # requirements = ["blas-devel", 'pip:prox-tv']

    stopping_strategy = "callback"

    parameters = {'use_acceleration': [False, True]}

    def skip(self, A, reg, delta, data_fit, y, isotropy):
        if isotropy != "isotropic":
            return True, "ftvp supports only iso"
        return False, None

    def set_objective(self, A, reg, delta, data_fit, y, isotropy):
        self.reg, self.delta = reg, delta
        self.isotropy = isotropy
        self.data_fit = data_fit
        self.A, self.y = A, y

    def run(self, callback):
        n, m = self.y.shape
        stepsize = 1. / (get_l2norm(self.A) ** 2)
        u = np.zeros((n, m))
        u_acc = u.copy()
        u_old = u.copy()

        t_new = 1
        while callback(u):
            if self.use_acceleration:
                t_old = t_new
                t_new = (1 + np.sqrt(1 + 4 * t_old ** 2)) / 2
                u_old[:] = u
                u[:] = u_acc
            u = pyftvp.prox_tv( 255. *
                (u - stepsize * grad_F(self.y, self.A, u,
                                      self.data_fit, self.delta)) ,
                self.reg * stepsize * 255.)[0] / 255.0
            if self.use_acceleration:
                u_acc = u + (t_old - 1.) / t_new * (u - u_old)
        self.u = u

    def get_result(self):
        return self.u
