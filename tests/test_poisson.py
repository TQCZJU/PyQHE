import numpy as np
from matplotlib import pyplot as plt

from pyqhe.equation.poisson import PoissonFDM, PoissonODE


def test_poisson1d_ode_vs_fdm():

    grid = np.linspace(0, 1, 80)
    eps = np.ones(grid.shape)
    sigma = np.zeros(grid.shape)
    sigma[20:31] = 1
    sigma[40:51] = -1
    sigma[50:61] = 1
    sol = PoissonFDM(grid, sigma, eps, bound_neumann=[[True, False]])
    sol.calc_poisson()

    sol2 = PoissonODE(grid, sigma, eps)
    sol2.calc_poisson()
    assert np.allclose(sol.v_potential - sol.v_potential[0], sol2.v_potential, atol=1e-4)
    assert np.allclose(sol.e_field, sol2.e_field)
