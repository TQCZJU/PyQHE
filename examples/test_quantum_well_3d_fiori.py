# %%
import numpy as np
from scipy import optimize
from scipy.stats import norm
from matplotlib import pyplot as plt
from matplotlib import cm

from pyqhe.core.structure import Layer, Structure3D
from pyqhe.schrodinger_poisson import SchrodingerPoisson
from pyqhe.equation.poisson import PoissonFDM, PoissonODE
from pyqhe.equation.schrodinger import SchrodingerMatrix, SchrodingerFiori
from pyqhe.fiori_poisson import FioriPoisson


def calc_omega(thickness=10, tol=5e-5):
    layer_list = []
    layer_list.append(Layer(20, 0.24, 0.0, name='barrier'))
    layer_list.append(Layer(2, 0.24, 5e17, name='n-type'))
    layer_list.append(Layer(5, 0.24, 0.0, name='spacer'))
    layer_list.append(Layer(thickness, 0, 0, name='quantum_well'))
    layer_list.append(Layer(5, 0.24, 0.0, name='spacer'))
    layer_list.append(Layer(2, 0.24, 5e17, name='n-type'))
    layer_list.append(Layer(20, 0.24, 0.0, name='barrier'))

    model = Structure3D(layer_list,
                        length=10,
                        width=10,
                        temp=10,
                        delta=1,
                        bound_neumann=[True, True, False])
    # instance of class SchrodingerPoisson
    schpois = FioriPoisson(
        model,
        schsolver=SchrodingerFiori,
        poisolver=PoissonFDM,
        # quantum_region=(255 - 20, 255 + thickness + 30),
    )
    # test = schpois.sch_solver.calc_evals()
    # perform self consistent optimization
    res, loss = schpois.self_consistent_minimize(tol=tol)
    if loss > tol:
        res, loss = schpois.self_consistent_minimize(tol=tol)
    # plot 2DES areal electron density
    yv, zv = np.meshgrid(*schpois.grid[1:], indexing='ij')
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(yv,
                           zv,
                           res.sigma[0] * thickness * 1e14,
                           cmap=cm.coolwarm,
                           linewidth=0,
                           antialiased=False)
    plt.show()

    return res


# %%
res = calc_omega(20)
# res.plot_quantum_well()
# %%
yv, zv = np.meshgrid(*res.grid[1:], indexing='ij')
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(yv,
                       zv,
                       res.electron_density[:, 0, :],
                       cmap=cm.coolwarm,
                       linewidth=0,
                       antialiased=False)
# ax.view_init(0, 90)
# %%
shape = np.array([dim / 2 for dim in res.sigma.shape], dtype=int)
plt.plot(res.grid[-1], res.sigma[shape[0], shape[1], :] * 20 * 1e14)
plt.show()
plt.plot(res.grid[0], res.sigma[:, :, shape[2]] * 20 * 1e14)
plt.show()
xv, yv = np.meshgrid(*res.grid[:-1], indexing='ij')
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(xv,
                       yv,
                       res.sigma[:, :, shape[2]] * 20 * 1e14,
                       cmap=cm.coolwarm,
                       linewidth=0,
                       antialiased=False)
# %%
