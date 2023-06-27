# %%
import numpy as np
from scipy import optimize
from scipy.stats import norm
from matplotlib import pyplot as plt
from matplotlib import cm

from pyqhe.core.structure import Layer, Structure2D, Structure3D
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

    model = Structure2D(layer_list,
                        width=50,
                        temp=10,
                        delta=1,
                        bound_neumann=[[True, True], [False, False]],
                        bound_period=None)
    # add boundary condition
    grid = model.universal_grid
    delta = grid[0][1] - grid[0][0]
    xv, yv = np.meshgrid(*grid, indexing='ij')
    plate_length = (xv < 35) * (xv > 15)
    top_plate = (yv <= 15) * (yv >= 10)
    bottom_plate = (yv <= 65) * (yv >= 60)
    bound = np.empty_like(xv)
    bound[:] = np.nan
    bound[top_plate * plate_length] = -0.02  # meV
    # bound[bottom_plate] = 0
    # model.add_dirichlet_boundary(bound)
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
    xv, yv = np.meshgrid(*schpois.grid, indexing='ij')
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(xv,
                           yv,
                           res.sigma * thickness * 1e14,
                           cmap=cm.coolwarm,
                           linewidth=0,
                           antialiased=False)
    plt.show()

    return res


# %%
res = calc_omega(20)
# res.plot_quantum_well()
# %%
xv, yv = np.meshgrid(*res.grid, indexing='ij')
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(xv,
                       yv,
                       res.electron_density,
                       cmap=cm.coolwarm,
                       linewidth=0,
                       antialiased=False)
plt.show()
# ax.view_init(0, 90)
# %%
shape = np.array([dim / 2 for dim in res.sigma.shape], dtype=int)
plt.plot(res.grid[1], res.sigma[shape[0]] * 1e21)
plt.show()
plt.plot(res.grid[0], res.sigma[:, 30] * 1e21)
plt.show()
# %%
xv, yv = np.meshgrid(*res.grid, indexing='ij')
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(xv,
                       yv,
                       res.v_potential,
                       cmap=cm.coolwarm,
                       linewidth=0,
                       antialiased=False)
plt.show()
plt.plot(res.grid[1], res.v_potential[shape[0]])
# %%
e_field = np.sqrt(res.e_field[0]**2 + res.e_field[1]**2)
plt.pcolormesh(xv, yv, e_field)
plt.colorbar()
plt.xlabel('Axis X(nm)')
plt.ylabel('Axis Z(nm)')
plt.show()
# %%
