# %%
import numpy as np
from matplotlib import pyplot as plt

from pyqhe.equation.poisson import PoissonFDMCircular
from pyqhe.utility.constant import const

const.to_natural_unit()
# %%
# initialize grid and input arguments
grid = [np.arange(0, 101), np.arange(-50, 51)]
dim = [grid_axis.shape[0] for grid_axis in grid]
xv, yv = np.meshgrid(*grid, indexing='ij')

eps = np.ones(dim) * const.eps0
charge_density = np.zeros_like(eps)


def run_simulation():
    # add boundary condition
    plate_length = (xv <= radius)
    upper_plate = (yv == loc_upper_plate) * plate_length
    lower_plate = (yv == loc_lower_plate) * plate_length
    bound_dirichlet = np.empty_like(xv, dtype=float)
    bound_dirichlet[:] = np.nan
    # set the potential of plate
    bound_dirichlet[upper_plate] = v0
    bound_dirichlet[lower_plate] = -v0
    # initialize Poisson solver
    poi_solver = PoissonFDMCircular(grid,
                                    charge_density,
                                    eps,
                                    bound_dirichlet,
                                    bound_period=[None, None],
                                    bound_neumann=[[True, True], [False, False]])
    # calculate Poisson equation by charge density
    potential = poi_solver.calc_poisson()
    e_field = poi_solver.e_field
    return potential, e_field


# %%
# construct the circular parallel plate capacitor
radius = 14
loc_upper_plate = 18
loc_lower_plate = -18
v0 = 1

potential, e_field = run_simulation()
# data analysis
# plot potential
plt.pcolormesh(xv, yv, potential)
plt.colorbar()
plt.xlabel('Axis r(nm)')
plt.ylabel('Axis h(nm)')
plt.show()
# cut one quadrant
quad_xv = xv[:, :50]
quad_yv = yv[:, :50]
quad_potential = potential[:, :50]
# plot the contour of potential
contour_list = [-1.0, -0.8, -0.6, -0.4, -0.2, -0.1, 0]
plt.contour(quad_xv, quad_yv, quad_potential, contour_list)
plt.xlabel('Axis r(nm)')
plt.ylabel('Axis h(nm)')
plt.colorbar()
plt.show()
# %%
# construct the circular parallel plate capacitor
radius = 24
v0 = 1
dist_list = [4, 12, 36]
res = []
for dist in dist_list:
    loc_upper_plate = dist
    loc_lower_plate = -dist
    res.append(run_simulation())
# %%
# data analysis
# plot E against r in the median plane
for kappa, (_, e_field) in zip(dist_list, res):
    plt.plot(e_field[1][yv == 0] * kappa * -1.0 / v0,
             label=r'$\kappa$=' + f'{kappa / 12:.3f}')
plt.ylabel(r'$E/2V_0\kappa^{-1}$')
plt.xlabel('Axis r(nm)')
plt.title(r'$E_z$')
plt.legend()
plt.show()
# %%
