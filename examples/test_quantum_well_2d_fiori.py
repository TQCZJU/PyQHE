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
from pyqhe.utility.constant import const


def calc_omega(thickness, gate_voltage, tol=2e-3):
    layer_list = []
    layer_list.append(Layer(20, 0.36, 0.0, name='barrier'))
    layer_list.append(Layer(2, 0.36, 5e17, name='n-type'))
    layer_list.append(Layer(5, 0.36, 0.0, name='spacer'))
    layer_list.append(Layer(12, 0, 0, name='quantum_well'))
    layer_list.append(Layer(25, 0.36, 0.0, name='spacer'))
    layer_list.append(Layer(thickness, 0, 0, name='quantum_well'))
    # layer_list.append(Layer(10, 0.36, 0.0, name='spacer'))
    # layer_list.append(Layer(10, 0, 0, name='quantum_well'))
    layer_list.append(Layer(5, 0.36, 0.0, name='spacer'))
    # layer_list.append(Layer(2, 0.36, 5e17, name='n-type'))
    layer_list.append(Layer(30, 0.36, 0.0, name='barrier'))
    width = 100
    model = Structure2D(
        layer_list,
        width,
        temp=10,
        delta=1,
        bound_neumann=[[True, True], [False, False]],
        # bound_period=[True, False]
    )
    # add boundary condition
    grid = model.universal_grid
    delta = grid[0][1] - grid[0][0]
    xv, yv = np.meshgrid(*grid, indexing='ij')
    plate_length = (xv > width - 10) + (xv < 10)
    top_plate = (yv <= 15) * (yv >= 10)
    bottom_plate = (yv <= 65) * (yv >= 60)
    bound = np.empty_like(xv)
    bound[:] = np.nan
    bound[top_plate * plate_length] = gate_voltage  # meV
    # bound[top_plate * plate_length] = gate_voltage  # meV
    # bound[bottom_plate] = 0
    model.add_dirichlet_boundary(bound)
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
                           res.sigma * 1e21,
                           cmap=cm.coolwarm,
                           linewidth=0,
                           antialiased=False)
    ax.set_xlabel('Axis X(nm)')
    ax.set_ylabel('Axis Z(nm)')
    ax.set_zlabel(r'Charge density $cm^{-3}$')
    ax.zaxis.labelpad = -3
    plt.show()

    return res


# %%
res = calc_omega(30, 0.02)
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
# calculate sheet density
screen_mask = (yv < 45)
screen_mask_index = (res.grid[-1] < 45)
sheet_density = np.trapezoid(
    res.electron_density[screen_mask].reshape(res.dim[:-1] + [
        -1,
    ]),
    res.grid[-1][screen_mask_index],
    axis=-1)
plt.plot(res.grid[0], sheet_density * 1e14)  # unit in cm^-2
# %%
shape = np.array([dim / 2 for dim in res.sigma.shape], dtype=int)
plt.plot(res.grid[1], res.sigma[shape[0]] * 1e21)
plt.show()
plt.plot(res.grid[0], res.sigma[:, shape[1]] * 1e21)
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
# %%
# Create the figure and the first y-axis
fig, ax1 = plt.subplots()

# Plot the first dataset using the first y-axis
ax1.plot(res.grid[1], res.v_potential[5], 'orange', label='Gate')
ax1.plot(res.grid[1], res.v_potential[shape[0]], 'r', label='QPC')
ax1.legend()
ax1.set_xlabel('z (nm)')
ax1.set_ylabel('Gamma band edge (eV)', color='r')

# Create the second y-axis
ax2 = ax1.twinx()

# Plot the second dataset using the second y-axis
ax2.plot(res.grid[1], res.electron_density[5] * 1e21, label='Gate')
ax2.plot(res.grid[1], res.electron_density[shape[0]] * 1e21, 'b', label='QPC')
ax2.legend()
ax2.set_ylabel(r'Electron density ($cm^{-3}$)', color='b')
# %%
e_field = np.sqrt(res.e_field[0]**2 + res.e_field[1]**2)
plt.pcolormesh(xv, yv, e_field)
plt.colorbar()
plt.xlabel('Axis X(nm)')
plt.ylabel('Axis Z(nm)')
plt.show()
# %%
plt.pcolormesh(xv, yv, res.repulsive)
plt.colorbar()
plt.xlabel('Axis X(nm)')
plt.ylabel('Axis Z(nm)')
plt.title('Potential')
# calculate sheet density
screen_mask = (yv < 45)
screen_mask_index = (res.grid[-1] < 45)
sheet_density = np.trapezoid(
    res.electron_density[screen_mask].reshape(res.dim[:-1] + [
        -1,
    ]),
    res.grid[-1][screen_mask_index],
    axis=-1)
plt.plot(res.grid[0], sheet_density * 1e5, 'r')  # unit in cm^-2
plt.show()
# %%
plt.plot(res.grid[1], res.repulsive[shape[0]])
plt.show()
# %%
plt.pcolormesh(xv, yv, res.v_potential)
plt.colorbar()
plt.xlabel('Axis X(nm)')
plt.ylabel('Axis Z(nm)')
plt.show()
# %%
plt.plot(res.grid[0], res.v_potential[:, 12])
plt.show()
# %%
