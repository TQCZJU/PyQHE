# %%
from typing import List
import numpy as np
from matplotlib import pyplot as plt

from pyqhe.equation.schrodinger import SchrodingerMatrix, SchrodingerSolver, SchrodingerShooting
from pyqhe.equation.poisson import PoissonSolver, PoissonODE, PoissonFDM
from pyqhe.utility.fermi import FermiStatistic
from pyqhe.core.structure import Structure2D, Layer

from pyqhe.submodules import basis, interaction, hamiltonian, pseudo, utils


loc_2deg = 30
loc_metal_plane = 65
# parameters for 2deg
length_b = 7.1
nu = 1 / 3
num_elec = 6
num_orbit = 18
radius = np.sqrt(2 * num_elec / nu)  # disk enclose exactly N/nu magnetic flux quanta
momentum = 65
# eff_thickness = 1
# %%
# construct 2d mesh grid for simulation
layer_list = []
layer_list.append(Layer(100, 0, 0, name='simulation_range'))
model2d = Structure2D(layer_list, width=80, delta=1, bound_neumann=[True, False])
grid = model2d.universal_grid
# add boundary condition
xv, yv = np.meshgrid(*grid, indexing='ij')
plate_length = (xv < radius * length_b)
metal_plate = (yv == loc_metal_plane) * plate_length
# bound = np.empty_like(xv)
# bound[:] = np.nan
# # assume the metal plane be grounded
# bound[metal_plate] = 0  # meV
# # model2d.add_dirichlet_boundary(bound)

# add Neumann boundary condition to rotation symmetry axis
# neumann_bound = np.empty_like(xv)
# neumann_bound[:] = np.nan
# neumann_bound[0] = 0  # set Neumann bound
# model2d.add_neumann_boundary(neumann_bound)
# %%
def calc_laughlin_density(background_pot=None, momentum=None):
    """Calculate the electron density of FQHE."""
    bvecs = basis.BasisVectors(num_elec, num_orbit, momentum)
    ccco = interaction.calculate_matrix_coeff(bvecs.number_orbitals,
                                              pseudo.pseudo_coulomb())
    ge, state = hamiltonian.calculate_ground_state(
        bvecs, ccco, background_potential=background_pot)
    # note the laughlin wave function represent the in-plate density
    return lambda z: state.electron_density(z / length_b)  # radial direction


# implement fqhtools for calculating 2deg charge density
screen_dist = abs(loc_2deg - loc_metal_plane) / length_b
# consider background potential
background_potential = utils.calculate_background_potential(
    num_orbit, num_elec, radius, screen_dist)
ll_density = calc_laughlin_density(background_pot=background_potential,
                                   momentum=momentum)
# convert 1d electron density to 2d electron density
charge_density = np.zeros_like(model2d.doping)
elecgas_plate = (yv == loc_2deg)
# %%
# compare sharp edge and Laughlin state
def calc_laughlin_state(background_pot=None, momentum=45):
    """Calculate the electron density of laughlin state."""
    bvecs = basis.BasisVectors(num_elec, num_orbit, momentum)
    ccco = interaction.calculate_matrix_coeff(bvecs.number_orbitals,
                                              pseudo.pseudo_v1())
    ge, state = hamiltonian.calculate_ground_state(
        bvecs, ccco, background_potential=background_pot)
    # note the laughlin wave function represent the in-plate density
    return lambda z: state.electron_density(z / length_b)  # radial direction


ll_state = calc_laughlin_state()
# %%
plt.plot(xv[elecgas_plate] / length_b, ll_state(xv[elecgas_plate]) * 2 * np.pi, label='Laughlin State')
plt.plot(xv[elecgas_plate] / length_b, ll_density(xv[elecgas_plate]) * 2 * np.pi, label='Sharp Edge')
plt.xlabel(r'$r / l_B$')
plt.ylabel(r'$2 \pi l^2_B  \rho(r)$')
plt.legend()
plt.title(f'd = {abs(loc_2deg - loc_metal_plane) / length_b:.2f}  M = 65')
plt.show()
# %%
charge_density[elecgas_plate] = -1.0 * ll_density(xv[elecgas_plate])
# accumulate charge density
accumulate_q = charge_density
for igrid in grid[::-1]:
    accumulate_q = np.trapezoid(accumulate_q, x=igrid)
# assume charge has uniform distribution in the metal
charge_density[metal_plate] = -accumulate_q / radius / length_b
# %%
# initialize Poisson solver
poi_solver = PoissonFDM(grid, charge_density, model2d.eps,
                        model2d.bound_dirichlet, model2d.bound_period, model2d.bound_neumann)
# calculate Poisson equation by charge density
potential = poi_solver.calc_poisson()
e_field = poi_solver.e_field
# %%
# data analysis
# load the initial charge distribution
plt.pcolormesh(xv, yv, potential)
plt.colorbar()
plt.xlabel('Axis r(nm)')
plt.ylabel('Axis h(nm)')
plt.show()
# %%
# add magnetic field, let the electrons form multi-body Laughlin wavefuntion
# calculate in-plane charge density by fraction quantum hall theory
# normalize in-plane density
plt.plot(np.arange(len(charge_density[:, loc_2deg])), charge_density[:, loc_2deg])
# %%
# pass the modulate charge density to poisson function
plt.pcolormesh(xv, yv, e_field[0])
plt.colorbar()
plt.xlabel('Axis r(nm)')
plt.ylabel('Axis h(nm)')
plt.title(r'$E_r$')
plt.show()
# %%
# pass the modulate charge density to poisson function
plt.pcolormesh(xv, yv, e_field[1])
plt.colorbar()
plt.xlabel('Axis r(nm)')
plt.ylabel('Axis h(nm)')
plt.title(r'$E_h$')
plt.show()
# %%
plt.pcolormesh(xv, yv, np.sqrt(e_field[0]**2 + e_field[1]**2))
plt.colorbar()
plt.xlabel('Axis r(nm)')
plt.ylabel('Axis h(nm)')
plt.title(r'$\sqrt{E_r^2 + E_h^2}$')
plt.show()

# %%
plt.plot(e_field[0][0])

# %%
