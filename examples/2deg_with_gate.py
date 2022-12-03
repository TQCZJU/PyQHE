# %%
from typing import List
import numpy as np
from matplotlib import pyplot as plt

from pyqhe.equation.schrodinger import SchrodingerMatrix, SchrodingerSolver, SchrodingerShooting
from pyqhe.equation.poisson import PoissonSolver, PoissonODE, PoissonFDM
from pyqhe.utility.fermi import FermiStatistic
from pyqhe.core.structure import Structure2D, Layer

from pyqhe.submodules import basis, interaction, hamiltonian, pseudo, utils

length_b = 7.1
loc_metal_plane = 70
loc_2deg = 30
# parameters for 2deg
num_elec = 6
num_orbit = 16
momentum = 45
eff_thickness = 5
# %%
# construct 2d mesh grid for simulation
layer_list = []
layer_list.append(Layer(100, 0, 0, name='simulation_range'))
model2d = Structure2D(layer_list, width=80, delta=1)
grid = model2d.universal_grid
# add boundary condition
xv, yv = np.meshgrid(*grid, indexing='ij')
plate_length = (xv < 70) * (xv > 10)
metal_plate = (yv <= loc_metal_plane + 1) * (yv >= loc_metal_plane - 1)
bound = np.empty_like(xv)
bound[:] = np.nan
# assume the metal plane be grounded
bound[metal_plate * plate_length] = 5  # meV
model2d.add_dirichlet_boundary(bound)


# %%
def calc_laughlin_density(background_pot=None, momentum=None):
    """Calculate the electron density of laughlin state."""
    bvecs = basis.BasisVectors(num_elec, num_orbit, momentum)
    ccco = interaction.calculate_matrix_coeff(bvecs.number_orbitals,
                                              pseudo.pseudo_coulomb())
    ge, state = hamiltonian.calculate_ground_state(
        bvecs, ccco, background_potential=background_pot)
    # note the laughlin wave function represent the in-plate density
    return lambda z: state.electron_density(z / length_b)  # radial direction


# implement fqhtools for calculating 2deg charge density
screen_dist = abs(loc_2deg - loc_metal_plane) / length_b
radius = np.sqrt(2 * num_orbit)
# consider background potential
background_potential = utils.calculate_background_potential(
    num_orbit, num_elec, radius, screen_dist)
ll_density = calc_laughlin_density(background_pot=background_potential,
                                   momentum=momentum)
# convert 1d electron density to 2d electron density
charge_density = np.zeros_like(model2d.doping)
elecgas_plate = (yv <= loc_2deg + eff_thickness / 2) * (
    yv >= loc_2deg - eff_thickness / 2)
# %%
charge_density[elecgas_plate] = -1.0 * ll_density(xv[elecgas_plate] -
                                                  grid[0][-1] / 2)
# %%
# initialize Poisson solver
poi_solver = PoissonFDM(grid, charge_density, model2d.eps,
                        model2d.bound_dirichlet, model2d.bound_period)
# calculate Poisson equation by charge density
potential = poi_solver.calc_poisson()
e_field = poi_solver.e_field
# %%
# data analysis
# load the initial charge distribution
plt.pcolormesh(xv, yv, potential)
plt.colorbar()
plt.xlabel('Axis X(nm)')
plt.ylabel('Axis Z(nm)')
plt.show()
# %%
# add magnetic field, let the electrons form multi-body Laughlin wavefuntion
# calculate in-plane charge density by fraction quantum hall theory
# normalize in-plane density
plt.plot(np.arange(len(charge_density[:, 30])), charge_density[:, 30])
# %%
# pass the modulate charge density to poisson function
plt.pcolormesh(xv, yv, e_field[0])
plt.colorbar()
plt.xlabel('Axis X(nm)')
plt.ylabel('Axis Z(nm)')
plt.show()
# %%
# pass the modulate charge density to poisson function
plt.pcolormesh(xv, yv, e_field[1])
plt.colorbar()
plt.xlabel('Axis X(nm)')
plt.ylabel('Axis Z(nm)')
plt.show()
# %%
plt.pcolormesh(xv, yv, np.sqrt(e_field[0]**2 + e_field[1]**2))
plt.colorbar()
plt.xlabel('Axis X(nm)')
plt.ylabel('Axis Z(nm)')
plt.show()

# %%
