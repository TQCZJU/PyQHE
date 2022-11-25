# %%
from typing import List
import numpy as np
from matplotlib import pyplot as plt

from pyqhe.equation.schrodinger import SchrodingerMatrix, SchrodingerSolver, SchrodingerShooting
from pyqhe.equation.poisson import PoissonSolver, PoissonODE, PoissonFDM
from pyqhe.utility.fermi import FermiStatistic
from pyqhe.core.structure import Structure2D

from pyqhe.submodules import basis, interaction, hamiltonian, pseudo, utils

length_b = 7.1
# %%
class OptimizeResult:
    """Optimize result about self-consistent iteration."""

    def __init__(self) -> None:
        # Storage of grid configure
        self.grid = None
        # Optimizer result
        self.params = None
        # Fermi Statistic
        self.fermi_energy = None
        self.n_states = None
        self.sigma = None
        # electron properties
        self.eig_val = None
        self.wave_function = None
        # electric field properties
        self.v_potential = None
        self.e_field = None
        # Accumulate electron density
        self.electron_density = None
        self.laughlin_density = None

    def plot_quantum_well(self):
        """Plot dressed conduction band of quantum well, and electrons'
        eigenenergy and wave function.
        """

        wave_func_rescale = 0.2
        ax = plt.subplot(1, 1, 1)
        ax.plot(self.grid[0], self.v_potential, "k")
        # just plot the three lowest eigenenergy
        colors = ['y', 'c', 'm']
        for i, (energy, state) in enumerate(
                zip(self.eig_val[:3], self.wave_function[:3])):
            ax.axhline(energy,
                       0.1,
                       0.9,
                       ls="--",
                       color=colors[i],
                       label=f'E_{i}: {energy:3f}')  # eigenenergy
            # plot rescaled wave function
            ax.plot(self.grid[0], state * wave_func_rescale + energy, color='b')
        ax.axhline(self.fermi_energy,
                   0.1,
                   0.9,
                   color="r",
                   ls="--",
                   label=f'E_fermi: {float(self.fermi_energy):3f}')
        ax.set_xlabel("Position (nm)")
        ax.set_ylabel("Energy (eV)")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        ax.grid(True)

        return ax


class LaughlinPoisson:
    """Self-consistent Laughlin electron wave_function - Poisson solver
    Args:
        sch_solver: Schrodinger equation solver.
        poi_solver: Poisson equation solver.
        fermi_util: Use Fermi statistic for Fermi level and energy bands' distribution.
    """

    def __init__(self,
                 num_elec,
                 num_orbit,
                 screen_dist,
                 length_b,
                 model: Structure2D,
                 schsolver: SchrodingerSolver = SchrodingerMatrix,
                 poisolver: PoissonSolver = PoissonFDM,
                 learning_rate=0.5,
                 quantum_region: List[float] = None,
                 **kwargs) -> None:
        self.model = model
        # load material's properties
        self.temp = model.temp  # temperature
        self.fi = model.fi  # Band structure's potential
        self.cb_meff = model.cb_meff  # Conduction band effective mass
        self.eps = model.eps  # dielectric constant(multiply eps_0)
        self.doping = model.doping  # doping profile
        # load fraction quantum hall parameters
        self.num_elec = num_elec
        self.num_orbit = num_orbit
        self.length_b = length_b
        # load grid configure
        self.grid = model.universal_grid
        # load boundary condition
        self.bound_dirichlet = model.bound_dirichlet
        # Load period boundary condition
        self.bound_period = model.bound_period
        # Setup Quantum region
        # if quantum_region is not None and len(quantum_region) == 2:
        #     self.quantum_mask = (self.grid > quantum_region[0]) * (
        #         self.grid < quantum_region[1])
        # else:
        #     self.quantum_mask = (np.ones_like(self.grid) == 1)
        # adjust optimizer
        self.learning_rate = learning_rate
        # load solver
        self.sch_solver: SchrodingerSolver = schsolver(self.grid,
                                    self.fi,
                                    self.cb_meff,
                                    self.bound_period)
        self.fermi_util = FermiStatistic(self.grid,
                                         self.cb_meff,
                                         self.doping)
        self.poi_solver: PoissonSolver = poisolver(self.grid, self.doping, self.eps,
                                    self.bound_dirichlet,
                                    self.bound_period)
        # accumulate charge density
        self.accumulate_q = self.doping
        for grid in self.grid[::-1]:
            self.accumulate_q = np.trapz(self.accumulate_q, x=grid)
        # distance to the screening metal
        self.screen_dist = screen_dist / self.length_b
        # Cache parameters
        self.sch_solver.v_potential = self.fi
        self.eig_val = self.sch_solver.calc_evals()
        self.params = None
        self.bg_pot = None
        # initialize charge distribution without magnetic
        # assume background potential is static
        radius = np.sqrt(2 * self.num_orbit)
        self.bg_pot = utils.calculate_background_potential(self.num_orbit, self.num_elec, radius, self.screen_dist)
        # laughlin wavefunction
        self.ll_density = self._calc_laughlin_density(self.bg_pot, 45)

    def _calc_laughlin_density(self, background_pot, momentum=None):
        """Calculate the electron density of laughlin state."""
        bvecs = basis.BasisVectors(self.num_elec, self.num_orbit, momentum)
        ccco = interaction.calculate_matrix_coeff(bvecs.number_orbitals,
                                                  pseudo.pseudo_coulomb())
        ge, state = hamiltonian.calculate_ground_state(
            bvecs, ccco, background_potential=background_pot)
        # note the laughlin wave function represent the in-plate density
        return state.electron_density((self.grid[0] - self.grid[0][-1] / 2) / self.length_b)  # radial direction

    def _calc_net_density(self, n_states, wave_func, mode='modulate'):
        """Calculate the net charge density."""

        elec_density = np.zeros_like(self.doping)
        for i, distri in enumerate(n_states):
            elec_density += distri * wave_func[i] * np.conj(wave_func[i])
        if mode == 'modulate':
            # modulation 2d single electron wave function
            # use numpy broadcast
            elec_density = elec_density * self.ll_density[:, np.newaxis]
        else:
            # Separation of variable method, just multiply in-plane and out-plane wavefunction
            # cut the center array of out-plane density
            outplane = elec_density[int(len(elec_density) / 2)]
            elec_density = np.kron(self.ll_density, outplane.reshape(-1, 1))
            # note here the method loss information about edges
        # normalize by electric neutrality
        accumu_elec = elec_density.copy()
        for grid in self.grid[::-1]:
            accumu_elec = np.trapz(accumu_elec, x=grid)
        norm = self.accumulate_q / accumu_elec
        elec_density *= norm
        # Let dopants density minus electron density
        self.test_density = elec_density
        net_density = self.doping - elec_density

        return net_density

    def _iteration(self, params):
        """Perform a single iteration of self-consistent Schrodinger-Poisson
        calculation.

        Args:
            v_potential: optimizer parameters.
        """

        v_potential = self.fi + params
        self.sch_solver.v_potential = v_potential
        eig_val, wave_func = self.sch_solver.calc_esys()
        # calculate energy band distribution
        _, n_states = self.fermi_util.fermilevel(eig_val, wave_func, self.temp)
        # calculate the net charge density
        sigma = self._calc_net_density(n_states, wave_func)
        # TODO: implement self-consistence background potential solver
        # Maybe Green function? FDM?
        # perform poisson solver
        self.poi_solver.charge_density = sigma
        params = self.poi_solver.calc_poisson()
        # return eigenenergy loss
        loss = np.abs(self.eig_val[0] - eig_val[0])
        self.eig_val = eig_val

        return loss, params

    def self_consistent_minimize(self,
                                 num_iter=10,
                                 learning_rate=0.5,
                                 tol=1e-5,
                                 logging=True):
        """Self consistent optimize parameters `v_potential` to get solution.

        Args:
            learning_rate: learning rate between adjacent iteration.
        """
        if self.params is None:
            self.params = 0  # v_potential
        for i, _ in enumerate(range(num_iter)):
            # perform a iteration
            loss, temp_params = self._iteration(self.params)
            if logging:
                print(
                    f'Loss: {loss}, energy_0: {self.eig_val[0]}, '
                    f'energy_1: {self.eig_val[1]}, energy_2: {self.eig_val[2]}')
            # self-consistent update params
            self.params += (temp_params - self.params) * learning_rate
            if i and loss < tol:
                break
        # save optimize result
        # optimal_index = np.argmin(loss_list[1:])
        # self.params = param_list[optimal_index]
        res = OptimizeResult()
        res.params = self.params
        res.grid = self.grid
        res.v_potential = self.params + self.fi
        # reclaim convergence result
        self.sch_solver.v_potential = res.v_potential
        res.eig_val, res.wave_function = self.sch_solver.calc_esys()
        res.fermi_energy, res.n_states = self.fermi_util.fermilevel(
            res.eig_val, res.wave_function, self.temp)
        res.sigma = self._calc_net_density(res.n_states, res.wave_function)
        res.laughlin_density = self.ll_density
        # noted the 2d charge density is rho tensors sigma.T
        res.electron_density = self.doping - res.sigma
        # full wave function
        full_wave_function = []
        for wf in res.wave_function:
            new_wf = wf
            full_wave_function.append(new_wf)
        res.wave_function = np.asarray(full_wave_function)
        self.poi_solver.charge_density = res.electron_density
        self.poi_solver.calc_poisson()
        res.e_field = self.poi_solver.e_field
        # # Accumulate electron areal density in the subbands
        # res.electron_density = np.zeros_like(self.doping)
        # for i, distri in enumerate(res.n_states):
        #     res.electron_density += distri * res.wave_function[i] * np.conj(
        #         res.wave_function[i])

        return res, loss

    def test(self):
        self.params = 0  # coulomb potential
        eig_val, wave_func = self.sch_solver.calc_esys()
        self.temp_wf = wave_func
        # calculate energy band distribution
        _, n_states = self.fermi_util.fermilevel(eig_val, wave_func, self.temp)
        # calculate the out-plane charge density
        self.elec_density = np.zeros_like(self.doping)
        for i, distri in enumerate(n_states):
            self.elec_density += distri * wave_func[i] * np.conj(wave_func[i])
        # Separation of variable method, just multiply in-plane and out-plane wavefunction
        # cut the center array of out-plane density

        # loss, temp_params = self._iteration(self.params)
# %%
from pyqhe.core import Layer


layer_list = []
layer_list.append(Layer(20, 0.24, 0.0, name='barrier'))  # insert background screening plane in the middle
layer_list.append(Layer(2, 0.24, 5e17, name='n-type'))
layer_list.append(Layer(5, 0.24, 0.0, name='spacer'))
layer_list.append(Layer(20, 0, 0, name='quantum_well'))
layer_list.append(Layer(5, 0.24, 0.0, name='spacer'))
layer_list.append(Layer(2, 0.24, 5e17, name='n-type'))
layer_list.append(Layer(20, 0.24, 0.0, name='barrier'))

dist = 27  # electron locate at the center of wall
model2d = Structure2D(layer_list, width=100, temp=10, delta=1, bound_period=[True, False])
# add boundary condition
grid = model2d.universal_grid
delta = grid[0][1] - grid[0][0]
xv, yv = np.meshgrid(*grid, indexing='ij')
plate_length = (xv < 90) * (xv > 10)
top_plate = (yv <= 10 + 1) * (yv >= 10 - 1)
bound = np.empty_like(xv)
bound[:] = np.nan
bound[top_plate * plate_length] = 0  # meV
model2d.add_dirichlet_boundary(bound)
lp = LaughlinPoisson(num_elec=6, num_orbit=18, length_b=length_b, model=model2d, screen_dist=dist)
# %%
loss, charge_pot = lp._iteration(0)
# %%
lp.test()
# load the initial charge distribution
plt.pcolormesh(xv, yv, lp.temp_wf[0] * np.conj(lp.temp_wf[0]))
plt.colorbar()
plt.xlabel('Axis X(nm)')
plt.ylabel('Axis Z(nm)')
plt.show()
# %%
# add magnetic field, let the electrons form multi-body Laughlin wavefuntion
# calculate in-plane charge density by fraction quantum hall theory
# normalize in-plane density
plt.plot(lp.grid[0], lp.ll_density * 2 * np.pi)
# %%
# Separation of variable method, just multiply in-plane and out-plane wavefunction
# mode 'separate_var'
outplane = lp.elec_density[int(len(lp.elec_density) / 2)]
plt.plot(lp.grid[1], outplane)
# %%
charge_density = np.kron(lp.ll_density, outplane.reshape(-1, 1))
plt.pcolormesh(xv, yv, charge_density.T)
plt.colorbar()
plt.xlabel('Axis z(nm)')
plt.ylabel('Axis w(nm)')
plt.show()
# %%
# modulation 2d single electron wave function
# mode 'modulate'
plt.pcolormesh(xv, yv, lp.test_density)
plt.colorbar()
plt.xlabel('Axis z(nm)')
plt.ylabel('Axis w(nm)')
plt.show()
# %%
plt.pcolormesh(xv, yv, lp.elec_density)
plt.colorbar()
plt.xlabel('Axis Z(nm)')
plt.ylabel('Axis W(nm)')
plt.show()
# %%
# pass the modulate charge density to poisson function
plt.pcolormesh(xv, yv, charge_pot)
plt.colorbar()
plt.xlabel('Axis X(nm)')
plt.ylabel('Axis Z(nm)')
plt.show()
# %%
# pass the modulate charge density to poisson function
plt.pcolormesh(xv, yv, lp.sigma)
plt.colorbar()
plt.xlabel('Axis X(nm)')
plt.ylabel('Axis Z(nm)')
plt.show()
# %%
res, loss = lp.self_consistent_minimize()
# %%
