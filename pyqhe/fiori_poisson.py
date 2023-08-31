"""The basic Fiori's method Poisson-Schrodinger iteration solver.
"""
# %%
from typing import List
import numpy as np
from matplotlib import pyplot as plt

from pyqhe.equation.schrodinger import SchrodingerMatrix, SchrodingerSolver, SchrodingerShooting, SchrodingerFiori
from pyqhe.equation.poisson import PoissonSolver, PoissonODE, PoissonFDM
from pyqhe.utility.fermi import FermiStatistic
from pyqhe.core.structure import Structure2D
from pyqhe.utility.constant import const


class OptimizeResult:
    """Optimize result about self-consistent iteration."""

    def __init__(self) -> None:
        # Storage of grid configure
        self.grid = None
        self.dim = None
        # Optimizer result
        self.repulsive = None
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


class FioriPoisson:
    """Self-consistent Schrodinger Poisson solver
    Args:
        sch_solver: Schrodinger equation solver.
        poi_solver: Poisson equation solver.
        fermi_util: Use Fermi statistic for Fermi level and energy bands' distribution.
    """

    def __init__(self,
                 model: Structure2D,
                 schsolver: SchrodingerSolver = SchrodingerFiori,
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
        # load grid configure
        self.grid = model.universal_grid
        self.dim = [grid_axis.shape[0] for grid_axis in self.grid]
        # load boundary condition
        self.bound_dirichlet = model.bound_dirichlet
        self.bound_neumann = model.bound_neumann
        # Load period boundary condition
        self.bound_period = model.bound_period
        # Load rotational symmetry
        self.rotational_symmetry = model.rotational_symmetry
        # Setup Quantum region
        # if quantum_region is not None and len(quantum_region) == 2:
        #     self.quantum_mask = (self.grid > quantum_region[0]) * (
        #         self.grid < quantum_region[1])
        # else:
        #     self.quantum_mask = (np.ones_like(self.grid) == 1)
        # adjust optimizer
        self.learning_rate = learning_rate
        # load solver
        self.sch_solver = schsolver(self.grid,
                                    self.fi,
                                    self.cb_meff,
                                    self.bound_period,
                                    quantum_region)
        self.fermi_util = FermiStatistic(self.grid[-1],
                                         self.cb_meff.reshape(-1, len(self.grid[-1]))[-1],
                                         self.doping.reshape(-1, len(self.grid[-1]))[-1])
        self.poi_solver = poisolver(self.grid, self.doping, self.eps,
                                    self.bound_dirichlet,
                                    self.bound_period,
                                    self.bound_neumann,
                                    self.rotational_symmetry)
        # accumulate charge density
        self.accumulate_q = self.doping
        for grid in self.grid[::-1]:
            self.accumulate_q = np.trapz(self.accumulate_q, x=grid)
        # Cache parameters
        self.eig_val = self.sch_solver.calc_evals()
        self.repulsive = None

    def _calc_net_density(self, n_states, wave_func):
        """Calculate the net charge density."""

        elec_density = np.zeros_like(self.doping)
        for i, distri in enumerate(n_states):
            elec_density += distri * wave_func[i] * np.conj(wave_func[i])
        # normalize by electric neutrality
        accumu_elec = elec_density.copy()
        for grid in self.grid[::-1]:
            accumu_elec = np.trapz(accumu_elec, x=grid)
        norm = self.accumulate_q / accumu_elec
        elec_density *= norm
        # noted we consider a electron in the conductive band, and high electron
        # density are repulsive to an electron.
        net_density = elec_density - self.doping
        # the sign will be conflicted with general potential definition due to self.fi

        return net_density

    def _iteration(self, repulsive):
        """Perform a single iteration of self-consistent Schrodinger-Poisson
        calculation.

        Args:
            repulsive: the repulsive effect will be optimized in the iteration
        """

        # perform schrodinger solver
        v_potential = self.fi + repulsive
        self.sch_solver.v_potential = v_potential
        eig_val, wave_func = self.sch_solver.calc_esys()
        # calculate energy band distribution
        if wave_func.ndim == 3:
            _, n_states = self.fermi_util.fermilevel(eig_val[:, 0], wave_func[:, 0, :], self.temp)
        elif wave_func.ndim == 4:
            _, n_states = self.fermi_util.fermilevel(eig_val[:, 0, 0], wave_func[:, 0, 0, :], self.temp)
        else:
            raise ValueError('Unsupported wave function.')
        # calculate the net charge density
        sigma = self._calc_net_density(n_states, wave_func)
        #TODO: we need specific net density calculator in different regime(IQHE, FQHE)
        # perform poisson solver
        self.poi_solver.charge_density = sigma * const.e
        repulsive = self.poi_solver.calc_poisson()
        # return eigenenergy loss
        loss = np.mean(np.abs(self.eig_val[0] - eig_val[0]))
        self.eig_val = eig_val

        return loss, repulsive

    def self_consistent_minimize(self,
                                 num_iter=10,
                                 learning_rate=0.5,
                                 tol=1e-5,
                                 logging=True):
        """Self consistent optimize parameters `v_potential` to get solution.

        Args:
            learning_rate: learning rate between adjacent iteration.
        """
        if self.repulsive is None:
            self.repulsive = 0  # v_potential
        for i, _ in enumerate(range(num_iter)):
            # perform a iteration
            loss, temp_repulsive = self._iteration(self.repulsive)
            if logging:
                print(
                    f'Loss: {loss}, energy_0: {self.eig_val[0]}, '
                    f'energy_1: {self.eig_val[1]}, energy_2: {self.eig_val[2]}')
            # self-consistent update repulsive
            self.repulsive += (temp_repulsive - self.repulsive) * learning_rate
            if i and loss < tol:
                print('Self-Consistent!')
                break
        # save optimize result
        # optimal_index = np.argmin(loss_list[1:])
        # self.repulsive = param_list[optimal_index]
        res = OptimizeResult()
        res.repulsive = self.repulsive
        res.grid = self.grid
        res.dim = self.dim
        res.v_potential = self.repulsive + self.fi
        # reclaim convergence result
        self.sch_solver.v_potential = res.v_potential
        res.eig_val, res.wave_function = self.sch_solver.calc_esys()
        if res.wave_function.ndim == 3:
            res.fermi_energy, res.n_states = self.fermi_util.fermilevel(
                res.eig_val[:, 0], res.wave_function[:, 0, :], self.temp)
        elif res.wave_function.ndim == 4:
            res.fermi_energy, res.n_states = self.fermi_util.fermilevel(
                res.eig_val[:, 0, 0], res.wave_function[:, 0, 0, :], self.temp)
        else:
            raise ValueError('Unsupported wave function.')
        res.sigma = self._calc_net_density(res.n_states, res.wave_function)
        # full wave function
        res.wave_function = np.asarray(res.wave_function)
        self.poi_solver.charge_density = res.sigma * const.e
        self.poi_solver.calc_poisson()
        res.e_field = self.poi_solver.e_field
        # Accumulate electron areal density in the subbands
        res.electron_density = np.zeros_like(self.doping)
        for i, distri in enumerate(res.n_states):
            res.electron_density += distri * res.wave_function[i] * np.conj(
                res.wave_function[i])

        return res, loss
# %%
# Quick test
