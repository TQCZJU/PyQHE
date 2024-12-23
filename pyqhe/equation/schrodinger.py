# %%
from functools import reduce
from abc import ABC, abstractmethod
from typing import List, Union
import numpy as np
import scipy.linalg as sciLA
import scipy.sparse as sp
import scipy.sparse.linalg as sciLAS
from scipy import optimize

from pyqhe.utility.constant import const
from pyqhe.utility.utils import csr_broadcast


class SchrodingerSolver(ABC):
    """Meta class for Schrodinger equation solver."""

    def __init__(self) -> None:
        # properties
        self.grid = None
        self.v_potential = None
        self.cb_meff = None
        # Cache parameters
        self.psi = None

    @abstractmethod
    def calc_evals(self):
        """Calculate eigenenergy of any bound states in the chosen potential."""

    @abstractmethod
    def calc_esys(self):
        """Calculate wave function and eigenenergy."""


class SchrodingerShooting(SchrodingerSolver):
    """Shooting method solver for calculation Schrodinger equation."""

    def __init__(self, grid: np.ndarray, v_potential, cb_meff, *args,
                 **kwargs) -> None:
        if isinstance(grid, list):  # nd grid
            grid = grid[0]
        # Schrodinger equation's parameters
        self.v_potential = v_potential
        self.cb_meff = cb_meff
        # parse grid configuration
        self.grid = grid
        self.delta_z = grid[1] - grid[0]
        # Shooting method parameters for Schrödinger Equation solution
        # Energy step (eV) for initial search. Initial delta_E is 1 meV.
        self.delta_e = 0.5 / 1e3

    def _psi_iteration(self, energy_x0):
        """Use `numpy.nditer` to get iteration solution.

        Args:
            energy_x0: energy to start wavefunction iteration.
        Returns:
            Diverge of psi at infinite x.
        """

        psi = np.zeros(self.grid.shape)
        psi[0] = 0.0
        psi[1] = 1.0
        const_0 = 2 * (self.delta_z / const.hbar)**2
        with np.nditer(psi, flags=['c_index'], op_flags=['writeonly']) as it:
            for x in it:
                if it.index <= 1:
                    continue
                const_1 = 2.0 / (self.cb_meff[it.index - 1] +
                                 self.cb_meff[it.index - 2])
                const_2 = 2.0 / (self.cb_meff[it.index - 1] +
                                 self.cb_meff[it.index])
                x[...] = ((const_0 *
                           (self.v_potential[it.index - 1] - energy_x0) +
                           const_2 + const_1) * psi[it.index - 1] -
                          const_1 * psi[it.index - 2]) / const_2
        self.psi = psi

        return psi[-1]  # psi at inf

    def calc_evals(self,
                   energy_x0=None,
                   max_energy=None,
                   num_band=None,
                   **kwargs):
        """Calculate eigenenergy of any bound states in the chosen potential.

        Args:
            max_energy: shooting eigenenergy that smaller than `max_energy`
            num_band: number of band to shoot eigenenergy.
            energy_x0: minimum energy to start subband search. (Unit in Joules)
            kwargs: argument for `scipy.optimize.root_scalar`
        """

        # find brackets contain eigenvalue
        if energy_x0 is None:
            energy_x0 = np.min(self.v_potential) * 0.9
        if max_energy is None:
            max_energy = np.max(self.v_potential) + 0.1 * (
                np.max(self.v_potential) - np.min(self.v_potential))
        # shooting energy list
        num_shooting = round((max_energy - energy_x0) / self.delta_e)
        energy_list = np.linspace(energy_x0, max_energy, num_shooting)
        psi_list = [self._psi_iteration(energy) for energy in energy_list]
        # check sign change
        shooting_index = np.argwhere(np.diff(np.sign(psi_list)))
        # truncated eigenenergy
        if num_band is not None:
            shooting_index = shooting_index[:num_band]
        # find root in brackets
        shooting_bracket = [
            [energy_list[idx], energy_list[idx + 1]] for idx in shooting_index
        ]
        result_sol = []
        for bracket in shooting_bracket:
            sol = optimize.root_scalar(self._psi_iteration,
                                       bracket=bracket,
                                       **kwargs)
            result_sol.append(sol)
        # acquire eigenenergy
        eig_val = [sol.root for sol in result_sol]

        return eig_val

    def calc_esys(self, **kwargs):
        """Calculate wave function and eigenenergy.

        Args:
            max_energy: shooting eigenenergy that smaller than `max_energy`
            num_band: number of band to shoot eigenenergy.
            energy_x0: minimum energy to start subband search. (Unit in Joules)
            kwargs: argument for `scipy.optimize.root_scalar`
        """

        eig_val = self.calc_evals(**kwargs)
        wave_function = []
        for energy in eig_val:
            self._psi_iteration(energy)
            norm = np.sqrt(
                np.trapezoid(self.psi * np.conj(self.psi),
                             x=self.grid))  # l2-norm
            wave_function.append(self.psi / norm)

        return eig_val, np.asarray(wave_function)


class SchrodingerMatrix(SchrodingerSolver):
    """N-D Schrodinger equation solver based on discrete Laplacian and FDM.

    Args:
        grid: ndarray or list of ndarray
            when solving n-d schrodinger equation, just pass n grid array
    """

    def __init__(self,
                 grid: Union[List[np.ndarray], np.ndarray],
                 v_potential: np.ndarray,
                 cb_meff: np.ndarray,
                 bound_period: list = None,
                 quantum_region=None) -> None:
        super().__init__()

        if not isinstance(grid, list):  # 1d grid
            grid = [grid]
        self.grid = grid
        self.dim = [grid_axis.shape[0] for grid_axis in self.grid]
        # check matrix dim
        if v_potential.shape == tuple(self.dim) or len(self.dim) == 1:
            self.v_potential = v_potential
        else:
            raise ValueError('The dimension of v_potential is not match')
        if cb_meff.shape == tuple(self.dim) or len(self.dim) == 1:
            self.cb_meff = cb_meff
        else:
            raise ValueError('The dimension of cb_meff is not match.')
        if quantum_region is None:
            self.quantum_region = [
                np.ones_like(grid) == 1 for grid in self.grid
            ]
        else:
            if not isinstance(quantum_region, list):  # 1d grid
                quantum_region = [quantum_region]
            self.quantum_region = quantum_region
        self.quantum_dim = [
            len(g[m]) for g, m in zip(self.grid, self.quantum_region)
        ]
        self.quantum_mask = reduce(np.outer,
                                   self.quantum_region).reshape(self.dim)
        if bound_period is None:
            self.bound_period = [None] * len(self.dim)
        elif len(bound_period) == len(self.dim):
            self.bound_period = bound_period
        else:
            raise ValueError('The dimension of bound_period is not match.')

    def build_kinetic_operator(self, loc):
        """Build 1D time independent Schrodinger equation kinetic operator.

        Args:
            dim: dimension of kinetic operator.
        """
        delta = self.grid[loc][1] - self.grid[loc][0]
        # o(h) approach
        # mat_d = csr_broadcast(
        #     self.build_first_order_differential_operator(loc, 'backward'),
        #     1 / self.cb_meff)
        # mat_d = self.build_first_order_differential_operator(
        #     loc, 'forward') @ mat_d / delta**2

        # o(h^2) approach
        mat_d = self.build_second_order_differential_operator(loc) / delta**2
        if self.bound_period[loc]:  # add period boundary condition
            mat_d[0, -1] = 1
            mat_d[-1, 0] = 1
        return mat_d

    def build_second_order_differential_operator(self, loc):
        """Build 1D time independent Schrodinger equation second order
        differential operator.

        Args:
            loc: index of grid axis.
        """
        inv_cb_meff = 1 / self.cb_meff
        mean_inv_cb_meff = (inv_cb_meff[:-1] + inv_cb_meff[1:]) / 2
        mean_inv_cb_meff = np.append(mean_inv_cb_meff, inv_cb_meff[-1])
        mean_inv_cb_meff = np.insert(mean_inv_cb_meff, 0, inv_cb_meff[0])
        mat_d = sp.diags([
            mean_inv_cb_meff[1:-1], -mean_inv_cb_meff[:-1] -
            mean_inv_cb_meff[1:], mean_inv_cb_meff[1:-1]
        ], [-1, 0, 1],
                         format='csr')
        return mat_d

    def build_first_order_differential_operator(self, loc, difftype='central'):
        """Build 1D time independent Schrodinger equation first order
        differential operator default use the central difference method.

        Args:
            loc: index of grid axis.
            diff_type: type of differential method.
        """
        if difftype == 'central':
            mat_d = sp.diags([
                0.5 * np.ones(self.dim[loc] - 1),
                -0.5 * np.ones(self.dim[loc] - 1)
            ], [1, -1],
                             format='csr')
        elif difftype == 'forward':
            mat_d = sp.diags(
                [np.ones(self.dim[loc] - 1), -np.ones(self.dim[loc])], [1, 0],
                format='csr')
        elif difftype == 'backward':
            mat_d = sp.diags(
                [np.ones(self.dim[loc]), -np.ones(self.dim[loc] - 1)], [0, -1],
                format='csr')
        return mat_d

    def build_potential_operator(self):
        """Build 1D time independent Schrodinger equation potential operator. """
        vp = self.v_potential[self.quantum_mask].flatten()
        return sp.diags(vp)

    def hamiltonian(self):
        """Construct time independent Schrodinger equation."""
        # construct V and cb_meff matrix
        # discrete laplacian
        k_mat_list = []
        for loc, _ in enumerate(self.dim):
            mat = self.build_kinetic_operator(loc)
            coeff = -0.5 * const.hbar**2
            k_mat_list.append(mat * coeff)
            # construct n-d kinetic operator by tensor product
            # k_opt = tensor(*kron_list)
            # tensor contraction
            #TODO: the PDEM method looks broken
            # k_opt = np.einsum(k_opt.reshape(self.quantum_dim * 2),
            #                   np.arange(len(self.quantum_dim * 2)), coeff,
            #                   np.arange(len(self.quantum_dim)),
            #                   np.arange(len(self.quantum_dim * 2)))
            # k_opt = (k_opt.T * coeff).T  # quantum well has the smallest m_e
            # k_mat_list.append(
            #     k_opt.reshape(np.prod(self.quantum_dim),
            #                   np.prod(self.quantum_dim)))
            # k_mat_list.append(k_opt / coeff)
        k_mat = reduce(sp.kronsum, k_mat_list[::-1])
        v_mat = self.build_potential_operator()

        return k_mat + v_mat

    def calc_evals(self, k=3):
        ham = self.hamiltonian()
        evals = sciLAS.eigsh(ham, k=k, which='SA', return_eigenvectors=False)
        return evals

    def calc_esys(self, k=3):
        ham = self.hamiltonian()
        # eig_val, eig_vec = sciLAS.eigsh(ham, k=k, which='SA')
        # assert np.allclose(ham.toarray(), ham.toarray().T)
        eig_val, eig_vec = sciLA.eigh(ham.toarray())
        wave_func = []
        for mask_vec in eig_vec.T:
            # reshape eigenvector to discrete wave function
            vec = np.zeros(self.dim)
            vec[self.quantum_mask] = mask_vec
            vec = vec.reshape(self.dim)
            # normalize
            norm = vec * np.conj(vec)
            for grid in self.grid[::-1]:
                norm = np.trapezoid(norm, grid)
            wave_func.append(vec / np.sqrt(norm))

        return eig_val, np.array(wave_func)


class SchrodingerFiori(SchrodingerSolver):
    """Approach to multi-body 3d wave function of electrons system.
    The method from Journal of Computational Electronics 1: 39–42, 2002. The 3D
    schrodinger equation can be separated into 1d and 2d part.
    """

    def __init__(self,
                 grid: Union[List[np.ndarray], np.ndarray],
                 v_potential: np.ndarray,
                 cb_meff: np.ndarray,
                 bound_period: list = None,
                 quantum_region=None) -> None:
        super().__init__()

        if not isinstance(grid, list):  # 1d grid
            grid = [grid]
        self.grid = grid
        self.dim = [grid_axis.shape[0] for grid_axis in self.grid]
        # check matrix dim
        if v_potential.shape == tuple(self.dim) or len(self.dim) == 1:
            self.v_potential = v_potential
        else:
            raise ValueError('The dimension of v_potential is not match')
        if cb_meff.shape == tuple(self.dim) or len(self.dim) == 1:
            self.cb_meff = cb_meff
        else:
            raise ValueError('The dimension of cb_meff is not match.')
        if quantum_region is None:
            self.quantum_region = [
                np.ones_like(grid) == 1 for grid in self.grid
            ]
        else:
            if not isinstance(quantum_region, list):  # 1d grid
                quantum_region = [quantum_region]
            self.quantum_region = quantum_region
        self.quantum_dim = [
            len(g[m]) for g, m in zip(self.grid, self.quantum_region)
        ]
        self.quantum_mask = reduce(np.outer, self.quantum_region).reshape(
            self.quantum_dim)
        if bound_period is None:
            self.bound_period = [None] * len(self.dim)
        elif len(bound_period) == len(self.dim):
            self.bound_period = bound_period
        else:
            raise ValueError('The dimension of bound_period is not match.')

    def build_kinetic_operator(self):
        """Build 1D time independent Schrodinger equation kinetic operator.

        Args:
            dim: dimension of kinetic operator.
        """
        # construct the kinetic operator for z axis equation
        mat_d = -2 * np.eye(self.dim[-1]) + np.eye(self.dim[-1], k=-1) + np.eye(
            self.dim[-1], k=1)
        if self.bound_period[-1]:  # add period boundary condition
            mat_d[0, -1] = 1
            mat_d[-1, 0] = 1
        return mat_d

    def build_potential_operator(self, lateral_idx):
        """Build 1D time independent Schrodinger equation potential operator.

        Args:
            lateral_idx: index along the lateral direction
        """

        return np.diag(self.v_potential[lateral_idx].flatten())

    def hamiltonian(self, lateral_idx):
        """Construct time independent Schrodinger equation."""
        # construct V and cb_meff matrix
        # discrete laplacian

        k_opt = self.build_kinetic_operator()
        delta = self.grid[-1][1] - self.grid[-1][0]
        coeff = -0.5 * const.hbar**2 / self.cb_meff[lateral_idx] / delta**2
        # row-wise multiplication
        k_opt = k_opt * coeff.reshape(-1, 1)
        v_mat = self.build_potential_operator(lateral_idx)

        return k_opt + v_mat

    def calc_evals(self):
        # we use a set of index describe the point in lateral direction
        num_lateral = np.prod(self.dim[:-1])
        # initialize data set
        eigval_set = []
        for flat_idx in range(num_lateral):
            multi_idx = np.unravel_index(flat_idx, self.dim[:-1])
            # collect related data for Schrodinger equation
            ham = self.hamiltonian(multi_idx)
            # calculate eigensystems
            eig_val = sciLA.eigh(ham, eigvals_only=True)
            eigval_set.append(eig_val)
        # reshape the set of eigensystems
        # the multi-dimension index of eigval_set is [lateral_idx0, lateral_idx1, i-th eigenval]
        eigval_set = np.array(eigval_set).reshape(self.dim)
        # swap axis to unify tensor definition [i-th eigenvec, lateral_idx0, lateral_idx1, growth axis]
        for i in range(len(self.dim), 1, -1):
            eigval_set = np.swapaxes(eigval_set, i - 1, i - 2)
        return eigval_set

    def calc_esys(self):
        # we use a set of index describe the point in lateral direction
        num_lateral = np.prod(self.dim[:-1])
        # initialize data set
        eigval_set = []
        wave_func_set = []
        for flat_idx in range(num_lateral):
            multi_idx = np.unravel_index(flat_idx, self.dim[:-1])
            # collect related data for Schrodinger equation
            ham = self.hamiltonian(multi_idx)
            # calculate eigensystems
            eig_val, eig_vec = sciLA.eigh(ham)
            eigval_set.append(eig_val)
            # assemble eigen-energy set
            wave_func = []
            for vec in eig_vec.T:
                # normalize
                norm = vec * np.conj(vec)
                norm = np.trapezoid(norm, self.grid[-1])
                wave_func.append(vec / np.sqrt(norm))
            wave_func = np.array(wave_func)
            wave_func_set.append(wave_func)
        # reshape the set of eigensystems
        # the multi-dimension index of eigval_set is [lateral_idx0, lateral_idx1, i-th eigenval]
        eigval_set = np.array(eigval_set).reshape(self.dim)
        # For wave_func_set, we use [lateral_idx0, lateral_idx1, i-th eigenvec, growth axis]
        wave_func_set = np.array(wave_func_set).reshape(self.dim + [
            self.dim[-1],
        ])
        # swap axis to unify tensor definition [i-th eigenvec, lateral_idx0, lateral_idx1, growth axis]
        for i in range(len(self.dim), 1, -1):
            eigval_set = np.swapaxes(eigval_set, i - 1, i - 2)
            wave_func_set = np.swapaxes(wave_func_set, i - 1, i - 2)
        return eigval_set, wave_func_set


# %%
# QuickTest
if __name__ == '__main__':
    from matplotlib import pyplot as plt
    from matplotlib import cm

    # x = np.linspace(-1, 1, 5)
    # y = np.linspace(-1, 1, 20)
    # z = np.linspace(-1, 1, 30)
    # xv, yv, zv = np.meshgrid(x, y, z, indexing='ij')
    # z_barrier = (zv <= -0.5) + (zv >= 0.5)
    # v_potential = np.zeros([5, 20, 30])
    # v_potential[z_barrier] = 10  # set barrier
    # sol = SchrodingerFiori([x, y, z], v_potential,
    #                        np.ones_like(v_potential) * const.m_e)
    # eig_val, wf = sol.calc_esys()
    # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # surf = ax.plot_surface(yv[0],
    #                        zv[0],
    #                        wf[2, 2, :],
    #                        cmap=cm.coolwarm,
    #                        linewidth=0,
    #                        antialiased=False)
    # ax.set_xlabel('y')
    # ax.set_ylabel('z')
    # ax.set_zlabel(r'$\psi(r,z)$')
    # # %%
    # x = np.linspace(-1, 1, 50)
    # y = np.linspace(-1, 1, 55)
    # xv, yv = np.meshgrid(x, y, indexing='ij')
    # x_barrier = (xv <= -0.5) + (xv >= 0.5)
    # y_barrier = (yv <= -0.5) + (yv >= 0.5)
    # v_potential = np.zeros([50, 55])
    # v_potential[x_barrier + y_barrier] = 1  # set barrier
    # sol = SchrodingerMatrix([x, y], v_potential,
    #                         np.ones_like(v_potential) * const.m_e)
    # eig_val, wf = sol.calc_esys()
    # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # surf = ax.plot_surface(xv,
    #                        yv,
    #                        wf[3],
    #                        cmap=cm.coolwarm,
    #                        linewidth=0,
    #                        antialiased=False)
    # %%
    import time

    def sweep_num_mesh(num_mesh, plot=False):
        grid = np.linspace(-1, 1, num_mesh)
        psi = np.zeros(grid.shape)
        v_potential = np.zeros(grid.shape)
        # Quantum well
        z_barrier = (grid <= -0.5) + (grid >= 0.5)
        v_potential[z_barrier] = 10
        quantum_region = (grid <= 0) * (grid > 0 - 0.5)
        cb_meff = np.ones(grid.shape) * const.m_e
        cb_meff[z_barrier] = const.m_e * 1.35
        start_time = time.time()
        solver = SchrodingerMatrix(
            grid,
            v_potential,
            cb_meff,
            #    quantum_region=quantum_region,
        )
        val, vec = solver.calc_esys()
        end_time = time.time()
        wall_time_mat = end_time - start_time
        print(f'Matrix method wall time: {wall_time_mat}')
        print(val[:3])
        if plot:
            fig, (ax1,
                  ax2) = plt.subplots(2,
                                      1,
                                      figsize=(8, 10),
                                      gridspec_kw={'height_ratios': [3, 1]})
            # Top subplot
            ax1.plot(solver.grid[0], solver.v_potential, label='Potential')
            ax1.plot(solver.grid[0],
                     vec[0] + val[0],
                     label=r'$\psi_0$, E={:.3f}'.format(val[0]))
            ax1.plot(solver.grid[0],
                     vec[1] + val[1],
                     label=r'$\psi_1$, E={:.3f}'.format(val[1]))
            ax1.plot(solver.grid[0],
                     vec[2] + val[2],
                     label=r'$\psi_2$, E={:.3f}'.format(val[2]))
            ax1.set_title(
                'Quantum Well simulation by matrix method and shooting method')
            ax1.set_ylabel('Energy')

            # Bottom subplot
            ax2.plot(grid, cb_meff / const.m_e, label='Effective Mass')
            ax2.legend()
            ax2.set_ylabel('Effective Mass')
            ax2.set_xlabel('Position')

        start_time = time.time()
        solver = SchrodingerShooting(
            grid,
            v_potential,
            cb_meff,
            #    quantum_region=quantum_region,
        )
        val_s, vec_s = solver.calc_esys()
        end_time = time.time()
        wall_time_shooting = end_time - start_time
        if plot:
            ax1.plot(solver.grid,
                     vec_s[0] + val_s[0],
                     '.',
                     label=r'$\psi_0$ by shooting method, E={:.3f}'.format(
                         val_s[0]))
            ax1.plot(solver.grid,
                     vec_s[1] + val_s[1],
                     '.',
                     label=r'$\psi_1$ by shooting method, E={:.3f}'.format(
                         val[1]))
            ax1.plot(solver.grid,
                     vec_s[2] + val_s[2],
                     '.',
                     label=r'$\psi_2$ by shooting method, E={:.3f}'.format(
                         val_s[2]))
            ax1.legend()
            plt.tight_layout()
            plt.show()

        print(val_s[:3])
        print(f'Shooting method wall time: {wall_time_shooting}')
        # compute the overlap of the ground state wave function
        overlap_0 = np.vdot(vec[0], vec_s[0]) / np.vdot(vec[0], vec[0])
        overlap_1 = np.vdot(vec[1], vec_s[1]) / np.vdot(vec[1], vec[1])
        overlap_2 = np.vdot(vec[2], vec_s[2]) / np.vdot(vec[2], vec[2])
        return np.array([overlap_0, overlap_1,
                         overlap_2]), wall_time_mat, wall_time_shooting

    sweep_num_mesh(110, plot=True)
    # %%
    overlap_list = []
    wall_time_mat_list = []
    wall_time_shooting_list = []
    for num_mesh in range(10, 1000, 100):
        overlap, wall_time_mat, wall_time_shooting = sweep_num_mesh(num_mesh)
        overlap_list.append(overlap)
        wall_time_mat_list.append(wall_time_mat)
        wall_time_shooting_list.append(wall_time_shooting)
    # %%
    # plot the overlap versus the number of meshes
    fig, ax = plt.subplots()
    ax.plot(range(110, 1000, 100),
            np.abs(overlap_list[1:])**2,
            label=[
                r'$|\langle\psi_{shoot,0}|\psi_{mat,0}\rangle|^2$',
                r'$|\langle\psi_{shoot,1}|\psi_{mat,1}\rangle|^2$',
                r'$|\langle\psi_{shoot,2}|\psi_{mat,2}\rangle|^2$'
            ])
    ax.plot(range(110, 1000, 100),
            np.abs(overlap_list[1:])**2,
            '.',
            color='black')
    ax.set_xlabel('Number of meshes')
    ax.set_ylabel(r'Overlap $|\langle\psi_{shoot}|\psi_{mat}\rangle|^2$')
    ax.set_yscale('log')
    ax.set_title(
        'Wave function overlap between shooting method and matrix method')
    ax.legend()
    # %%
    # plot the wall time versus the number of meshes
    fig, ax = plt.subplots()
    ax.plot(range(10, 1000, 100), wall_time_mat_list, label='Matrix method')
    ax.scatter(range(10, 1000, 100), wall_time_mat_list)
    ax.plot(range(10, 1000, 100),
            wall_time_shooting_list,
            label='Shooting method')
    ax.scatter(range(10, 1000, 100), wall_time_shooting_list)
    ax.set_xlabel('Number of meshes')
    ax.set_ylabel('Wall time (s)')
    ax.set_title('Wall time of Schrodinger equation solver')
    ax.set_yscale('log')
    ax.legend()
    # %%
    fig, ax1 = plt.subplots()

    # Plot the data on the first (bottom) axes
    ax1.plot(solver.grid[0],
             solver.v_potential,
             label='Potential in z coordinate')
    ax1.legend()
    ax1.set_xlabel('z coordinate')
    ax1.set_xticks(solver.grid[0])
    ax1.tick_params(which='both', direction='in')
    # Create a second (top) axes, sharing the y-axis with the first
    ax2 = ax1.twiny()
    # Plot the data on the second (top) axes
    ax2.plot(solver._coord_phi,
             solver.v_potential,
             'r',
             alpha=0.,
             label='Potential in phi coordinate')
    # ax2.legend(loc='lower left')
    # Set the values of the second x-axis
    ax2.set_xticks(solver._coord_phi)
    # Set the scale of the second x-axis to log
    # Set the label of the second x-axis
    ax2.set_xlabel(r'$\phi$ coordinate')
    ax2.tick_params(which='both', direction='in')
    # Show the plot
    # plt.title(r'Computing state and gradients ($\delta t$=0.01, T=50)')
    plt.tight_layout()
