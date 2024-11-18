from abc import ABC, abstractmethod
from functools import reduce
import warnings

import numpy as np
from scipy.integrate import cumulative_trapezoid
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from pyqhe.utility.constant import const
from pyqhe.utility.utils import tensor, csr_broadcast


class PoissonSolver(ABC):
    """Meta-class for Poisson equation solver."""

    def __init__(self) -> None:
        # properties
        self.grid = None
        self.charge_density = None
        self.eps = None
        # Cache parameters
        self.e_field = None
        self.v_potential = None

    @abstractmethod
    def calc_poisson(self):
        """Solve Poisson equation and get related electric field and potential.
        """


class PoissonODE(PoissonSolver):
    """ODE integration solver for 1d Poisson equation.

    Args:
        charge_density: The net charge density. Typically, use the dopants
            density minus the electron density.
    """

    def __init__(self, grid: np.ndarray, charge_density: np.ndarray,
                 eps: np.ndarray, *args, **kwargs) -> None:
        super().__init__()
        if isinstance(grid, list):  # nd grid
            grid = grid[0]
        self.grid = grid
        self.charge_density = charge_density
        self.eps = eps

    def calc_poisson(self, **kwargs):
        """Calculate electric field."""
        # Gauss's law
        d_z = cumulative_trapezoid(self.charge_density, self.grid, initial=0)
        self.e_field = d_z / self.eps
        # integral the potential
        # note here we put a electron, dV/dz = E
        self.v_potential = cumulative_trapezoid(-1.0 * self.e_field,
                                                self.grid,
                                                initial=0)

        return self.v_potential


class PoissonFDM(PoissonSolver):
    """Finite difference method solver for nd Poisson equation.

    Args:
        grid: list of array
            the grid for FDM
        charge_density: 2d-array
        eps: 2d-array
            the electric permittivity
        bound_dirichlet: 2d-array
            the matrix contains information about the dirichlet boundary,
            set np.nan to non-boundary points.
        bound_period: list of bool
            true for set the period boundary condition at the edge of the
            simulation area
        bound_neumann: list of list of bool
            true for set the neumann boundary condition `dV/dn=const` at the
            edges of the simulation area
    """

    def __init__(self,
                 grid: np.ndarray,
                 charge_density: np.ndarray,
                 eps: np.ndarray,
                 bound_dirichlet: np.ndarray = None,
                 bound_period: list = None,
                 bound_neumann: np.ndarray = None,
                 rotational_symmetry: list = None) -> None:
        super().__init__()

        if not isinstance(grid, list):  # 1d grid
            grid = [grid]
        self.grid = grid
        self.dim = [grid_axis.shape[0] for grid_axis in self.grid]
        # check matrix dim
        if charge_density.shape == tuple(self.dim) or len(self.dim) == 1:
            self.charge_density = charge_density
        else:
            raise ValueError('The dimension of charge_density is not match')
        if eps.shape == tuple(self.dim) or len(self.dim) == 1:
            self.eps = eps
        else:
            raise ValueError('The dimension of eps is not match.')

        if bound_dirichlet is None:
            self.bound_dirichlet = None
        elif bound_dirichlet.shape == tuple(self.dim) or len(self.dim) == 1:
            self.bound_dirichlet = bound_dirichlet
        else:
            raise ValueError('The dimension of bound_dirichlet is not match.')

        # Deprecated(We only need to assign Neumann boundary on both sides)
        # if bound_neumann is None:
        #     self.bound_neumann = None
        # elif bound_neumann.shape == tuple(self.dim) or len(self.dim) == 1:
        #     self.bound_neumann = bound_neumann
        # else:
        #     raise ValueError('The dimension of bound_neumann is not match.')
        # noted the above method should consider normal direction carefully

        if bound_neumann is None:
            self.bound_neumann = [[None] * 2] * len(self.dim)
        elif len(bound_neumann) == len(self.dim):
            self.bound_neumann = bound_neumann
        else:
            raise ValueError('The dimension of bound_neumann is not match.')

        if bound_period is None:
            self.bound_period = [None] * len(self.dim)
        elif len(bound_period) == len(self.dim):
            self.bound_period = bound_period
        else:
            raise ValueError('The dimension of bound_period is not match.')

        if rotational_symmetry is None:
            self.rotational_symmetry = [None] * len(self.dim)
        elif len(rotational_symmetry) == len(self.dim):
            self.rotational_symmetry = rotational_symmetry
        else:
            raise ValueError(
                'The dimension of rotational_symmetry is not match.')

    def build_second_order_differential_operator(self, loc):
        """Build 1D Poisson equation second order differential operator.

        Args:
            loc: index of grid axis.
        """
        mat_d = sp.diags([
            np.ones(self.dim[loc] - 1), -2 * np.ones(self.dim[loc]),
            np.ones(self.dim[loc] - 1)
        ], [-1, 0, 1],
                         format='csr')
        return mat_d

    def builf_first_order_differential_operator(self, loc):
        """Build 1D Poisson equation first order differential operator
        use the central difference method.

        Args:
            loc: index of grid axis.
        """
        mat_d = sp.diags([
            0.5 * np.ones(self.dim[loc] - 1), -0.5 * np.ones(self.dim[loc] - 1)
        ], [1, -1],
                         format='csr')
        # compute the position-dependent dielectric constant
        eps = self.eps.flatten()
        pddc_term = (np.roll(eps, -1) - np.roll(eps, 1)) / eps / 2
        # pddc_term = np.diff(eps, prepend=eps[0]) / eps
        return csr_broadcast(mat_d, pddc_term)

    def build_d_matrix(self, loc):
        """Build 1D time independent Schrodinger equation kinetic operator.

        Args:
            dim: dimension of kinetic operator.
        """
        if self.rotational_symmetry[loc]:
            coeff_p1 = 1 + 1 / (2 * np.arange(0, self.dim[loc] - 1))
            coeff_p1[0] = 1.5
            coeff_m1 = 1 - 1 / (2 * np.arange(1, self.dim[loc]))
            mat_d = sp.diags([coeff_m1, -2 * np.ones(self.dim[loc]), coeff_p1],
                             [-1, 0, 1],
                             format='csr')
        else:
            mat_d = self.build_second_order_differential_operator(
                loc) + self.builf_first_order_differential_operator(loc)
        if self.bound_period[loc]:  # add period boundary condition
            mat_d[0, -1] = 1
            mat_d[-1, 0] = 1

        if self.bound_neumann[loc][0]:  # add Neumann boundary condition
            delta = self.grid[loc][1] - self.grid[loc][0]
            bound_a = np.zeros(self.dim[loc])
            # set matrix element
            bound_a[0] = -delta
            bound_a[1] = delta
            mat_d[0] = bound_a
        if self.bound_neumann[loc][1]:
            # note each axis should has two Neumann boundary
            delta = self.grid[loc][1] - self.grid[loc][0]
            bound_b = np.zeros(self.dim[loc])
            bound_b[-1] = -delta
            bound_b[-2] = delta
            mat_d[-1] = bound_b
        return mat_d

    def calc_poisson(self, **kwargs):
        """Calculate electric field."""

        # discrete laplacian
        a_mat_list = []

        for loc, _ in enumerate(self.dim):
            mat = self.build_d_matrix(loc)
            delta = self.grid[loc][1] - self.grid[loc][0]
            # assume all elements of `self.eps` are same
            # eps = self.eps.flatten()[0]
            a_mat_list.append(mat / delta**2)
            # # construct n-d kinetic operator by tensor product
            # d_opt = tensor(*kron_list)
            # # tensor contraction
            # d_opt = np.einsum(d_opt.reshape(self.dim * 2),
            #                   np.arange(len(self.dim * 2)), self.eps / delta**2,
            #                   np.arange(len(self.dim)),
            #                   np.arange(len(self.dim * 2)))
            # a_mat_list.append(
            #     d_opt.reshape(np.prod(self.dim), np.prod(self.dim)))

        a_mat = reduce(sp.kronsum, a_mat_list[::-1])
        b_vec = -1.0 * self.charge_density.flatten() / self.eps.flatten()

        # add Neumann boundary condition with second order accurate method
        for loc, _ in enumerate(self.dim):
            if any(self.bound_neumann[loc]
                  ):  # now adjust b_vec for Neumann boundary
                kernel_vec = np.zeros(self.dim[loc])
                if self.bound_neumann[loc][0]:
                    kernel_vec[0] = 0.5 * delta
                # add another Neumann boundary in axis
                if self.bound_neumann[loc][1]:
                    kernel_vec[-1] = 0.5 * delta
                kron_list = [np.ones(idim) for idim in self.dim[:loc]] + [
                    kernel_vec
                ] + [np.ones(idim) for idim in self.dim[loc + 1:]
                    ] + [1]  # auxiliary element for 1d solver
                diff_b_vec = tensor(*kron_list)
                bound_loc = np.nonzero(diff_b_vec)
                b_vec[bound_loc] *= diff_b_vec[bound_loc]

        # TODO: the following method should consider the normal direction of boundary
        # Deprecated(We only need to assign Neumann boundary on both sides)
        # if self.bound_neumann is not None:
        #     # the delta step for second-order accurate
        #     delta = self.grid[0][1] - self.grid[0][0]
        #     bound_b = self.bound_neumann * self.eps
        #     bound_a = np.zeros_like(b_vec)
        #     bound_loc = np.flatnonzero(~np.isnan(self.bound_neumann))
        #     bound_a[bound_loc] = -1.0 * self.eps.flatten()[bound_loc] / delta
        #     neumann_dir_index = self.dim[1]
        #     bound_mat = np.diag(bound_a) - np.diag(bound_a[:-neumann_dir_index], k=neumann_dir_index)
        #     a_mat[bound_loc] = bound_mat[bound_loc]
        #     b_vec[bound_loc] = bound_b.flatten()[bound_loc] + 0.5 * delta * b_vec[bound_loc]

        if self.bound_dirichlet is not None:
            # set temporary coefficient delta
            delta = self.grid[0][1] - self.grid[0][0]
            # adjust coefficient let matrix match the LAPACK's requirement
            bound_b = self.bound_dirichlet * self.eps / delta**2
            bound_a = np.zeros_like(b_vec)
            bound_loc = np.flatnonzero(~np.isnan(self.bound_dirichlet))
            bound_a[bound_loc] = 1 * self.eps.flatten()[bound_loc] / delta**2
            # tensor contraction
            bound_mat = np.diag(bound_a)
            a_mat[bound_loc] = bound_mat[bound_loc]
            b_vec[bound_loc] = bound_b.flatten()[bound_loc]
        # call the LAPACK
        self.v_potential = spsolve(a_mat, b_vec).reshape(self.dim)
        # calculate gradient of potential
        self.e_field = np.gradient(-1.0 * self.v_potential, *self.grid)
        return self.v_potential


class PoissonFDMCircular(PoissonFDM):
    """Finite difference method solver for quasi-3d Poisson equation.
    Circular symmetry help the solver reduce 3d equation to 2d.
    We assume the symmetry axis parallel to growth axis.

    Args:
        grid: list of array
            the grid for FDM
        charge_density: 2d-array
        eps: 2d-array
            the electric permittivity
        bound_dirichlet: 2d-array
            the matrix contains information about the dirichlet boundary,
            set np.nan to non-boundary points.
        bound_period: list of bool
            true for set the period boundary condition at the edge of the
            simulation area
        bound_neumann: list of list of bool
            true for set the neumann boundary condition `dV/dn=const` at the
            edges of the simulation area. Noted symmetry axis(left edge) must be set
    """

    def build_d_matrix(self, loc):
        """Build 1D time independent Schrodinger equation kinetic operator.

        Args:
            dim: dimension of kinetic operator.
        """
        if loc == 0:  # lateral axis
            coeff_p1 = 1 + 1 / (2 * np.arange(0, self.dim[loc] - 1))
            coeff_p1[0] = 1.5
            coeff_m1 = 1 - 1 / (2 * np.arange(1, self.dim[loc]))
            mat_d = sp.diags([coeff_m1, -2 * np.ones(self.dim[loc]), coeff_p1],
                             [-1, 0, 1],
                             format='csr')
        else:  # growth axis
            mat_d = sp.diags([
                np.ones(self.dim[loc] - 1), -2 * np.ones(self.dim[loc]),
                np.ones(self.dim[loc] - 1)
            ], [-1, 0, 1],
                             format='csr')

        if self.bound_period[loc]:  # add period boundary condition
            mat_d[0, -1] = 1
            mat_d[-1, 0] = 1

        # check left edge status
        if not self.bound_neumann[0][0]:
            warnings.warn('Left Neumann boundary must be set.')
            self.bound_neumann[0][0] = True

        if self.bound_neumann[loc][0]:  # add Neumann boundary condition
            delta = self.grid[loc][1] - self.grid[loc][0]
            bound_a = np.zeros(self.dim[loc])
            # set matrix element
            bound_a[0] = -delta
            bound_a[1] = delta
            mat_d[0] = bound_a
        if self.bound_neumann[loc][1]:
            # note each axis should has two Neumann boundary
            delta = self.grid[loc][1] - self.grid[loc][0]
            bound_b = np.zeros(self.dim[loc])
            bound_b[-1] = -delta
            bound_b[-2] = delta
            mat_d[-1] = bound_b
        return mat_d


# %%
# # QuickTest
if __name__ == '__main__':
    from matplotlib import pyplot as plt
    grid = np.linspace(-1, 1, 100)
    eps = np.ones(grid.shape)
    sigma = np.zeros(grid.shape)
    # Quantum well
    z_barrier = (grid <= -0.5) + (grid >= 0.5)
    sigma[z_barrier] = 10
    eps[z_barrier] = 1.3
    sol = PoissonFDM(grid, sigma, eps, bound_neumann=[[True, False]])
    sol.calc_poisson()

    plt.plot(grid, sol.v_potential - sol.v_potential[0], label='V')
    plt.plot(grid, sol.e_field, label='E')
    plt.legend()
    sol = PoissonODE(grid, sigma, eps)
    sol.calc_poisson()

    plt.plot(grid, sol.v_potential)
    plt.plot(grid, sol.e_field)
    # %%
    delta = 0.02
    x = np.arange(-1, 1, delta)
    y = np.arange(-0.5, 0.5, delta)
    xv, yv = np.meshgrid(x, y, indexing='ij')
    top_plate = (yv <= 0.1 + delta / 2) * (yv >= 0.1 - delta / 2)
    bottom_plate = (yv <= -0.1 + delta / 2) * (yv >= -0.1 - delta / 2)
    length = (xv <= 0.5) * (xv >= -0.5)
    bound = np.empty_like(xv)
    bound[:] = np.nan
    bound[top_plate * length] = 1
    bound[bottom_plate * length] = -1
    sol = PoissonFDM([x, y], np.zeros_like(xv),
                     np.ones_like(xv) * const.eps0, bound)
    # sol = PoissonFDM([x, y], bound * 10,
    #                   np.ones_like(bound) * const.eps0)
    v_p = sol.calc_poisson()
    # v potential
    plt.pcolormesh(xv, yv, v_p)
    plt.show()
    # e field
    plt.pcolormesh(xv, yv, np.sqrt(sol.e_field[0]**2 + sol.e_field[1]**2))
# %%
