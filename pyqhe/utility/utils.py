import numpy as np
import scipy.sparse as sp


def tensor(*opt):
    if len(opt) == 1:
        opt = opt[0]
    kron_product = opt[0]
    for element in opt[1:]:
        kron_product = np.kron(kron_product, element)
    return kron_product


def csr_broadcast(csr_mat, vec):
    """Broadcast multiplication for CSR matrix.
    The function is equal to `(csr_mat.T * vex).T` in dense form

    Args:
        csr_mat: CSR matrix
        vec: 1D array
    """
    if not isinstance(csr_mat, sp.csr_matrix):
        raise ValueError('Matrix must be CSR.')
    new_mat = csr_mat.copy()
    new_mat.data *= vec.repeat(np.diff(new_mat.indptr))
    return new_mat
