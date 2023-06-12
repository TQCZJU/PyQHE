from math import pi

__all__ = ['const']


class _Constant:
    """The class for global unit setting."""

    def __init__(self) -> None:
        # using eV units
        self.e = 1.602176e-19  # C
        self.kb = 8.6173e-5  # eV/K
        self.nii = 0.0
        self.hbar = 1.054588757e-34 / self.e
        self.m_e = 0.511e6 / (2.99792458e17)**2  # eV/(nm * s^-1)^2
        self.pi = pi
        self.eps0 = 8.8541878176e-21  # F/nm

    def to_natural_unit(self):
        """Convert the global unit to the natural unit."""
        self.hbar = 1.0
        self.eps0 = 1.0

        self.m_e = 13.12301617266444
        self.e = 18.095121009464684  # sqrt(4 * \pi * \alpha)

    @property
    def h(self):
        return self.hbar * pi


const = _Constant()
