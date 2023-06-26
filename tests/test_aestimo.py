# %%
import numpy as np
from matplotlib import pyplot as plt

from pyqhe.core.structure import Layer, Structure1D
from pyqhe.schrodinger_poisson import SchrodingerPoisson
from pyqhe.equation.poisson import PoissonFDM, PoissonODE
from pyqhe.equation.schrodinger import SchrodingerMatrix, SchrodingerShooting
from pyqhe.utility.constant import const

const.to_natural_unit()
# %%
# construct model
layer_list = []
layer_list.append(Layer(10, 0.3, 0.0, name='barrier'))
layer_list.append(Layer(5, 0.3, 5e17, name='n-type'))
layer_list.append(Layer(5, 0.3, 0.0, name='spacer'))
layer_list.append(Layer(11, 0, 0, name='quantum_well'))
layer_list.append(Layer(5, 0.3, 0.0, name='spacer'))
layer_list.append(Layer(5, 0.3, 5e17, name='n-type'))
layer_list.append(Layer(10, 0.3, 0.0, name='barrier'))

model = Structure1D(layer_list, temp=60, dz=0.1)
# instance of class SchrodingerPoisson
schpois = SchrodingerPoisson(
    model,
    schsolver=SchrodingerMatrix,
    poisolver=PoissonFDM,
)
# perform self consistent optimization
res, _ = schpois.self_consistent_minimize()
res.plot_quantum_well()
# %%
# state, Energy, Population, effective mass
#      ,meV    , cm**-1    , m_e**-1
#     0 987.852       5e+15, 6.15e-32
#     1 1063.06    2.15e+10, 6.3e-32
#     2 1177.93         5.5, 6.97e-32
# Schrodinger shooting
# 0.987865, 1.062842, 1.177560
# Schrodinger Matrix: not consider Position-dependent effective mass
# 0.989861, 1.068063, 1.179662
# Schrodinger Shooting + FDM
# 0.983025, 1.058021, 1.173038
# Schrodinger Matrix + FDM
# 0.985020, 1.063246, 1.175279
