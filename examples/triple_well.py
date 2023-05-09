# %%
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import jax
import jax.numpy as jnp

from pyqhe.core.structure import Layer, Structure1D
from pyqhe.schrodinger_poisson import SchrodingerPoisson
from pyqhe.equation.poisson import PoissonFDM, PoissonODE
from pyqhe.equation.schrodinger import SchrodingerMatrix, SchrodingerShooting
# %%

# construct model
layer_list = []
# layer_list.append(Layer(1, 1., 0.0, name='ohmic contact'))
layer_list.append(Layer(10, 0., 0.0, name='gate'))
layer_list.append(Layer(60, 0.36, 0.0, name='barrier'))
layer_list.append(Layer(5, 0.36, 1.8e18, name='n-type'))
layer_list.append(Layer(35, 0.36, 0.0, name='barrier'))
# add top screening layer
layer_list.append(Layer(2, 1.0, 0.0, name='cladding'))
layer_list.append(Layer(12, 0., 0.0, name='screen well'))
layer_list.append(Layer(2, 1.0, 0.0, name='cladding'))
# add primary quantum well
layer_list.append(Layer(25, 0.36, 0.0, name='spacer'))
layer_list.append(Layer(30, 0, 0, name='quantum_well'))
layer_list.append(Layer(25, 0.36, 0.0, name='spacer'))
# add bottom screening well
layer_list.append(Layer(2, 1.0, 0.0, name='cladding'))
layer_list.append(Layer(12, 0., 0.0, name='screen well'))
layer_list.append(Layer(2, 1.0, 0.0, name='cladding'))
# barrier and modulation doping
layer_list.append(Layer(30, 0.36, 0.0, name='barrier'))
layer_list.append(Layer(5, 0.36, 6.4e17, name='n-type'))
layer_list.append(Layer(60, 0.36, 0.0, name='substrate'))

model = Structure1D(layer_list, temp=10, dz=0.2)
grid = model.universal_grid[0]
schottky_barrier = np.full_like(grid, np.nan)
schottky_barrier[0] = 0.6
model.bound_dirichlet = schottky_barrier
quantum_region = (grid > 100) * (grid < 230)
# instance of class SchrodingerPoisson
schpois = SchrodingerPoisson(
    model,
    schsolver=SchrodingerMatrix,
    poisolver=PoissonFDM,
    quantum_region=quantum_region
)
# perform self consistent optimization
# for _ in range(10):
res, _ = schpois.self_consistent_minimize(20, 0.1)

res.plot_quantum_well()
plt.show()
# %%
plt.plot(res.grid[0], res.params)
# %%
