import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.lines import Line2D

from nuclina1 import *
from nuclina2 import *

import os, sys
sys.path.append(os.path.join('egtsimplex'))
import cosa

#%%
class DinamicaReplicador():
  def __init__(self, matriz_pagos):
    self.A = matriz_pagos
  def __call__(self, x, t):
    f = self.A @ x
    f_promedio = x @ f
    return x * (f - f_promedio)

#%% parametros
G, C = 1, 2

#%%
A_bravucones = np.array([
  [(G-C)/2, G, G],
  [0, G/2, 0],
  [0, G, G/2]
])
bravucones = DinamicaReplicador(A_bravucones)
dynamics = cosa.simplex_dynamics(bravucones)

# sns.set_context("paper", font_scale=1.5)
fig, ax = plt.subplots()
dynamics.plot_simplex(ax, typelabels=["Halcones","Palomas","Bravucones"])
nullclinas_bravucones(dynamics, G, C, ax)
fig.tight_layout()
fig.savefig("../figuras/ex01-a.pdf")

#%%
A_vengativos = np.array([
  [(G-C)/2, G, (G-C)/2],
  [0, G/2, G/2],
  [(G-C)/2, G/2, G/2]
])
vengativos = DinamicaReplicador(A_vengativos)
dynamics = cosa.simplex_dynamics(vengativos)

# sns.set_context("paper", font_scale=1.5)
fig, ax = plt.subplots()
dynamics.plot_simplex(ax, typelabels=["Halcones","Palomas","Vengativos"])
nullclinas_vengativos(dynamics, G, C, ax)
fig.tight_layout()
fig.savefig("../figuras/ex01-b.pdf")