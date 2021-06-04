import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
#%%
#@title Ejecutar para cargar nullclinas
#@markdown Fórmulas analíticas de las nullclinas.
from matplotlib.lines import Line2D

def nullclinas_bravucones(dynamics, G, C, ax):
    densidad = np.linspace(0, 1, 1000)
    h_nullclinas = []
    h = np.zeros_like(densidad)
    h_nullclinas.append(
        dynamics.curva_simplex(h, densidad, 1 - densidad - h)
    )
    h = np.ones_like(densidad)
    h_nullclinas.append(
        dynamics.curva_simplex(h, densidad, 1 - densidad - h)
    )
    h = np.full_like(densidad, G / C)
    h_nullclinas.append(
        dynamics.curva_simplex(h, densidad, 1 - densidad - h)
    )
    for nullclina in h_nullclinas:
        ax.plot(nullclina[:, 0], nullclina[:, 1], lw=3, alpha=0.9, c="r")
    d_nullclinas = []
    d = np.zeros_like(densidad)
    d_nullclinas.append(
        dynamics.curva_simplex(densidad, d, 1 - densidad - d)
    )
    d = 1 - (C / G) * densidad ** 2 
    d_nullclinas.append(
        dynamics.curva_simplex(densidad, d, 1 - densidad - d)
    )
    for nullclina in d_nullclinas:
        ax.plot(nullclina[:, 0], nullclina[:, 1], lw=3, alpha=0.9, c="green", ls=(0, (1, 0.5)))
    b_nullclinas = []
    d = densidad - (C / G) * densidad ** 2 
    b_nullclinas.append(
        dynamics.curva_simplex(densidad, d, 1 - densidad - d)
    )
    d = 1 - densidad
    b_nullclinas.append(
        dynamics.curva_simplex(densidad, d, 1 - densidad - d)
    )
    for nullclina in b_nullclinas:
        ax.plot(nullclina[:, 0], nullclina[:, 1], lw=3, alpha=0.9, c="b", ls=(0, (5, 5)))
    
    handles = [
        Line2D([0], [0], lw=3, alpha=0.9, c="r"),
        Line2D([0], [0], lw=3, alpha=0.9, c="green", ls=(0, (1, 0.5))),
        Line2D([0], [0], lw=3, alpha=0.9, c="b", ls="dashed"),
        Line2D([0], [0], ls="", c="k", marker="o", markersize=7, alpha=0.75)
    ]
    labels = [
        "Halcones",
        "Palomas",
        "Bravucones",
        "Puntos fijos\n(cálculo numérico)"
    ]
    ax.legend(handles, labels, loc="upper left", title="Nullclinas")

def nullclinas_vengativos(dynamics, G, C, ax):
    densidad = np.linspace(0, 1, 1000)
    h_nullclinas = []
    h = np.zeros_like(densidad)
    h_nullclinas.append(
        dynamics.curva_simplex(h, densidad, 1 - densidad - h)
    )
    d = C * (densidad ** 2 - 2 * densidad + 1) / (G + C - 2 * C * densidad)
    h_nullclinas.append(
        dynamics.curva_simplex(densidad, d, 1 - densidad - d)
    )
    for nullclina in h_nullclinas:
        ax.plot(nullclina[:, 0], nullclina[:, 1], lw=3, alpha=0.9, c="r")
    d_nullclinas = []
    h = np.zeros_like(densidad)
    d_nullclinas.append(
        dynamics.curva_simplex(h, densidad, 1 - densidad - h)
    )
    h = -2 * densidad + 2 - G / C
    d_nullclinas.append(
        dynamics.curva_simplex(h, densidad, 1 - densidad - h)
    )
    for nullclina in d_nullclinas:
        ax.plot(nullclina[:, 0], nullclina[:, 1], lw=3, alpha=0.9, c="green", ls=(0, (1, 0.5)))
    v_nullclinas = []
    h = np.zeros_like(densidad)
    v_nullclinas.append(
        dynamics.curva_simplex(h, densidad, 1 - densidad - h)
    )
    h = 1 - 2 * densidad
    v_nullclinas.append(
        dynamics.curva_simplex(h, densidad, 1 - densidad - h)
    )
    h = 1 - densidad
    v_nullclinas.append(
        dynamics.curva_simplex(h, densidad, 1 - densidad - h)
    )
    for nullclina in v_nullclinas:
        ax.plot(nullclina[:, 0], nullclina[:, 1], lw=3, alpha=0.9, c="b", ls=(0, (5, 5)))
    
    handles = [
        Line2D([0], [0], lw=3, alpha=0.9, c="r"),
        Line2D([0], [0], lw=3, alpha=0.9, c="green", ls=(0, (1, 0.5))),
        Line2D([0], [0], lw=3, alpha=0.9, c="b", ls="dashed"),
        Line2D([0], [0], ls="", c="grey", marker="o", markersize=7, alpha=0.75)
    ]
    
    labels = [
        "Halcones",
        "Palomas",
        "Vengativos",
        "Puntos fijos\n(numéricamente)"
    ]
    ax.legend(handles, labels, loc="upper left", title="Nullclinas")
#%%
class DinamicaReplicador():
  def __init__(self, matriz_pagos):
    self.A = matriz_pagos
  def __call__(self, x, t):
    f = self.A @ x
    f_promedio = x @ f
    return x * (f - f_promedio)

import os, sys
sys.path.append(os.path.join('egtsimplex'))
import cosa

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