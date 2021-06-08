import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
#%%
#@title Ejecutar para cargar nullclinas
#@markdown Fórmulas analíticas de las nullclinas.
from matplotlib.lines import Line2D

def nullclinas_vengativos(dynamics, G, C, ax):
    densidad = np.linspace(0, 1, 1000)
    
    h = np.zeros_like(densidad)
    
    aux = dynamics.curva_simplex(h, densidad, 1 - densidad - h)
    h_nullclinas = [aux]
    
    d = C * (densidad ** 2 - 2 * densidad + 1) / (G + C - 2 * C * densidad)
    
    aux = dynamics.curva_simplex(d, densidad, 1 - densidad - d)
    h_nullclinas.append(aux)
    
    for nullclina in h_nullclinas:
        ax.plot(nullclina[:, 0], nullclina[:, 1], lw=3, alpha=0.9, c="r")
    
    d_nullclinas = []
    h = np.zeros_like(densidad)
    
    aux = dynamics.curva_simplex(h, densidad, 1 - densidad - h)
    d_nullclinas.append(aux)
    
    h = -2 * densidad + 2 - G / C
    
    aux = dynamics.curva_simplex(h, densidad, 1 - densidad - h)
    d_nullclinas.append(aux)
    
    for nullclina in d_nullclinas:
        ax.plot(nullclina[:, 0], nullclina[:, 1], lw=3, alpha=0.9, c="green", ls=(0, (1, 0.5)))
    
    v_nullclinas = []
    h = np.zeros_like(densidad)
    
    aux = dynamics.curva_simplex(h, densidad, 1 - densidad - h)
    v_nullclinas.append(aux)
    
    h = 1 - 2 * densidad
    
    aux = dynamics.curva_simplex(h, densidad, 1 - densidad - h)
    v_nullclinas.append(aux)
    
    h = 1 - densidad
    
    aux = dynamics.curva_simplex(h, densidad, 1 - densidad - h)
    v_nullclinas.append(aux)
    
    for nullclina in v_nullclinas:
        ax.plot(nullclina[:, 0], nullclina[:, 1], lw=3, alpha=0.9, c="b", ls=(0, (5, 5)))
    
    handles = [
        Line2D([0], [0], lw=3, alpha=0.9, c="r"),
        Line2D([0], [0], lw=3, alpha=0.9, c="green", ls=(0, (1, 0.5))),
        Line2D([0], [0], lw=3, alpha=0.9, c="b", ls="dashed"),
        Line2D([0], [0], ls="", c="grey", marker="o", markersize=7, alpha=0.75)
    ]
    
    labels = ["Halcones",
              "Palomas",
              "Vengativos",
              "Puntos fijos\n(numéricamente)"]
    ax.legend(handles, 
              labels, 
              loc="upper left", 
              title="Nullclinas") 
