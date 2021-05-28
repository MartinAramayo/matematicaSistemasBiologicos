import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns 
import pylab
import pandas as pd
"""
Benchmark (hyperfine):
Benchmark #1: python ex03-a.py
  Time (mean ± σ):      3.980 s ±  0.451 s    [User: 3.124 s, System: 0.797 s]
  Range (min … max):    3.562 s …  4.719 s    10 runs
  
Hardware (neofetch):
Kernel: 5.9.16-1-MANJARO
CPU: AMD Ryzen 3 3250U with Radeon Graphics (4) @ 2.600GHz
GPU: AMD ATI 04:00.0 Picasso
Memory: 5434MiB / 13971MiB
"""
# plt.ion()
#%%
def plot_household(fig, ax, filename):
    ax.legend()
    ax.set_ylabel("$x(t)$")
    ax.set_xlabel("$t$")
    fig.tight_layout()
    fig.savefig(filename)
#%%
def bh_f(n, r, K):
    """Funcion de Bevertoh-Holt"""
    return r * n / (1 + n * ( (r - 1) / K) )

class mapBH:
    def __init__(self,n0,r,K):
        self.current = n0
        self.r = r
        self.K = K
    
    def __iter__(self):
        return self

    def __next__(self):
        n = self.current
        self.current = bh_f(n, self.r, self.K)
        return n

def bh_map(n0, r, K, n_steps=20):
    """Mapeo de Beverton-Holt."""
    t = np.arange(n_steps)
    
    map = mapBH(n0, r, K)
    map = iter(map)
    n_t = [next(map) for _ in range(n_steps)]
    return t, n_t

def plot_map(ax, n0, r, K, n_steps, color=None):
    # label = f'$K$ = {K:3.2f}' 
    label = f'$r$ = {r:3.2f}' 
    t, n_t = bh_map(n0, r, K, n_steps)
    # ax.scatter(t, n_t, label=label, color=color, s=20)
    ax.plot(t, n_t, label=label, color=color)
    return t, n_t

def bh_f(n, r, K):
    """Funcion de Bevertoh-Holt"""
    return r * n / (1 + n * ( (r - 1) / K) )

def tuple_generar(t, N, b, d, N_steps):
    counter = N_steps
    iter_limit = 2**40
    while N>0 and counter > 0 and N<iter_limit:
        t = tiempo(t, b, d)
        N = simular(N, b, d)
        counter -= 1
        yield (t, N)
        
        
fig, ax = plt.subplots()
b_nacer = 0.1
d_morir = 0.001
N0 = 10

s = 2 * b_nacer 
r = 2 * b_nacer - d_morir
def macro_logistic(t, r, s, y0):
    """Funcion de Bevertoh-Holt"""
    return r  / (s - (s - r/y0) * np.exp(-r * t) )

def plot_map(ax, n0, r, s, n_steps, color=None):
    # label = f'$K$ = {K:3.2f}' 
    label = f'$r$ = {r:3.2f}' 
    t = np.arange(n_steps)
    n_t = macro_logistic(t, r, s, n0)
    # ax.scatter(t, n_t, label=label, color=color, s=20)
    ax.plot(t, n_t, label=label, color=color)
    return t, n_t

n_steps = 100

r_floor = r
for N0 in np.linspace(50, 500, num=10):
# for r in np.linspace(r_floor*(1.1), r_floor*(0.9), num=10):
    plot_map(ax, N0, r, s, n_steps, color=None)

plot_household(fig, ax, '../figuras/test.pdf')