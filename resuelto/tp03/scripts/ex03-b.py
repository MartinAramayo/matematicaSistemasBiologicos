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
# throw a p weighted coin
def random_p_binary_choice(p):
    return np.random.choice([0,1], size=1, p=(1-p, p))[0]

def simular(N, b, d):
    return N+1 if random_p_binary_choice(b/(b+d)) else N-1

def tiempo(t, b, d):
    aLambda = (b + d)
    return t + np.random.exponential(scale=1/aLambda, size=1)[0]

def tuple_generar(t, N, b, d, N_steps):
    counter = N_steps
    while N>0 and counter > 0 and N<1e4:
        t = tiempo(t, b, d)
        N = simular(N, b, d)
        counter -= 1
        yield (t, N)
        
args = {'b_nacer': 0.1,
        'd_morir': 0.1,
        # 'd_morir': 0.09,
        # 'd_morir': 0.001,
        'N0': 100,
        'N_steps': 1000,
        'N_simulaciones': 50,
        'savefile': '../figuras/ex03-b-histograma-01.pdf'}
#####################################################
def simulation_histogram(data, n_simulacion, names, columns):
    tuple_index = list(zip(
                           [n_simulacion] * len(data), 
                           tuple(range(len(data)))
                           )
                       )
    index = pd.MultiIndex.from_tuples(tuple_index, names=names)
    aux_df = pd.DataFrame(data, columns=columns, index=index)
    return aux_df
#####################################################

names = ["n_simulacion", "indice"]
columns = ["tiempo", "N"]

aux = {'b': args['b_nacer'], 'd': args['d_morir'], 'N_steps': args['N_steps']}
simulaciones_df = pd.DataFrame()
for n_simulacion in range(args['N_simulaciones']):
    generador = tuple_generar(0, args['N0'],**aux)
    
    data = list(generador)
    data.insert(0, (0, args['N0'])) # insert initial condition
      
    aux_df = simulation_histogram(data, n_simulacion, names, columns)
    simulaciones_df = simulaciones_df.append(aux_df)

fig, ax = plt.subplots()
    
sns.histplot(simulaciones_df, 
             x='tiempo',
             y='N',
             stat='density',
             discrete=(False, True), 
             cbar=True, 
             ax=ax)

fig.tight_layout()
ax.set_xlabel('$t$')
ax.set_ylabel('$x(t)$')
# ax.set_yscale('symlog', base=10)
fig.savefig(args['savefile'])