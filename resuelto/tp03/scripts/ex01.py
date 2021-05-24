import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns 
import pylab

"""
Benchmark (hyperfine):
Benchmark #1: python ex01.py
  Time (mean ± σ):     64.944 s ±  2.208 s    [User: 65.449 s, System: 0.577 s]
  Range (min … max):   62.140 s … 69.651 s    10 runs
  
Hardware (neofetch):
██████████████████  ████████   sauer@hal
██████████████████  ████████   ---------
██████████████████  ████████   OS: Manjaro Linux x86_64
██████████████████  ████████   Host: HP Laptop 14-dk1xxx
████████            ████████   Kernel: 5.9.16-1-MANJARO
████████  ████████  ████████   Uptime: 2 hours, 44 mins
████████  ████████  ████████   Packages: 1702 (pacman), 7 (flatpak)
████████  ████████  ████████   Shell: bash 5.1.8
████████  ████████  ████████   Resolution: 1600x900
████████  ████████  ████████   DE: Plasma 5.21.5
████████  ████████  ████████   WM: KWin
████████  ████████  ████████   WM Theme: Oxygen
████████  ████████  ████████   Theme: Breeze Dark [Plasma], Breath-Dark [GTK2/3]
████████  ████████  ████████   Icons: breath2-dark [Plasma], breath2-dark [GTK2/3]
                               Terminal: alacritty
                               CPU: AMD Ryzen 3 3250U with Radeon Graphics (4) @ 2.600GHz
                               GPU: AMD ATI 04:00.0 Picasso
                               Memory: 5434MiB / 13971MiB
"""

# plt.ion()
#%%
## config
# Set figure size
SMALL_SIZE = int( 8 * 1.5)
MEDIUM_SIZE = int(10 * 1.5)
BIGGER_SIZE = int(12 * 1.5)

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

params = {
    "font.family": "serif",
    "text.usetex": True,
    "pgf.rcfonts": False,
}
pylab.rcParams.update(params)

#%%
def axes_no_corner(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

# throw a p weighted coin
def random_p_binary_choice(p):
    return np.random.choice([0,1], size=1, p=(1-p, p))[0]

def num_deaths(p, total_people): # numero de 1 generados al azar con
    return np.sum(np.random.choice([0,1], size=total_people, p=(1-p, p)))

def bar_year(fig, ax, savefile):
    ax.legend()
    ax.set_xlabel("Población")
    axes_no_corner(ax)
    fig.tight_layout()
    fig.savefig(savefile)
#%%
n_poblacion = 1000
n_experimentos = 10000
simulacion = np.ones(n_experimentos,dtype=int) * n_poblacion
tiempo = 100
death_prob = 0.01

all_simulations = {}

for t in range(tiempo):
    iterable = (num_deaths(death_prob, N) for N in simulacion)
    num_muertos = np.fromiter(iterable, dtype=int, count=len(simulacion))
    simulacion -= num_muertos    
    all_simulations.update({t:deepcopy(simulacion)})
    
# colormap
num_iterations = tiempo
cm_subsection = np.linspace(0, 1, num_iterations) 
colors = tuple( cm.gnuplot(x) for x in cm_subsection )    

fig_bar_mpl, ax_bar_mpl = plt.subplots()
num_plots = 10
iterador_plot = range(1, num_iterations, num_iterations//num_plots)
for step in iterador_plot:    
    array = all_simulations[step]
    start, end, binwidth = 0, max(array), 5
    aux_args = {
        'data': array,
        'kde': True,
        'stat': 'density',
        'bins': range(start, end + binwidth, binwidth),
        'label': f'Paso Temporal = {step}',
        'color': colors[step],
        'ax': ax_bar_mpl
    }
    sns.histplot(**aux_args)

xlim = [all_simulations[tiempo-1][0] - 2*binwidth, 
        all_simulations[0][-1] + 2*binwidth]
ax_bar_mpl.set_xlim(xlim)
bar_year(fig_bar_mpl, ax_bar_mpl, '../figuras/evolucion_temporal.pdf')
#%%
from scipy.stats import binom
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

array = all_simulations[t]
start, end, binwidth = 0, max(array), 5
aux_args = {
    'data': array,
    'stat': 'density',
    'bins': range(start, end + binwidth, binwidth),
    'label': f'Paso Temporal = {t}',
    'color': colors[t],
    'ax': ax
}
sns.histplot(**aux_args)

# Distribucion
mean = array.mean()
var = array.var()

x = np.arange(array.min(), array.max())
p = 1 - (var / mean)
n = mean / p
prob = binom.pmf(x, n, p)
ax.plot(x, prob, label="Distribución binomial")

# plotting
xlim = [x[ 0] - 2*binwidth, 
        x[-1] + 2*binwidth]
ax.set_xlim(xlim)
bar_year(fig, ax, '../figuras/ultima_iteracion.pdf')