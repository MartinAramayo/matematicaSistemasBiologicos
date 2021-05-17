import pylab
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from jitcdde import jitcdde, y, t
from scipy.integrate import odeint
from math import pi
# pip3 install jitcdde 

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

def axes_no_corner(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    
## Start

def logistical(y, t, r, K, A):
    dydt = r * y * (1 - y / K) * (y / A -1)
    return dydt

# interactive plots
# plt.ion()

# parametros de la ecuacion logistica
r = 1.1
K = 1
A = 0.01
t_max = 200

# the array of parameter A
array_A = np.linspace(0.1, 2, num=20)
array_A_size = array_A.size

## the array of parameters to numerically run
concat = (np.linspace(0.1, 0.9, num=1), np.asarray([1]))
array_r = np.concatenate(concat)

concat = (array_r, np.linspace(1.15, 3, num=1))
array_r = np.concatenate(concat)
array_r_size = array_r.size

## set the colormap things
# number_of_lines = array_r_size
number_of_lines = array_A_size
cm_subsection = np.linspace(0, 1, number_of_lines) 

colors = [ cm.gnuplot(x) for x in cm_subsection ]

# condicion inicial
y0 = 0.5 

# puntos de equ son A, 0 y K

for i, r in np.ndenumerate(np.flipud(array_r)): # in reverse
    fig, ax = plt.subplots()

        
    for j, A in np.ndenumerate(np.flipud(array_A)):
        # primer iteracion
        t_curva_all = np.linspace(0, t_max)
        sol_all = odeint(logistical, y0, t_curva_all, args=(r, K, A)).flatten()
                    
        # my_label = f'$r$ = {r:3.2f}'
        my_label = f'$A$ = {A:2.1f}'
        ax.plot(t_curva_all, sol_all, label=my_label, color=colors[j[0]])

    # style
    lgd = ax.legend(ncol=1, bbox_to_anchor=(1.25, 0.5), loc='center right')
    ax.set_ylabel("Población $N$")
    ax.set_xlabel("tiempo(años)")
    ax.set_title(f'$r$ = {r:3.2f}, $K$ = {K:4.3f}, $N_0$ = {y0:2.1f}')
    axes_no_corner(ax)

    fig.savefig('plots/ex6-y0chico' + f'{i[0]:02}' + '.pdf',
                format='pdf',
                bbox_extra_artists=(lgd,),
                bbox_inches='tight')

# condicion inicial
y0 = 2 
for i, r in np.ndenumerate(np.flipud(array_r)): # in reverse
    fig, ax = plt.subplots()

        
    for j, A in np.ndenumerate(np.flipud(array_A)):
        # primer iteracion
        t_curva_all = np.linspace(0, t_max)
        sol_all = odeint(logistical, y0, t_curva_all, args=(r, K, A)).flatten()
                    
        # my_label = f'$r$ = {r:3.2f}'
        my_label = f'$A$ = {A:2.1f}'
        ax.plot(t_curva_all, sol_all, label=my_label, color=colors[j[0]])

    # style
    lgd = ax.legend(ncol=1, bbox_to_anchor=(1.25, 0.5), loc='center right')
    ax.set_ylabel("Población $N$")
    ax.set_xlabel("tiempo(años)")
    ax.set_title(f'$r$ = {r:3.2f}, $K$ = {K:3.2f}, $N_0$ = {y0:2.1f}')
    axes_no_corner(ax)

    fig.savefig('plots/ex6-y0grande' + f'{i[0]:02}' + '.pdf',
                format='pdf',
                bbox_extra_artists=(lgd,),
                bbox_inches='tight')


# plt.close('all')