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

def logistical(y, t, r, K):
    dydt = r * y * (1 - y / K) 
    return dydt

# interactive plots
# plt.ion()


# parametros de desastre
# λ = 0.1
n_desastres = 50
p = 0.5 

# parametros de la ecuacion logistica
r = 1.01
K = 1

## the array of parameters to numerically run
array_lambda = np.linspace(0.001, 4, num=20)

## the array of parameters to numerically run
array_p = np.linspace(0, 1, num=20)
array_p_size = array_p.size

## set the colormap things
number_of_lines = array_p_size
cm_subsection = np.linspace(0, 1, number_of_lines) 

colors = [ cm.gnuplot(x) for x in cm_subsection ]

# condicion inicial
y0 = 0.1 


for j, λ in np.ndenumerate(array_lambda):
    fig, ax = plt.subplots()

    # primer desastre
    t_caos_0 = np.random.exponential(scale=1/λ, size=n_desastres)[0]
    # secuencia de desastres
    t_caos_array = np.random.exponential(scale=1/λ, size=n_desastres)
    for i, p in np.ndenumerate(np.flipud(array_p)): # in reverse
        
        # primer iteracion
        t_curva_all = np.linspace(0, t_caos_0)
        sol_all = odeint(logistical, y0, t_curva_all, args=(r, K)).flatten()

        for t_caos in np.nditer(t_caos_array):
            
            t_last, y_last = t_curva_all[-1], sol_all[-1]
            
            t_curva = np.linspace(t_last, t_last + t_caos)
            sol = odeint(logistical, p*y_last, t_curva, args=(r, K)).flatten()
            
            t_curva_all = np.concatenate((t_curva_all, t_curva))
            sol_all = np.concatenate((sol_all, sol))
            
        my_label = f'$p$ = {p:3.2f}'
        ax.plot(t_curva_all, sol_all, label=my_label, color=colors[i[0]])

    # style
    lgd = ax.legend(ncol=1, bbox_to_anchor=(1.25, 0.5), loc='center right')
    ax.set_ylabel("Población $N$")
    ax.set_xlabel("tiempo(años)")
    ax.set_title(f'$r$ = {r:3.2f}, $\lambda$ = {λ:4.3f}, $N_0$ = {y0:4.3f}')
    axes_no_corner(ax)

    fig.savefig('plots/ex5-' + f'{j[0]:02}' + '.pdf',
                format='pdf',
                bbox_extra_artists=(lgd,),
                bbox_inches='tight')

# plt.close('all')