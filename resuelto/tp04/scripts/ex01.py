import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib import cm
from numpy import linspace

def g_rep(x, a, b, c, h):
    return a / ( b + c * (x**h) ) 

def model_goodwin(y, t, am, ae, ap, bm, be, bp, a, b, c, h):
    m,e,p = y
    dmdt = am * g_rep(p,a,b,c,h) - bm * m
    dedt = ae * m - be * e
    dpdt = ap * e - bp * p
    return dmdt, dedt, dpdt

def plot_run(time, parameters, parameter_h_array ,savefile):
    
    fig = plt.figure(figsize=(6,6))
    gs = fig.add_gridspec(3, 1, hspace=0, wspace=0)
    ax = gs.subplots(sharex='col')
    
    ## colormap
    start, stop = 0.0, 1
    number_of_lines = len(parameter_h_array)
    color_slice = linspace(start, stop, number_of_lines+1) 

    colors1 = [ cm.summer(x) for x in color_slice ]
    colors2 = [ cm.autumn(x) for x in color_slice ]
    colors3 = [ cm.winter(x) for x in color_slice ]

    for i, h in np.ndenumerate(parameter_h_array):
        i = i[0]
        
        sol = odeint(model_goodwin, y0, t, args=(*parameters, h))
        m, e, p = sol.T 
        
        ax[0].plot(t, m, label=f'$h=${h:3.1f}', color=colors1[i])
        ax[1].plot(t, e, label=f'$h=${h:3.1f}', color=colors2[i])
        ax[2].plot(t, p, label=f'$h=${h:3.1f}', color=colors3[i])

    ax[0].set_ylabel('$m$')
    ax[1].set_ylabel('$e$')
    ax[2].set_ylabel('$p$')

    ax[2].set_xlabel('$t$')

    ax[0].tick_params(bottom=False)
    ax[1].tick_params(bottom=False)

    for i in range(len(ax)): ax[i].legend(ncol=4, fontsize=9)

    fig.tight_layout()
    fig.savefig(savefile)

aDir = "../figuras/" 
#############################################################
## dinamica a disintos h
t = np.linspace(0,500,10000)

# parametros
y0 = 0.1, 0, 0
am = ae = ap = 1
bm = be = bp = 0.1
a = b = c = 1
params = (am, ae, ap, bm, be, bp, a, b, c)

# arreglo de parametros h
all_h = np.linspace(1, 40, 4)

plot_run(t, params, all_h, aDir + 'ex01-concentracion-h.pdf')
#############################################################
## Oscilaciones

bm = be = bp = 0.5
params = (am, ae, ap, bm, be, bp, a, b, c)

all_h = np.linspace(10, 10, 1)

plot_run(t, params, all_h, aDir + 'ex01-concentracion-h-osc.pdf')
#############################################################
## Matar oscilaciones

bm = be = bp = 0.8
params = (am, ae, ap, bm, be, bp, a, b, c)

all_h = np.linspace(10, 10, 1)

plot_run(t, params, all_h, aDir + 'ex01-concentracion-h-osc-kill.pdf')
####################################################################
# x = np.linspace(0,10,100)
# vec_h = [2,4,6,8,10]
# fig, ax = plt.subplots(figsize=(10,8))
# for i in vec_h:
#     y = g_rep(x,1,1,1,i)
#     ax.plot(x,y,label='$h =$'+str(i),linewidth=2)
# ax.legend(fontsize=12)
# ax.set_xlabel('$p$',fontsize=14)
# ax.set_ylabel('$g_{R} (p)$',fontsize=14)