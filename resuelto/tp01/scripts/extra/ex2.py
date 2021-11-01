import pylab
import numpy as np
import matplotlib.pyplot as plt
from jitcdde import jitcdde, y, t
from scipy import signal
from math import pi
from matplotlib import cm
# pip3 install jitcdde 

# ## config
# # Set figure size
# SMALL_SIZE = int( 8 * 1.8)
# MEDIUM_SIZE = int(10 * 1.8)
# BIGGER_SIZE = int(12 * 1.8)

# plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
# plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
# plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
# plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
# plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# params = {
#     "font.family": "serif",
#     "text.usetex": True,
#     "pgf.rcfonts": False,
# }
# pylab.rcParams.update(params)

def axes_no_corner(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    
## Start
def set_DDE(r, K, T, N0):
    f = [ r * y(0) * (1 - (y(0, t - T)) / K) ]
    DDE = jitcdde(f)
    DDE.constant_past([N0]) # initial condition
    DDE.step_on_discontinuities() # fix
    return DDE

"""
Solucion de la ecuacion logistica con delay para 3 valores del parametro r
"""
# interactive plots
# plt.ion()
fig, ax = plt.subplots()

# Parameters
T = 1.0
K = 10.0
N0 = 2.0
r_list = [1.7, 1.2, 0.3]
for r in r_list:
    label = f'$r$ = {r:2.1f}' 
    
    DDE = set_DDE(r, K, T, N0)

    t_range = np.arange(DDE.t, DDE.t + 50, 0.01)
    data = [DDE.integrate(time)[0] for time in t_range]

    ax.plot(t_range, np.asarray(data), label=label)
    
# style
lgd = ax.legend(loc='center right')
ax.set_ylabel("Población $N$")
ax.set_xlabel("Iteración")
ax.legend(ncol=2)
axes_no_corner(ax)

fig.savefig('ex2-cosa.pdf',
            format='pdf',
            bbox_inches='tight'
            )

plt.close('all')

# """
# Solucion de la ecuacion logistica con delay 
# metodo numerico y aproximacion analitica
# """
# # Parameters
# N0 = 2.0
# K = 10.0
# r = 0.3
# π = pi
# ε = 0.01
# T = ε + π/(2 * r)

# # want to iterate different ihnitial conditions
# array_N0 = np.linspace(K*1.01, 1.1*K, 5)
# array_N0_size = array_N0.size

# ## set the colormap things
# number_of_lines = array_N0_size
# cm_subsection = np.linspace(0, 1, number_of_lines) 

# colors = [ cm.prism(x) for x in cm_subsection ]

# # plt.ion()
# fig2, ax2 = plt.subplots()

# for i, N0 in np.ndenumerate(array_N0):
#     c = N0/K - 1 
#     π = pi
#     ε = 0.00001
#     T = ε + π/(2 * r)
#     deno = 1 + π**2/4

#     label = f'$N_0$ = {N0:3.2f}' 

#     DDE = set_DDE(r, K, T, N0)
#     t0 = DDE.t # starting point of integration
    
#     # datos
#     t_range = np.arange(t0, t0 + 1000, 0.001)
#     data = [DDE.integrate(time)[0] for time in t_range]

#     # Aproximacion
#     aprox = K*(1 
#              + c 
#              * np.exp(ε * r * t_range / deno ) 
#              * np.exp( 1j * r * t_range * ( 1 - ε * π / ( 2 * deno )) )
#              )

#     ax2.plot(t_range[-45000:], 
#              np.asarray(data[-45000:]), 
#              label=label, 
#              color=colors[i[0]], 
#              linewidth=0.5, 
#              linestyle='dashed')

#     ax2.plot(t_range[-45000:], 
#              np.real(aprox)[-45000:], 
#              label=label, 
#              color = colors[i[0]])
        
# # style
# ax2.set_ylabel("Población $N$")
# ax2.set_xlabel("Iteración")
# axes_no_corner(ax2)

# box = ax2.get_position()
# ax2.set_position([box.x0, box.y0,
#                  box.width*2.1, box.height * 0.9])

# lgd2 = ax2.legend(ncol=5, 
#                   bbox_to_anchor=(0.5, -0.3),
#                   loc='center',
#                   )
# fig2.tight_layout()
# fig2.savefig('ex2-cosaPrueba.pdf',
#               format='pdf',
#               bbox_extra_artists=(lgd2,),
#               bbox_inches='tight')

# plt.close('all')

# """
# Ciclo limite a distintas amplitudes
# """

# # aprixmacion
# # Parameters
# K = 10.0
# N0 = 2.0
# r = 10.2
# π = pi
# ε = 0.01
# T = ε + π/(2 * r)

# ## set the colormap things
# array_N0 = np.linspace(K, 5*K, 5)
# number_of_lines = array_N0.size
# cm_subsection = np.linspace(0, 1, number_of_lines) 

# colors = [ cm.prism(x) for x in cm_subsection ]

# # plt.ion()
# fig2, ax2 = plt.subplots()
# for i, N0 in np.ndenumerate(array_N0):
#     label = f'$N_0$ = {int(N0)}'
    
#     DDE = set_DDE(r, K, T, N0)
#     t0 = DDE.t # starting point of integration
    
#     # datos
#     t_range = np.arange(t0, t0 + 50, 0.001)
#     data = [DDE.integrate(time)[0] for time in t_range]
    
#     ax2.plot(t_range, np.asarray(data), label=label, color=colors[i[0]], linewidth=0.5)

# box = ax2.get_position()
# ax2.set_position([box.x0, box.y0,
#                   box.width*2.2, box.height])

# # lgd2 = ax2.legend(bbox_to_anchor=(0, -0.32, 1, 0), loc="lower left", mode="expand", ncol=5)

# lgd2 = ax2.legend(
#     # bbox_to_anchor=(0.5, -0.3, 1.5, .102),
#     bbox_to_anchor=(0.5, -0.23),
#     # borderaxespad=0.01,
#     loc="center",
#     # loc="center",
#     # mode="expand",      
#     ncol=5,
# )

# ax2.set_xlim([0,8])
# ax2.set_title(f'$r$ = {r:2.1f}, $T$ = {T:2.1f}')
# axes_no_corner(ax2)

# fig2.tight_layout()

# ax2.set_ylabel("Población $N$")
# ax2.set_xlabel("Iteración")
# fig2.savefig('ex2-aprox.pdf',
#              format='pdf',
#              bbox_extra_artists=(lgd2,),
#              bbox_inches='tight'
#              )

# plt.close('all')

# """
# fourier
# """

# ## r independent

# ## set the colormap things
# array_r = np.linspace(K, (1.25)*K, 5)
# number_of_lines = array_N0.size
# cm_subsection = np.linspace(0, 1, number_of_lines) 

# colors = [ cm.prism(x) for x in cm_subsection ]
    
# # aprixmacion
# # Parameters
# K = 10.0
# N0 = 2.0
# π = pi
# ε = 0.01

# T = 0.2
# Tc = ε + π/(2 * r)

# # plt.ion()

# fig3, ax3 = plt.subplots()
# fig4, ax4 = plt.subplots()
# fig5, ax5 = plt.subplots()
# for i, r in np.ndenumerate(array_r):
    
#     T = π/(2 * r)
    
#     label = f'$r$ = {r:2.1f}'
    
#     DDE = set_DDE(r, K, T, N0)
#     t0 = DDE.t # starting point of integration
    
#     # datos
#     t_range = np.arange(t0, t0 + 20, 0.001)
#     data = [DDE.integrate(time)[0] for time in t_range]
    
#     ## la oscilacion
#     t_2tf = t_range[-10000:] 
#     d_2tf = data[-10000:]
#     d_2tf = d_2tf - np.mean(d_2tf)

#     ## Fourier transform 
#     f, Pxx_den = signal.periodogram(d_2tf, 1 / (t_range[1] - t_range[0]) )  
        
#     frec_real_index = np.where( Pxx_den == np.max(Pxx_den) )
#     real_frec = float(f[frec_real_index])
    
#     ax4.semilogy(f, Pxx_den, label=label, color=colors[i[0]], linewidth=0.5)
    
#     ax3.plot(f, Pxx_den, label=label, color=colors[i[0]]) # fourier transform

#     # style periodogram
#     ylim = ax3.get_ylim() 
#     ax3.axvline(1/(4*T), linestyle='dashed', color=colors[i[0]])
#     ax3.text(1/(4*T)*(1 - 0.015), 10, 
#              r"$\frac{1}{4T}$" + r"$=\frac{1}" + f"{{{4*T:4.3f}}}$", 
#              rotation='vertical')

#     ax3.axvline(real_frec, color=colors[i[0]])
#     str_text = f"$\frac{{{float(1/real_frec):4.3f}}}{{4T}}$"
#     ax3.text(real_frec * (1 + 0.005), 10, 
#             r"$\frac{1}" + f"{{{1/real_frec:4.3f}}}$",
#             rotation='vertical')

# ax3.set_xlim([1.5, 2.1])
# ax3.set_ylabel("Amplitud")
# ax3.set_xlabel("Frecuencia")
# axes_no_corner(ax3)

# box = ax3.get_position()
# ax3.set_position([box.x0, box.y0 + box.height * 0.1,
#                  box.width*1.4, box.height * 0.9])

# lgd3 = ax3.legend(
#     bbox_to_anchor=(0, -0.3, 1, .102),
#     loc="lower left",                
#     ncol=5,                 
#     mode="expand",                 
#     borderaxespad=0.01,
# )

# fig3.savefig('ex2-fourier-acotado.pdf',
#             format='pdf',
#             bbox_extra_artists=(lgd3,),
#             bbox_inches='tight')

# # style periodogram
# lgd4 = ax4.legend(
#     loc="upper center",                
# )

# ax4.set_ylim(bottom=1e-7)
# ax4.set_ylabel("Amplitud")
# ax4.set_xlabel("Frecuencia")
# axes_no_corner(ax4)
# fig4.savefig('ex2-fourier-log.pdf',
#              format='pdf',
#              bbox_extra_artists=(lgd4,),
#              bbox_inches='tight')

# plt.close('all')
