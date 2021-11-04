from scipy.integrate import solve_ivp
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns 
import pylab
import pandas as pd

from numpy.random import exponential, uniform

#%%
def aPhi(t, b, d, phi_0):  
    # r = b - d
    r = b
    s = d
    c = s - r / phi_0
    return r / (s - c * np.exp( -r * t))
#%%       
def tuple_generar(t, b, d, N0, N_steps):
    counter = N_steps
    N = N0
    yield (t, N)  # first iteration
    
    while N>0 and counter > 0 and N<1e4 or a_0==0:
        h_1, h_2 = N, N**2 # ? si no hago esto no anda je
     
        a_1, a_2 = h_1 * b, h_2 * d
        a_0 = a_1 + a_2
        
        t += exponential(1/a_0)

        N = N + 1 if uniform(0, a_0) <= a_1 else N - 1
        
        counter -= 1
        yield (t, N)
        
N_simulaciones = 1000

aux = {'b': 0.1,
       'd': 0.001,
       'phi_0': 20,
       'omega': 1,
       'N_steps': 4000}

aux_1 = {'b': aux['b'],
         'd': aux['d'],
         'N0': aux['phi_0'],
        #  'phi_0': 0.2,
        #  'omega': 100,
         'N_steps': aux['N_steps']}
#####################################################
def simulation_histogram(data, n_simulacion, names, columns):
    size = len(data)
    iterator = zip([n_simulacion] * size, [*range(size)])
    tuple_index = [*iterator]
    index = pd.MultiIndex.from_tuples(tuple_index, names=names)
    aux_df = pd.DataFrame(data, columns=columns, index=index)
    return aux_df
#####################################################

names = ["n_simulacion", "indice"]
columns = ["tiempo", "N"]

simulaciones_df = pd.DataFrame()
for n_simulacion in range(N_simulaciones):
    # iterador = tuple_generar(0,**aux)
    iterador = tuple_generar(0,**aux_1)
    
    data = list(iterador)
      
    aux_df = simulation_histogram(data, n_simulacion, names, columns)
    simulaciones_df = simulaciones_df.append(aux_df)

######################################### Plot histogram
fig, ax = plt.subplots()
    
aux_args = {'data': simulaciones_df, 
            'x': 'tiempo',
            'y': 'N',
            'stat': 'density',
            'discrete': (False, True), 
            'rasterized': True,
            'cbar': True, 
            'cbar_kws': {'label': 'Densidad'},
            'ax': ax}    
sns.histplot(**aux_args)

fig.tight_layout()
ax.set_xlabel('$t$')
ax.set_ylabel('$x(t)$')

######################################### plot theorical thing
b, d, phi_0, *_ = *args, omega, N_steps = aux.values()
t_max = simulaciones_df.tiempo.max() * 0.8
t = np.linspace(0, t_max, N_steps)

# no entiendo como hago que omega 
# infiera en el sistema, lo deje como 1 y deje phi_0 = N0

# La cosa logistica
macro = omega * aPhi(t, *args)

# fluctuaciones asintoticas
fluct = omega * phi_0 * d / (2 * b)
aux_list = [macro * ( 1 - fluct), macro * ( 1 + fluct)]
fluctuaciones = np.transpose(aux_list)

# test
asint = np.full_like(fluctuaciones, fluct)

##############################

aux_str = r"Ley macro $\Omega \phi (t)$"
ax.plot(t, macro, c="k", label=aux_str)

# fluctuaciones asintoticas
fluct = omega * phi_0 * d / (2 * b)
aux_list = [macro * ( 1 - fluct), macro * ( 1 + fluct)]
fluctuaciones = np.transpose(aux_list)

# aux_str = r"Fluct. Asint. $\sqrt{\Omega \langle \xi^2 \rangle_{t \to \infty}}$"
# ax.plot(t, fluctuaciones[:,0], c="k", ls=":", label=aux_str)
# ax.plot(t, fluctuaciones[:,1], c="k", ls=":")
ax.legend(loc='lower right')

ax.set_xlabel(r"Tiempo $t$")
ax.set_ylabel("Población")

ax.set_title(rf"{N_simulaciones} simulaciones, " 
             + rf"$b= $ {b}, " 
             + rf"$d= $ {d}, "
            #  + rf"$\phi_0= $ {phi_0}, "
            #  + rf"$\Omega= $ {omega}"
             )
fig.tight_layout()
fig.savefig('../figuras/ex03-a-SinCota.pdf')

######################################### asintotico 
fig, ax = plt.subplots()

bool_index = ((simulaciones_df.tiempo < t_max) 
              & (simulaciones_df.tiempo > t_max*0.9999))
n_limit = simulaciones_df[bool_index].N.values
aux_args = {'data': n_limit, 
            'stat': 'density',
            'discrete': True, 
            'cbar': True, 
            'color': 'royalblue',
            'ax': ax}    
sns.histplot(**aux_args)

hist, bins = np.histogram(n_limit, density=True)

promedio = np.mean(n_limit)
std = np.std(n_limit)
ax.vlines(
    promedio, 0, hist.max(), 
    color="k", 
    label="Promedio"
)
ax.vlines(
    [promedio - std, promedio + std], 0, hist.max(),
    color="k", 
    ls="dashed", 
    label="Desviación estándar"
)
ax.vlines(
    macro[-1], 0, hist.max(), 
    color='salmon',
    label=r"Ley macro $\Omega \phi (t\to\infty)$"
)
# ax.vlines(
#     [macro[-1] - asint[-1], macro[-1] + asint[-1]], 0, hist.max(),
#     ls="dashed", 
#     color='salmon',
#     label=r"Fluct. Asint. $\sqrt{\Omega \langle \xi^2 \rangle_{t \to \infty}}$"
# )

ax.legend(loc='best')
ax.set_xlabel('$N_A$')
ax.set_ylabel('Densidad')
b, d, phi_0 = args
ax.set_title(rf"{N_simulaciones} simulaciones, " 
             + rf"$b= $ {b}, " 
             + rf"$d= $ {d}, "
             + rf"$\phi_0= $ {phi_0}, "
             + rf"$\Omega= $ {omega}, "
             + rf"{t_max * 0.9:3.1f} $< t <$ {t_max:3.1f}")
fig.tight_layout()
fig.savefig('../figuras/ex03-b-SinCota.pdf')