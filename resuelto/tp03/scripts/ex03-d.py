from scipy.integrate import solve_ivp
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns 
import pylab
import pandas as pd
#%%
def plot_household(fig, ax, filename):
    ax.legend()
    ax.set_ylabel("$x(t)$")
    ax.set_xlabel("$t$")
    fig.tight_layout()
    fig.savefig(filename)
#%%
def phi_parameters(b, d, phi_0):
    r = 2 * b - d
    s = 2 * b
    c = s - r / phi_0
    return r, s, c
              
def aPhi(t, b, d, phi_0):
    s = 2 * b
    r = s - d
    c = s - r / phi_0
    return r / (s - c * np.exp( -r * t))

def F_1(t, b, d, phi_0):
    return - (d - 2 * b * (1 - 2 * aPhi(t, b, d, phi_0) ) )

def F_2(t, b, d, phi_0):
    phi_value = aPhi(t, b, d, phi_0) 
    return phi_value * (d + 2 * b * (1 - phi_value))

def fokker_planck(t, varianza, b , d, phi_0):
    args = t, b, d, phi_0
    return 2 * F_1(*args) * varianza + F_2(*args)
#%%       
def random_gillespie(N, a_0, vacio, b):
    return (np.random.uniform(size=1)[0] * a_0 < vacio * N * b)

def simular(N, a_0, vacio, b):
    return N+1 if random_gillespie(N, a_0, vacio, b) else N-1

def tiempo(t, a_0):
    w = np.random.uniform(size=1)[0]
    tau = -np.log(w) / a_0
    return t + tau

def tuple_generar(t, b, d, phi_0, omega, N_steps):
    counter = N_steps
    
    # auxiliar variables
    b = 2 * b / omega
    N = int(phi_0 * omega + 0.5)
    vacio = omega - N
    a_0 = N * (vacio * b + d)
    yield (t, N)  # first iteration
    
    while N>0 and counter > 0 and N<1e4 or a_0==0:
        
        vacio = omega - N
        a_0 = N * (vacio * b + d)
        
        t = tiempo(t, a_0)
        N = simular(N, a_0, vacio, b)
        
        counter -= 1
        yield (t, N)
        
N_simulaciones = 1000

aux = {'b': 0.1,
       'd': 0.3,
       'phi_0': 0.2,
       'omega': 100,
       'N_steps': 700}
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
    iterador = tuple_generar(0,**aux)
    
    data = list(iterador)
      
    aux_df = simulation_histogram(data, n_simulacion, names, columns)
    simulaciones_df = simulaciones_df.append(aux_df)

######################################### plot theorical thing
*args, omega, N_steps = aux.values()
t_max = simulaciones_df.tiempo.max() * 0.85
t = np.linspace(0, t_max, N_steps)

simulation = solve_ivp(fun=fokker_planck,
                       t_span=[0, t_max],
                       y0=[0],
                       args=args,
                       method="RK45",
                       dense_output=True)
varianza = simulation.sol(t)[0]
macro = omega * aPhi(t, *args)
fluct = np.sqrt(omega * varianza)
asint = np.full_like(fluct, fluct[-1])
#########################################
fig, ax = plt.subplots()

bool_index = (simulaciones_df.N == 2)
t_death = simulaciones_df[bool_index].tiempo.values
aux_args = {'data': t_death, 
            'stat': 'density',
            'discrete': True, 
            'cbar': True, 
            'color': 'royalblue',
            'ax': ax}    
sns.histplot(**aux_args)

hist, bins = np.histogram(t_death, density=True)

promedio = np.mean(t_death)
std = np.std(t_death)
ax.vlines(
    promedio, 0, hist.max(), 
    color="k", 
    label="Promedio"
)

alpha = 2
phi_r, phi_s, phi_c = phi_parameters(*args)
t_alpha = -1 / phi_r * np.log((phi_s - omega * phi_r / alpha) / phi_c)

ax.vlines(
    t_alpha,
    0, hist.max(), 
    label="Ley macroscópica",
    ls="dashed", 
    alpha=0.7
)

ax.set_xlabel('$t$')
ax.set_ylabel('Densidad')
ax.legend()
ax.set_title("Tiempo en el que la población es de solamente 2 individuos")

fig.tight_layout()
fig.savefig('../figuras/ex03-d.pdf')