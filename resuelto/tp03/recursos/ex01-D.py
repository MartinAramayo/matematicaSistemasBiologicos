#importo librerías que me van a ser útiles en hacer las simulaciones
import numpy as np
import matplotlib.pyplot as plt
from numpy import random
import pylab
import scipy, scipy.stats
plt.style.use('seaborn')

plt.ion()

import pylab
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

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

#%%
def time_evolution(n,d,tf):
    #inicializo el vector para la poblacion x(t) desde t=0 hasta t=tf
    x = np.zeros(shape=tf+1,dtype=int)
    #inicialmente, hay n individuos
    x[0] = n 
    #pasos iteración temporal
    for i in range(tf):
        #pasos iteración en individuos, veo si se mueren o no
        x[i+1] = np.copy(x[i]) #inicialmente, tiene igual que el paso anterior
        for j in range(x[i]):
            #genero un número random
            n_random = random.random()
            if (n_random <= d): #si el número es <=d
                x[i+1] = x[i+1] - 1 #entonces el individuo muere
    return x

#%%
N = 1000 #tamaño de la población, lo dejo fijo en 1000 individuos
d = 0.25 #valores de d que utilizamos
# tf = 50 #tiempo final de la simulación 
tf = 10 #tiempo final de la simulación 
rep = 10000 #cantidad de simulaciones independientes que haremos
pop = np.zeros(shape=(rep,tf+1)) #donde guardaremos las simulaciones
                                 #hay rep filas, cada una corresponde a una simulación
                                 #hay tf+1 columnas, que corresponden a los pasos temporales
#hacemos las repeticiones y guardamos los resultados en la matriz rep, para un 
for i in range(rep):
    pop[i,:] = time_evolution(N,d,tf)

# #%%
# print('Distribución de probabilidad $P(n,t)$ para d= '+str(d)+'y t = 10')
# plt.hist(pop[:,10],density=True)
# plt.xlabel('n')
# plt.ylabel('$P(n,t=10)$')


#%%
d = [0,0.25,0.5,0.75,1]
tf = 50
N = 1000
rep = 10000
for j in d:
    pop = np.zeros(shape=(rep,tf+1))
    for i in range(rep):
        pop[i,:] = time_evolution(N,j,tf)
    # fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4,figsize=(20,5))
    # fig.suptitle('Distribuciones de probabilidad para d = '+str(j)+' y diferentes instantes de tiempo t')

    # ax1.hist(pop[:,0],density=True,label='$t = 0$  \n $\mu$ = '+str(round(np.mean(pop[:,0]),2))+'\n $\sigma$ = '+str(round(np.std(pop[:,0]),2)))
    # ax2.hist(pop[:,1],density=True,label='$t = 1$ \n $\mu$ = '+str(round(np.mean(pop[:,1]),2))+'\n $\sigma$ = '+str(round(np.std(pop[:,1]),2)))
    # ax3.hist(pop[:,20],density=True,label='$t = 20$ \n $\mu$ = '+str(round(np.mean(pop[:,20]),2))+'\n $\sigma$ = '+str(round(np.std(pop[:,20]),2)))
    # ax4.hist(pop[:,50],density=True,label='$t = 50$ \n $\mu$ = '+str(round(np.mean(pop[:,50]),2))+'\n $\sigma$ = '+str(round(np.std(pop[:,50]),2)))

    # fig.tight_layout(pad=3.0)

    # ax1.legend()
    # ax1.set_xlabel('$n$')
    # ax1.set_ylabel('$P(n,0)$')

    # ax2.legend()
    # ax2.set_xlabel('$n$')
    # ax2.set_ylabel('$P(n,1)$')

    # ax3.legend()
    # ax3.set_xlabel('$n$')
    # ax3.set_ylabel('$P(n,20)$')

    # ax4.legend()
    # ax4.set_xlabel('$n$')
    # ax4.set_ylabel('$P(n,50)$')
# fig.savefig("../figuras/cosa1.pdf")
# plt.close('all')

# #%%
# d = [0.25,0.5,0.75]
# tf = 1
# N = 1000
# rep = 10000
# for j in d:
#     pop = np.zeros(shape=(rep,tf+1))        
#     for i in range(rep):
#         pop[i,:] = time_evolution(N,j,tf)
#     plt.hist(pop[:,1],density=True,label='Simulación')
#     mean_pop = np.mean(pop[:,1])
#     std_pop = np.std(pop[:,1])
    
#     #imprimo por consola todas las estadisticas
#     print('Valor de d=',j)
#     print('Datos de la distribución calculada:')
#     print('Valor Medio = ',round(mean_pop,2),' -  Para la binomial el valor medio =',round(N*(1-j),2))
#     print('Desviación Estándar = ',round(std_pop,2),'  - Para la binomial la desviación estándar =',round(np.sqrt(N*j*(1-j)),2))
    
#     #y aca hacemos los gráficos correspondientes
#     x = np.linspace(0,1000,1001)
#     pmf = scipy.stats.binom.pmf(x,1000,1-j)
#     plt.plot(x,pmf,'k',label='Binomial')
#     plt.xlabel('$n$')
#     plt.ylabel('$P(n,1)$')
#     plt.xlim((mean_pop-150,mean_pop+150))
#     plt.legend()
    
#     # plt.show()
#     print('\n')

# plt.savefig("../figuras/cosa2.pdf")