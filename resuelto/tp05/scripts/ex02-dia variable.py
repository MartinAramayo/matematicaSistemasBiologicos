import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import pandas as pd
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
import datetime
from numpy import cosh

def sech_sir(x, a, b, c): # sech_sir = lambda x, a, b, c : a*( cosh( b * x + c )**(-2) )
    return a*( cosh( b * x + c )**(-2) )

# plt.ion()
aDir = '../figuras/'

style = {'linewidth':0.5, 
         'ms':1.8, 
         'marker':'o', 
         'subplots':True}

#%% # abrir y pre procesar
df = pd.read_csv('2021-tp05-covid19.csv', index_col='Date')
rename_dict = {'i':'infectados',
               'd':'fallecidos'}
df = df.rename(columns=rename_dict)
df.index = pd.to_datetime(df.index)
df.index = df.index.rename('Fecha')

#%% ## feriados
file1 = 'datasetFeriadosArgentina/feriadosArgentina2020.csv'
file2 = 'datasetFeriadosArgentina/feriadosArgentina2021.csv'
feriados = pd.read_csv(file1)
feriados = pd.concat((feriados, pd.read_csv(file2))).reset_index()
month, day, year = feriados['month'], feriados['day'], feriados['year']
month, day, year = month.astype('str'), day.astype('str'), year.astype('str') 
feriados = month + ' ' + day + ' ' + year
feriados = pd.to_datetime(feriados, format='%B %d %Y')

feriados = feriados[(feriados >= df.index[0]) 
                    & (feriados <= df.index[-1])]

## para plotear
rename_plot = {'infectados':'Infectados',
               'fallecidos':'Fallecidos',
               'tests': 'Testeados',
               'difIt': 'Infectados-Fallecidos',}
plt_rename = {'columns':rename_plot}
#%% # diferencia entre infectados y testeados
df['difIt'] = df['infectados'].sub(df['tests'])
axs = df.rename(**plt_rename).plot(**style)
figs = axs[0].get_figure()
figs.tight_layout()
figs.savefig(aDir + 'ex02-resumen.pdf')
#%% ## normalizado
df_norm = df/df.max()
df_norm['difIt'] = df['difIt']/df['difIt'].min()
axs2 = df_norm.rename(**plt_rename).plot(**style)
figs2 = axs2[0].get_figure()
figs2.tight_layout()
figs2.savefig(aDir + 'ex02-resumen-normalizado.pdf')

#%% ## sin weekends
df_weekday_2 = df[(df.index.weekday != 5) # sabado 
                & (df.index.weekday != 6) # domingo
                # & (df.index.weekday != 4) # viernes
                # & (df.index.weekday != 3) # jueves
                # & (df.index.weekday != 2) # miercoles
                # & (df.index.weekday != 1) # martes
                # & (df.index.weekday != 0) # lunes
                ] 

aux_df = df_weekday_2.index
bool_index = aux_df != aux_df ## trivial case
for aux_datetime in feriados:
    bool_index |= (aux_df == aux_datetime)

# podes comentar esto si queres tener en cuenta los feriados
df_weekday_2 = df_weekday_2[~ bool_index] # feriados

axs3 = df_weekday_2.rename(**plt_rename).plot(**style)
figs3 = axs3[0].get_figure()
figs3.tight_layout()
figs3.savefig(aDir + 'ex02-sin-Finde.pdf')
# %%
pico_1 = datetime.datetime.strptime('2020-12-15','%Y-%m-%d')
df_pico_11 = df_weekday_2[df_weekday_2.index < pico_1]
df_pico_11.index = (df_pico_11.index - df_pico_11.index[0]).days
axs4 = df_pico_11.rename(**plt_rename).plot(**style)
figs4 = axs4[0].get_figure()
figs4.tight_layout()
figs4.savefig(aDir + 'ex02-sin-Finde-pico1.pdf')

##############################################
last_day = df.index[-1]
ref_day = df.index[0]
n_semanas_total = int((last_day - ref_day).days/7)

# pico real
infected= df_pico_11['infectados']
bool_index = infected == infected.max()
real_peak_day = int(df_pico_11[bool_index].index.values[0])
real_peak_date = ref_day + datetime.timedelta(days=real_peak_day)

# plt.ion()

fig, ax = plt.subplots()

prediction = {'error':[],
              'prediction':[],
              'n_semanas':[]}
for n_semanas in range(1, n_semanas_total):
    aux_df = df_pico_11[0:n_semanas*7]
    aux_args = {'f':sech_sir, 
                'xdata':aux_df.index, 
                'ydata':aux_df['infectados'],
                'p0':(5000, 0.01, -1)}
    try: 
        popt, pcov = curve_fit(**aux_args)
        a, b, c = popt

        forecast_date = ref_day + datetime.timedelta(days=int(-c/b))
        prediction['prediction'].append(forecast_date)

        dias_error = (real_peak_date - forecast_date).days
        prediction['error'].append(dias_error)
        
        prediction['n_semanas'].append(n_semanas)
    except RuntimeError:
        pass

columns = ('error', 'prediction', 'n_semanas')
aux_args = {'data': prediction, 
            'index': prediction['n_semanas'], 
            'columns': columns}
prediction_df = pd.DataFrame(**aux_args)

rename_dict = {'n_semanas':'\# de semanas despues de 2020-03-05 usadas en el ajuste',
               'prediction':'Predicción',
               'error': 'Error en días'}
aux_args = {'x': rename_dict['n_semanas'], 
            'y': rename_dict['error'], 
            'ax': ax, 
            'color': 'royalblue'} 
prediction_df.rename(columns=rename_dict).plot.scatter(**aux_args)

ax.axhline(y=0, color='k', ls='dashed')
    
fig.tight_layout()
fig.savefig(aDir + 'ex02-Error-prediccion-semanas.pdf')

fig, ax = plt.subplots()

aux_args = {'x': rename_dict['n_semanas'], 
            'y': rename_dict['prediction'], 
            'ax': ax, 
            'color': 'royalblue'} 
prediction_df.rename(columns=rename_dict).plot.scatter(**aux_args)
ax.axhline(real_peak_date, color='k', ls='dashed', label='Pico Real')

ax.legend()

fig.tight_layout()
fig.savefig(aDir + 'ex02-prediccion-semanas.pdf')

