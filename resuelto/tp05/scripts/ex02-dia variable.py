import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import pandas as pd
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
import datetime
from numpy import cosh

def sech_sir(x, a, b, c):
    return a*( cosh( b * x + c )**(-2) )

aDir = '../figuras/'

#%% # abrir y pre procesar
df = pd.read_csv('2021-tp05-covid19.csv', index_col='Date')
rename_dict = {'i':'infectados',
               'd':'fallecidos'}
df = df.rename(columns=rename_dict)
df.index = pd.to_datetime(df.index)
df.index = df.index.rename('Fecha')

#%% ## importar todos los feriados como objetos datetime
file1 = 'datasetFeriadosArgentina/feriadosArgentina2020.csv'
file2 = 'datasetFeriadosArgentina/feriadosArgentina2021.csv'
feriados_20_21 = (pd.read_csv(file1), pd.read_csv(file2))
feriados = pd.concat(feriados_20_21).reset_index()
month, day, year = feriados['month'], feriados['day'], feriados['year']
month, day, year = month.astype('str'), day.astype('str'), year.astype('str')
feriados = month + ' ' + day + ' ' + year
feriados = pd.to_datetime(feriados, format='%B %d %Y')

## para plotear
rename_plot = {'infectados':'Infectados',
               'fallecidos':'Fallecidos',
               'tests': 'Testeados',
            #    'difIt': 'Infectados-Fallecidos',
               }
style = {'linewidth':0.5, 
         'ms':1.8, 
         'marker':'o', 
         'subplots':True}
plt_rename = {'columns':rename_plot}

#%% # diferencia entre infectados y testeados
axs = df.rename(**plt_rename).plot(**style)
figs = axs[0].get_figure()
figs.tight_layout()
figs.savefig(aDir + 'ex02-resumen.pdf')

#%% ## sin weekends
df_weekday_2 = df[(df.index.weekday != 5) # sabado 
                & (df.index.weekday != 6) # domingo
                # & (df.index.weekday != 4) # viernes
                # & (df.index.weekday != 3) # jueves
                # & (df.index.weekday != 2) # miercoles
                # & (df.index.weekday != 1) # martes
                # & (df.index.weekday != 0) # lunes
                ] 
df_weekday_2 = df_weekday_2[~df_weekday_2.index.isin(feriados)]

#%% # Primer pico
pico_1 = datetime.datetime.strptime('2020-12-15','%Y-%m-%d')
df_pico_11 = df_weekday_2[df_weekday_2.index < pico_1]
df_pico_11.index = (df_pico_11.index - df_pico_11.index[0]).days
axs = df_pico_11.rename(**plt_rename).plot(**style)
figs = axs[0].get_figure()
figs.tight_layout()
figs.savefig(aDir + 'ex02-sin-Finde-pico1.pdf')

##################### graficos con y sin feriados
def check_fit(dataframe, aDir, sin_finde=False):
    
    suffix = ''    
    if sin_finde==True:
        suffix = '-sin-Finde'

    ## para plotear
    rename_plot = {'infectados':'Infectados',
                   'fallecidos':'Fallecidos',
                   'tests': 'Testeados',
                   'difIt': 'Infectados-Fallecidos',}
    style = {'linewidth':0.5, 
            'ms':1.8, 
            'marker':'o', 
            'subplots':True}
    plt_rename = {'columns':rename_plot}
    
    #%% # Primer pico
    pico_1 = datetime.datetime.strptime('2020-12-15','%Y-%m-%d')
    df_pico_11 = dataframe[dataframe.index < pico_1]
    df_pico_11.index = (df_pico_11.index - df_pico_11.index[0]).days
    axs = df_pico_11.rename(**plt_rename).plot(**style)
    figs = axs[0].get_figure()
    figs.tight_layout()
    figs.savefig(aDir + 'ex02-pico1'+suffix+'.pdf')
        
    aux_args = {'f':sech_sir, 
                'xdata':df_pico_11.index, 
                'ydata':df_pico_11['infectados'],
                'p0':(5000, 0.01, -1)}

    popt, pcov = curve_fit(**aux_args)
    a, b, c = popt

    x = df_pico_11.index.values
    y = sech_sir(x, a ,b ,c)

    fig, ax = plt.subplots()
    ax.scatter(y, df_pico_11['infectados'], color='royalblue')
    ylim = (min(y), max(y))
    ax.plot(ylim, ylim, '--k')
    ax.set_xlabel('Predicci??n')
    ax.set_ylabel('Valor real')
    fig.tight_layout()
    fig.savefig(aDir + 'ex02-qq'+suffix+'.pdf')

    fig, ax = plt.subplots()
    ax.scatter(x, df_pico_11['infectados'] - y, color='royalblue')
    ax.set_xlabel('Dias despu??s del 2020-03-05')
    ax.set_ylabel('Residuos')
    ax.axhline(y=0, color='k', ls='dashed')
    fig.tight_layout()
    fig.savefig(aDir + 'ex02-residuos'+suffix+'.pdf')

    fig, ax = plt.subplots()
    ax.scatter(x, df_pico_11['infectados'], color='royalblue')
    ax.plot(x, y, 'salmon')
    ax.set_xlabel('Dias despu??s del 2020-03-05')
    ax.set_ylabel('Infectados')
    fig.tight_layout()
    fig.savefig(aDir + 'ex02-fit'+suffix+'.pdf')

check_fit(df, aDir, sin_finde=False)
check_fit(df_weekday_2, aDir, sin_finde=True)

############## Prediccion vs numero de semanas de muestra
last_day = df.index[-1]
ref_day = df.index[0]
n_semanas_total = int((last_day - ref_day).days/7)

# pico real
infected = df_pico_11['infectados']
bool_index = infected == infected.max()
real_peak_day = int(df_pico_11[bool_index].index.values[0])
real_peak_date = ref_day + datetime.timedelta(days=real_peak_day)

# en funcion del numero de semanas de muestra
prediction = {'error':[],
              'prediction':[],
              'fecha_prediccion':[],
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

        prediction['n_semanas'].append(n_semanas)
        
        forecast_date = ref_day + datetime.timedelta(days=int(-c/b))
        prediction['prediction'].append(forecast_date)

        dias_error = (real_peak_date - forecast_date).days
        prediction['error'].append(dias_error)

        forecast_date = ref_day + datetime.timedelta(weeks=n_semanas)
        prediction['fecha_prediccion'].append(forecast_date)        
    except RuntimeError:
        pass

# columns = ('error', 'prediction', 'n_semanas')
columns = list(prediction.keys())
aux_args = {'data': prediction, 
            'index': prediction['n_semanas'], 
            'columns': columns}
prediction_df = pd.DataFrame(**aux_args)

fig, ax = plt.subplots()
rename_dict = {'n_semanas': '\# de semanas despu??s de 2020-03-05 usadas en el ajuste',
               'prediction': 'Predicci??n',
               'fecha_prediccion': 'Fecha de predicci??n',
               'error': 'Error en d??as'}
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

fig, ax = plt.subplots()
aux_args = {'x': rename_dict['fecha_prediccion'], 
            'y': rename_dict['prediction'], 
            'ax': ax, 
            'color': 'royalblue'} 
prediction_df.rename(columns=rename_dict).plot.scatter(**aux_args)
ax.axhline(real_peak_date, color='k', ls='dashed', label='Pico Real')
ax.legend()
ax.tick_params(axis='x', labelrotation=45)
fig.tight_layout()
fig.savefig(aDir + 'ex02-prediccion-fecha_prediccion.pdf')

