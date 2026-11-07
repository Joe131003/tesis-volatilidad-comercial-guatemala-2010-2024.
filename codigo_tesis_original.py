#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# MISMA LÓGICA Y MISMAS FIGURAS del TESIS_CODIGO_AJUSTADO.py original.
# Compactación de formato, SIN alterar cálculos. Jupyter: importar ejecutar_revision_terna.
"""TESIS · Código final de evaluación econométrica (Guatemala, 2010–2024).

Mantiene: retornos 100*Delta ln(precio), AR(1), t-Student, cinco productos,
variable dependiente exportaciones-importaciones y controles ITCER/TCN/remesas.
NUEVO: ML real GARCH(1,1)/EGARCH(1,1), tabla ADF/KPSS, ARCH-LM, Ljung–Box,
ARDL p=1..3/q=0..2, BIC/AIC, BG, HAC, asimetría NARDL candidata de sigma WTI,
y sensibilidad p=1..6 en otra muestra común; no atribuye cointegración sin validación.

El código NUNCA declara cointegración automáticamente; distingue el modelo
candidato de uno validado y no interpreta como causalidad.
En Jupyter: %run Tesis_Codigo_Corregido_Terna.py (o pegar el codigo en una celda)
En terminal: python Tesis_Codigo_Corregido_Terna.py --guardar
Requisitos: numpy pandas scipy statsmodels matplotlib openpyxl numba (opcional).
"""
from __future__ import annotations; import argparse, sys, warnings, json, hashlib
from datetime import datetime; from pathlib import Path
import numpy as np; import pandas as pd
from scipy.optimize import minimize; from scipy.special import gammaln
from scipy.stats import norm; import statsmodels.api as sm
from statsmodels.tsa.ardl import ARDL, UECM; from statsmodels.tsa.stattools import adfuller, kpss, zivot_andrews
try:
    from arch.unitroot import PhillipsPerron
except ImportError:
    PhillipsPerron = None
from statsmodels.stats.diagnostic import het_arch, acorr_ljungbox, acorr_breusch_godfrey, het_breuschpagan, breaks_cusumolsresid
from statsmodels.stats.stattools import jarque_bera; from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt; import matplotlib.dates as mdates
try:
    from numba import njit
except ImportError:

    def njit(*args, **kwargs):

        def ident(fn):
            return fn
        return ident
ARCHIVO = 'Base_datos_tesis_corregida_2010_2024.xlsx'
MAPA = {'Periodo': 'fecha', 'Exportaciones FOB (US$ millones)': 'exportaciones', 'Importaciones CIF (US$ millones)': 'importaciones', 'Balanza comercial (US$ millones)': 'balanza', 'Café Arábica (US$/kg)': 'cafe', 'Azúcar mundial (US$/kg)': 'azucar', 'Cardamomo: valor unitario exportación (US$/kg)': 'cardamomo', 'Banano EE.UU. (US$/kg)': 'banano', 'Petróleo WTI (US$/bbl)': 'petroleo', 'Tipo de cambio promedio (Q/USD)': 'tcn', 'Remesas familiares (US$ millones)': 'remesas', 'ITCER global (índice)': 'itcer'}
PRODUCTOS = ['cafe', 'azucar', 'banano', 'cardamomo', 'petroleo']
NOMBRES = {'cafe': 'Café', 'azucar': 'Azúcar', 'banano': 'Banano', 'cardamomo': 'Cardamomo', 'petroleo': 'Petróleo WTI'}
COLORES = ['#76523E', '#B59130', '#2C8C7C', '#8E579E', '#235B89']; TABLAS = {}
FIGURAS = {}

def emitir(nombre, tabla, mostrar, guardar, ruta):
    TABLAS[nombre] = tabla.copy()
    if guardar:
        tabla.to_csv(ruta / f'{nombre}.csv', index=False, encoding='utf-8-sig')
    if mostrar:
        try:
            from IPython.display import display, Markdown; display(Markdown('### ' + nombre.replace('_', ' ')))
            display(tabla.head(30).round(4))
        except ImportError:
            print('\n' + nombre + '\n' + tabla.head(20).round(4).to_string(index=False))

def resolver_archivo(nombre):
    p = Path(nombre).expanduser(); rutas = [Path.cwd()]
    ubicacion_script = globals().get('__file__')
    if ubicacion_script:
        rutas.append(Path(ubicacion_script).expanduser().resolve().parent)
    candidatos = [p] if p.is_absolute() else [base / p for base in rutas]
    if not p.is_absolute() and p.name == ARCHIVO:
        for base in rutas:
            candidatos.append(base / 'Base_datos_tesis_corregida_2010_2024(2).xlsx')
    for candidato in dict.fromkeys(candidatos):
        if candidato.is_file():
            return candidato.resolve()
    intentados = '\n'.join(('  - ' + str(x) for x in dict.fromkeys(candidatos)))
    raise FileNotFoundError(f'No se encontró el Excel «{nombre}».\nCarpeta de trabajo actual: {Path.cwd()}\nUbicaciones examinadas:\n{intentados}\nColoca el Excel en esa carpeta o pasa una ruta completa con --archivo.')

def leer_datos(ruta):
    r = pd.read_excel(ruta, sheet_name='BASE_CORREGIDA'); faltan = set(MAPA) - set(r.columns)
    if faltan:
        raise ValueError(f'Faltan columnas: {sorted(faltan)}')
    d = r[list(MAPA)].rename(columns=MAPA); d['fecha'] = pd.to_datetime(d['fecha'])
    d = d.set_index('fecha').sort_index().apply(pd.to_numeric, errors='raise'); fechas = pd.date_range('2010-01-01', '2024-12-01', freq='MS')
    if not d.index.equals(fechas):
        raise ValueError('Se requieren 180 meses completos 2010-01 a 2024-12.')
    if not np.isfinite(d.to_numpy()).all() or (d[PRODUCTOS] <= 0).any().any():
        raise ValueError('Hay valores faltantes, no finitos o precios no positivos.')
    np.testing.assert_allclose(d.balanza, d.exportaciones - d.importaciones, rtol=0, atol=2e-05); imputados = None
    try:
        imp = pd.read_excel(ruta, sheet_name='CALIDAD_CARDAMOMO'); imp['Periodo'] = pd.to_datetime(imp['Periodo'])
        imp = imp.set_index('Periodo').reindex(d.index); imputados = imp['Indicador imputado'].astype(int)
        if not imputados.isin([0, 1]).all():
            raise ValueError('Indicadores cardamomo inválidos')
    except ValueError:
        raise
    except Exception:
        pass
    for p in PRODUCTOS:
        d['r_' + p] = 100 * np.log(d[p]).diff()
    d['covid'] = ((d.index >= '2020-03-01') & (d.index <= '2021-12-01')).astype(int); return (d, imputados)

@njit(cache=False)
def _recursiones(e, omega, alpha, beta, gamma, egarch, basevar, ez):
    n = len(e); h = np.empty(n)
    h[0] = basevar
    for t in range(1, n):
        if egarch:
            z = e[t - 1] / np.sqrt(max(h[t - 1], 1e-10)); logh = omega + alpha * (abs(z) - ez) + gamma * z + beta * np.log(max(h[t - 1], 1e-10))
            logh = min(18.0, max(-18.0, logh)); h[t] = np.exp(logh)
        else:
            h[t] = omega + alpha * e[t - 1] * e[t - 1] + beta * h[t - 1]
    return h

def _loglike(par, x, egarch, var, mean_lags=1, burn_in=1):
    mu = par[0]; phis = par[1:1 + mean_lags]
    omega, alpha, beta = par[1 + mean_lags:4 + mean_lags]; gamma = par[4 + mean_lags] if egarch else 0.0
    nu = par[-1]
    if nu <= 2.03 or beta >= 0.999 or (not egarch and alpha + beta >= 0.999):
        return 10000000000.0
    e = x[burn_in:].copy() - mu
    for j, phi in enumerate(phis, 1):
        e -= phi * x[burn_in - j:len(x) - j]
    ez = float(np.exp(np.log(2) + 0.5 * np.log(nu - 2) + gammaln((nu + 1) / 2) - 0.5 * np.log(np.pi) - np.log(nu - 1) - gammaln(nu / 2)))
    h = _recursiones(e, omega, alpha, beta, gamma, egarch, var, ez)
    if not np.isfinite(h).all() or np.min(h) < 1e-09:
        return 10000000000.0
    con = gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(np.pi * (nu - 2))
    logdensity = con - 0.5 * np.log(h) - 0.5 * (nu + 1) * np.log1p(e * e / h / (nu - 2)); f = -float(np.sum(logdensity))
    return f if np.isfinite(f) else 10000000000.0

def garch_mle(y, modelo, mean_lags=1, burn_in=None):
    x = y.to_numpy(float); egarch = modelo == 'EGARCH'
    burn_in = mean_lags if burn_in is None else burn_in
    if not 1 <= mean_lags <= burn_in < len(x) - 25:
        raise ValueError('Media o burn_in inválidos')
    v = max(float(np.var(x)), 0.01); Z = np.column_stack([np.ones(len(x) - burn_in)] + [x[burn_in - j:len(x) - j] for j in range(1, mean_lags + 1)])
    m = np.linalg.lstsq(Z, x[burn_in:], rcond=None)[0]; base_bounds = [(-5 * np.sqrt(v), 5 * np.sqrt(v))] + [(-0.96, 0.96)] * mean_lags
    if egarch:
        bounds = base_bounds + [(-8.0, 8.0), (0.0, 2.0), (0.0, 0.995), (-2.0, 2.0), (2.05, 80.0)]
        volstarts = [[np.log(v) * 0.2, 0.16, 0.8, -0.1, 8], [np.log(v) * 0.5, 0.12, 0.5, 0.1, 12], [np.log(v) * 0.08, 0.3, 0.92, -0.2, 5]]
    else:
        bounds = base_bounds + [(1e-07, 20 * v), (1e-06, 0.8), (0.0, 0.998), (2.05, 80.0)]
        volstarts = [[v * 0.1, 0.1, 0.8, 8], [v * 0.4, 0.12, 0.5, 12], [v * 0.03, 0.05, 0.92, 5]]
    fits = []
    for volstart in volstarts:
        start = np.r_[m, volstart]
        fits.append(minimize(_loglike, start, args=(x, egarch, v, mean_lags, burn_in), bounds=bounds, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-09, 'maxls': 40}))
    valid = [r for r in fits if r.success and np.isfinite(r.fun) and (r.fun < 1000000000.0)]
    fit = min(valid, key=lambda r: r.fun) if valid else min(fits, key=lambda r: r.fun); par = fit.x
    mu = par[0]; phis = par[1:1 + mean_lags]
    omega, alpha, beta = par[1 + mean_lags:4 + mean_lags]; gamma = par[4 + mean_lags] if egarch else 0.0
    nu = par[-1]; ez = float(np.exp(np.log(2) + 0.5 * np.log(nu - 2) + gammaln((nu + 1) / 2) - 0.5 * np.log(np.pi) - np.log(nu - 1) - gammaln(nu / 2)))
    e = x[burn_in:].copy() - mu
    for j, phi in enumerate(phis, 1):
        e -= phi * x[burn_in - j:len(x) - j]
    h = _recursiones(e, omega, alpha, beta, gamma, egarch, v, ez); sigma = pd.Series(np.sqrt(h), index=y.index[burn_in:])
    z = pd.Series(e / np.sqrt(h), index=y.index[burn_in:]); k = len(par)
    n = len(e); se = np.full(k, np.nan)
    try:
        from statsmodels.tools.numdiff import approx_hess; H = approx_hess(par, lambda a: _loglike(a, x, egarch, v, mean_lags, burn_in))
        if np.linalg.eigvalsh(H).min() > 1e-07:
            se = np.sqrt(np.diag(np.linalg.inv(H)))
    except Exception:
        pass
    keys = ['Const'] + [f'AR({i})' for i in range(1, mean_lags + 1)] + ['omega', 'alpha', 'beta'] + (['gamma'] if egarch else []) + ['nu']
    param = pd.DataFrame({'parametro': keys, 'valor': par, 'error_estandar_aprox': se, 'p_normal_aprox': [2 * norm.sf(abs(a / b)) if np.isfinite(b) and b > 0 else np.nan for a, b in zip(par, se)]})
    lb = acorr_ljungbox(z, lags=[6, 12, 24], model_df=mean_lags, return_df=True); lb2 = acorr_ljungbox(z * z, lags=[6, 12, 24], return_df=True)
    arch6 = float(het_arch(z, nlags=6)[1]); arch12 = float(het_arch(z, nlags=12)[1])
    archpre = float(het_arch(sm.OLS(x[burn_in:], Z).fit().resid, nlags=12, ddof=1 + mean_lags)[1])
    diag = bool(min(lb.loc[12, 'lb_pvalue'], lb2.loc[12, 'lb_pvalue'], arch12) >= 0.05)
    numerico = bool(fit.success and nu > 2.12 and np.isfinite(sigma).all() and (beta < 0.995) and (egarch or alpha + beta < 0.999))
    registro = dict(producto=y.name.replace('r_', ''), modelo=modelo, media_ar=mean_lags, n=n, AIC=2 * k + 2 * fit.fun, BIC=np.log(n) * k + 2 * fit.fun, LL=-fit.fun, k=k, converge=bool(fit.success), nu=nu, alpha=alpha, beta=beta, gamma=gamma, arch_pre_p=archpre, lb6_p=float(lb.loc[6, 'lb_pvalue']), lb12_p=float(lb.loc[12, 'lb_pvalue']), lb24_p=float(lb.loc[24, 'lb_pvalue']), lb2_12_p=float(lb2.loc[12, 'lb_pvalue']), arch_post_6_p=arch6, arch_post_12_p=arch12, diagnosticos_5pct=diag, valido_numerico=numerico, vol_media_pct=float(sigma.mean()), estimador='MLE SciPy; innovaciones t-Student')
    return (registro, sigma, param)

def pruebas_integracion(d, cols):
    filas = []; clas = {}
    notas = {}
    for col in cols:
        s = d[col].dropna(); pr = {}
        for mode in ['c', 'ct']:
            for orden in [0, 1, 2]:
                serie = s if orden == 0 else s.diff().dropna() if orden == 1 else s.diff().diff().dropna()
                with warnings.catch_warnings(record=True) as ww:
                    warnings.simplefilter('always'); ad = adfuller(serie, regression=mode, autolag='BIC', maxlag=12)
                    ad_sens = adfuller(serie, regression=mode, autolag='AIC', maxlag=14); kp = kpss(serie, regression=mode, nlags='auto')
                pp = np.nan
                if PhillipsPerron is not None:
                    try:
                        pp = float(PhillipsPerron(serie, trend=mode).pvalue)
                    except Exception:
                        pass
                row = {'variable': col, 'terminos': mode, 'diferencias': orden, 'n': len(serie), 'ADF': ad[0], 'ADF_p': ad[1], 'ADF_lags': ad[2], 'ADF_sens14_p': ad_sens[1], 'ADF_sens14_lags': ad_sens[2], 'KPSS': kp[0], 'KPSS_p': kp[1], 'PP_p': pp, 'aviso_KPSS': ' | '.join((str(w.message) for w in ww if 'p-value' in str(w.message)))}
                filas.append(row); pr[mode, orden] = row
        modos_clas = ('c',) if col.startswith('vol_') else ('c', 'ct'); nivel_adf = all((pr[m, 0]['ADF_p'] < 0.05 for m in modos_clas))
        nivel_kpss = all((pr[m, 0]['KPSS_p'] > 0.05 for m in modos_clas)); diff_adf = all((pr[m, 1]['ADF_p'] < 0.05 for m in modos_clas))
        diff_kpss_any = any((pr[m, 1]['KPSS_p'] > 0.05 for m in modos_clas)); diff_conflict = any((pr[m, 1]['KPSS_p'] <= 0.05 for m in modos_clas))
        if nivel_adf and nivel_kpss:
            tipo = 'I(0) compatible'
        elif diff_adf and diff_kpss_any and (not nivel_adf):
            tipo = 'I(1) probable'
        elif not diff_adf and any((pr[m, 2]['ADF_p'] < 0.05 for m in ('c', 'ct'))):
            tipo = 'Indeterminado; investigar I(2)'
        else:
            tipo = 'Indeterminado'
        clas[col] = tipo
        notas[col] = ('KPSS con tendencia contradice resultado con constante; verificar. ' if col.startswith('vol_') and pr['ct', 0]['KPSS_p'] <= 0.05 else '') + ('Conflicto KPSS en primeras diferencias; revisar tendencia/rupturas. ' if diff_conflict else '') + ('ADF sensible al máximo de rezagos. ' if any(((pr[m, j]['ADF_p'] < 0.05) != (pr[m, j]['ADF_sens14_p'] < 0.05) for m in ('c', 'ct') for j in (0, 1))) else '')
    resumen = pd.DataFrame([{'variable': x, 'diagnostico': clas[x], 'advertencias': notas[x]} for x in cols])
    for variable in ['balanza', 'itcer', 'tcn', 'remesas']:
        if variable not in cols:
            continue
        try:
            za = zivot_andrews(d[variable].dropna(), regression='ct', maxlag=12, autolag='BIC')
            resumen.loc[resumen.variable == variable, 'ZA_niveles_p'] = float(za[1])
            resumen.loc[resumen.variable == variable, 'ZA_ruptura_indice'] = int(za[4])
        except Exception:
            pass
    return (pd.DataFrame(filas), resumen, clas)

def meses_dummies(index):
    d = pd.get_dummies(index.month, prefix='mes', drop_first=True, dtype=float); d.index = index
    return d

def estacionalidad(d):
    out = []
    for c in ['balanza', 'remesas', 'tcn', 'itcer'] + PRODUCTOS:
        X = meses_dummies(d.index); X['t'] = np.arange(len(d))
        r = sm.OLS(d[c], sm.add_constant(X)).fit(); R = np.zeros((11, len(r.params)))
        R[:, 1:12] = np.eye(11); out.append(dict(variable=c, p_meses=float(r.f_test(R).pvalue)))
    return pd.DataFrame(out)

def inspeccionar_ardl(fit):
    u = UECM.from_ardl(fit.model).fit(); yy = np.asarray(u.model._y, float)
    XX = np.asarray(u.model._x, float); ols = sm.OLS(yy, XX).fit()
    names = list(u.params.index)
    if np.max(np.abs(ols.params - u.params.to_numpy())) > 1e-05:
        raise ValueError('UECM y OLS no coinciden')
    vals = {f'BG{lag}': float(acorr_breusch_godfrey(ols, nlags=lag)[1]) for lag in (6, 12, 24)}
    vals.update(BP=float(het_breuschpagan(ols.resid, XX)[1]), JB=float(jarque_bera(ols.resid)[1]), CUSUM=float(breaks_cusumolsresid(ols.resid, ddof=XX.shape[1])[1]))
    return (vals, u, ols, names)

def seleccionar_dinamico(data, cols, nombre, usar_meses=False, criterio='bic'):
    X = data[cols]; y = data.balanza
    f = meses_dummies(data.index) if usar_meses else None; opt = []
    fits = []
    for p in (1, 2, 3):
        for q in (1, 2):
            try:
                fit = ARDL(y, lags=p, exog=X, order=q, fixed=f, trend='c', hold_back=3, missing='raise').fit(); diag, u, ols, idx = inspeccionar_ardl(fit)
                registro = {'nombre': nombre, 'meses': usar_meses, 'p': p, 'q': q, 'n': fit.nobs, 'AIC': fit.aic, 'BIC': fit.bic, 'parametros': len(fit.params), **diag}
                registro['pasa_BG'] = min((registro[z] for z in ['BG6', 'BG12', 'BG24'])) >= 0.05; opt.append(registro)
                fits.append((registro, fit))
            except Exception as ex:
                opt.append({'nombre': nombre, 'meses': usar_meses, 'p': p, 'q': q, 'error': str(ex)})
    buenos = [t for t in fits if t[0]['pasa_BG']]; elegir = min(buenos if buenos else fits, key=lambda z: z[0][criterio.upper()]) if fits else None
    return (pd.DataFrame(opt), elegir)

def analisis_elegido(fit, cols, nombre, clas):
    diag, u, ols, ix = inspeccionar_ardl(fit); yh = np.asarray(u.model._y, float)
    xx = np.asarray(u.model._x, float); hac = sm.OLS(yh, xx).fit(cov_type='HAC', cov_kwds={'maxlags': 6, 'use_correction': True})
    tab = pd.DataFrame({'modelo': nombre, 'termino': ix, 'coef': ols.params, 'p_convencional': ols.pvalues, 'EE_HAC': hac.bse, 'p_HAC': hac.pvalues})

    def wald(restr):
        R = np.zeros((len(restr), len(ix)))
        for j, coefs in enumerate(restr):
            for n, val in coefs.items():
                if n not in ix:
                    return np.nan
                R[j, ix.index(n)] = val
        return float(hac.f_test(R).pvalue)
    nivel_y = wald([{'balanza.L1': 1}]); nivel_x = wald([{f'{col}.L1': 1} for col in cols])
    volshort = [k for k in ix if k.startswith('D.vol_')]; corto = wald([{k: 1} for k in volshort]) if volshort else np.nan
    clases = [clas.get(x, 'INDETERMINADO') for x in ['balanza'] + cols]; admisible = all((z in ('I(0)', 'I(1)') for z in clases))
    bounds = 'NO interpretable: integración indeterminada, y/o sumas parciales de sigma'; F = np.nan
    cv0 = np.nan; cv1 = np.nan
    if admisible and 'NARDL' not in nombre:
        try:
            bt = u.bounds_test(case=3, asymptotic=True); F = float(bt.stat)
            cv0 = float(bt.crit_vals.loc[95.0, 'lower']); cv1 = float(bt.crit_vals.loc[95.0, 'upper'])
            bounds = 'F supera I(1), pero F_X y t-Bounds requieren validación' if F > cv1 else 'Sin rechazo' if F < cv0 else 'Inconcluso'
        except Exception as ex:
            bounds = f'Bounds no disponible: {ex}'
    asym_corto = np.nan; asym_niveles = np.nan
    if 'NARDL' in nombre:
        a = 'vol_petroleo_pos'; b = 'vol_petroleo_neg'
        asym_niveles = wald([{a + '.L1': 1, b + '.L1': -1}]); term = []
        for j in [0, 1]:
            if 'D.' + a + '.L' + str(j) in ix and 'D.' + b + '.L' + str(j) in ix:
                term.append({'D.' + a + '.L' + str(j): 1, 'D.' + b + '.L' + str(j): -1})
        asym_corto = wald(term) if term else np.nan
    return (tab, dict(modelo=nombre, **diag, bounds_F=F, I0_5=cv0, I1_5=cv1, bounds_estado=bounds, F_X_HAC_p=nivel_x, balanza_L1_HAC_p=nivel_y, vol_corto_HAC_p=corto, asimetria_corto_HAC_p=asym_corto, asimetria_niveles_HAC_p=asym_niveles))
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 16, 'axes.titlesize': 19, 'axes.labelsize': 16, 'xtick.labelsize': 13, 'ytick.labelsize': 13, 'legend.fontsize': 13, 'figure.titlesize': 22, 'figure.facecolor': 'white', 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42, 'savefig.dpi': 300})

def presentar_figura(fig, titulo, mostrar, guardar, ruta):
    fig.tight_layout(rect=[0, 0.035, 1, 0.95]); FIGURAS[titulo] = fig
    if guardar:
        fig.savefig(ruta / f'{titulo}.png', dpi=300, bbox_inches='tight'); fig.savefig(ruta / f'{titulo}.pdf', bbox_inches='tight')
    if mostrar:
        try:
            from IPython.display import display; display(fig)
        except ImportError:
            plt.show()
    plt.close(fig)

def graficar(d, candidatos, seleccion, comparacion, mostrar, guardar, ruta):
    anual = d.balanza.groupby(d.index.year).sum(); fig, axs = plt.subplots(2, 1, figsize=(15, 10))
    fig.suptitle('Saldo comercial de Guatemala · 2010–2024'); axs[0].plot(d.index, d.balanza, color='#235B89', lw=2)
    axs[0].axhline(0, color='gray', lw=0.6); axs[0].set_ylabel('US$ millones / mes')
    axs[0].set_title('Serie mensual'); axs[1].bar(anual.index, anual, color='#B5554F')
    axs[1].set_title('Saldo anual: suma mensual'); axs[1].set_ylabel('US$ millones / año')
    axs[1].set_xticks(anual.index[::2]); presentar_figura(fig, 'Figura_01_Balanza', mostrar, guardar, ruta)
    for p, color in zip(PRODUCTOS, COLORES):
        fig, axs = plt.subplots(2, 1, figsize=(15, 9), sharex=True); fig.suptitle(NOMBRES[p] + ' · precio y retorno logarítmico')
        axs[0].plot(d.index, d[p], color=color, lw=2); axs[0].set_ylabel('US$/bbl' if p == 'petroleo' else 'US$/kg')
        axs[0].set_title('Precio observado' if p != 'cardamomo' else 'Valor unitario exportación (contiene imputaciones)')
        axs[1].plot(d.index, d['r_' + p], color=color, lw=1.5); axs[1].axhline(0, color='gray', ls='--')
        axs[1].set_ylabel('100 × Δln(P) (%)'); axs[1].xaxis.set_major_locator(mdates.YearLocator(2))
        axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y')); presentar_figura(fig, 'Figura_02_' + p, mostrar, guardar, ruta)
    fig, axs = plt.subplots(3, 2, figsize=(17, 14)); fig.suptitle('Volatilidad condicional mensual')
    for ax, p, c in zip(axs.flat, PRODUCTOS, COLORES):
        ax.plot(d.index, d['vol_' + p], color=c, lw=1.7); ax.set_title(NOMBRES[p] + ' · ' + seleccion[p])
        ax.set_ylabel('σ mensual (%)'); ax.grid(alpha=0.2)
        ax.xaxis.set_major_locator(mdates.YearLocator(3)); ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    axs.flat[-1].axis('off')
    axs.flat[-1].text(0.02, 0.85, 'Las volatilidades están expresadas\ncomo desviación estándar condicional\nde retornos mensuales (%).\n\nNo equivalen a subidas o bajadas de precios.', fontsize=15, va='top')
    presentar_figura(fig, 'Figura_03_Volatilidades', mostrar, guardar, ruta)
    c = candidatos.groupby(['producto', 'modelo'], as_index=False)['BIC'].min().pivot(index='producto', columns='modelo', values='BIC').reindex(PRODUCTOS)
    fig, ax = plt.subplots(figsize=(15, 7)); z = np.arange(5)
    ax.bar(z - 0.18, c['GARCH'], 0.36, label='GARCH(1,1)', color='#497DA6'); ax.bar(z + 0.18, c['EGARCH'], 0.36, label='EGARCH(1,1)', color='#D8A451')
    ax.set_xticks(z, [NOMBRES[p] for p in PRODUCTOS]); ax.set_ylabel('BIC (menor = mejor)')
    ax.legend(); ax.set_title('Comparación GARCH / EGARCH')
    presentar_figura(fig, 'Figura_04_GARCH_EGARCH', mostrar, guardar, ruta); fig, axs = plt.subplots(1, 2, figsize=(16, 7))
    x = np.arange(len(comparacion)); axs[0].plot(x, comparacion.BIC, color='#3F759C', lw=2, marker='o', markersize=13)
    for i, val in enumerate(comparacion.BIC):
        axs[0].annotate(f'{val:.2f}', (i, val), xytext=(0, 11), textcoords='offset points', ha='center', fontsize=15)
    axs[0].set_ylim(comparacion.BIC.min() - 5, comparacion.BIC.max() + 8); axs[0].set_xticks(x, comparacion.modelo, rotation=16, ha='right')
    axs[0].set_ylabel('BIC (escala ampliada para comparar)'); axs[0].set_title('Comparación de modelos dinámicos')
    axs[1].bar(x, comparacion.BG12, color=['#3D937B' if t >= 0.05 else '#C57663' for t in comparacion.BG12]); axs[1].axhline(0.05, color='black', ls='--')
    axs[1].set_xticks(x, comparacion.modelo, rotation=16, ha='right'); axs[1].set_ylabel('p-valor BG(12)')
    axs[1].set_title('Autocorrelación residual (p ≥ 0.05 deseable)'); presentar_figura(fig, 'Figura_05_Comparacion_ARDL_NARDL', mostrar, guardar, ruta)

def diagnosticos_niveles(fit):
    yy = np.asarray(fit.model._y, dtype=float); xx = np.asarray(fit.model._x, dtype=float)
    mod = sm.OLS(yy, xx).fit(); np.testing.assert_allclose(mod.params, np.asarray(fit.params), rtol=1e-07, atol=1e-07)
    d = {f'BG{lag}': float(acorr_breusch_godfrey(mod, nlags=lag)[1]) for lag in (6, 12, 24)}; d['BP'] = float(het_breuschpagan(mod.resid, xx)[1])
    d['JB'] = float(jarque_bera(mod.resid)[1])
    try:
        d['CUSUM'] = float(breaks_cusumolsresid(mod.resid, ddof=xx.shape[1])[1])
    except Exception:
        d['CUSUM'] = np.nan
    d['BG_ok'] = all((d[f'BG{j}'] >= 0.05 for j in (6, 12, 24)))
    d['diagnosticos_ok'] = bool(d['BG_ok'] and d['BP'] >= 0.05 and (not np.isfinite(d['CUSUM']) or d['CUSUM'] >= 0.05)); return (d, mod)

def evaluar_grid(datos, modelo, columnas, maxp=3, maxq=2, meses=(False, True), hold_back=None):
    if hold_back is None:
        hold_back = maxp
    y = datos['balanza']; X = datos[columnas]
    tablas = []; ajustes = {}
    for usa_meses in meses:
        f = meses_dummies(datos.index) if usa_meses else None
        for p in range(1, maxp + 1):
            for q in range(maxq + 1):
                key = (modelo, bool(usa_meses), p, q)
                try:
                    fit = ARDL(y, lags=p, exog=X, order=q, trend='c', fixed=f, hold_back=hold_back, missing='raise').fit()
                    diag, ols = diagnosticos_niveles(fit)
                    a = {'modelo': modelo, 'meses': usa_meses, 'p': p, 'q': q, 'n': int(fit.nobs), 'k': int(len(fit.params)), 'AIC': float(fit.aic), 'BIC': float(fit.bic), **diag, 'compatible_UECM_statsmodels': bool(q >= 1)}
                    tablas.append(a); ajustes[key] = fit
                except (ValueError, TypeError, np.linalg.LinAlgError, RuntimeError) as e:
                    tablas.append({'modelo': modelo, 'meses': usa_meses, 'p': p, 'q': q, 'error': str(e)})
    return (pd.DataFrame(tablas), ajustes)

def mejores_candidatos(tabla):
    sal = []
    for mod in ['ARDL', 'NARDL_exploratorio']:
        t = tabla[(tabla['modelo'] == mod) & tabla['BIC'].notna()].copy()
        for crit in ['AIC', 'BIC']:
            for tag, filtro in [('Mínimo global', pd.Series(True, index=t.index)), ('Con BG(6,12,24)', t['BG_ok']), ('Con BG, BP y CUSUM', t['diagnosticos_ok'])]:
                sub = t.loc[filtro]
                if sub.empty:
                    continue
                b = sub.sort_values(crit).iloc[0].copy(); b['criterio'] = crit
                b['filtro'] = tag; sal.append(b)
    return pd.DataFrame(sal)

def resumen_coef_niveles(fit, nombre):
    diag, ols = diagnosticos_niveles(fit)
    h = sm.OLS(np.asarray(fit.model._y, float), np.asarray(fit.model._x, float)).fit(cov_type='HAC', cov_kwds={'maxlags': 6, 'use_correction': True})
    coef = pd.DataFrame({'modelo': nombre, 'termino': list(fit.params.index), 'coeficiente': np.asarray(fit.params), 'error_estandar_HAC': h.bse, 'p_HAC_aproximado': h.pvalues, 'p_OLS': ols.pvalues})
    return (coef, h)

def reparametrizar_ardl_q0(fit):
    if not all((len(lags) == 1 and lags[0] == 0 for lags in fit.model.dl_lags.values())):
        raise ValueError('Esta rutina es exclusivamente para regresores q=0')
    p = max(fit.model.ar_lags); y = fit.model.data.orig_endog
    x = fit.model.data.orig_exog
    if not isinstance(y, pd.Series) or not isinstance(x, pd.DataFrame):
        raise ValueError('Se requieren series con fechas originales para validar el ECM.')
    hb = fit.model.hold_back; ecm = pd.DataFrame(index=y.index)
    ecm['const'] = 1.0; ecm['y.L1'] = y.shift(1)
    for col in x.columns:
        ecm[col + '.L1'] = x[col].shift(1); ecm['D.' + col] = x[col].diff()
    for j in range(1, p):
        ecm[f'D.y.L{j}'] = y.diff().shift(j)
    fijo = getattr(fit.model, '_fixed', None)
    if fijo is not None and np.asarray(fijo).ndim == 2 and (np.asarray(fijo).shape[1] > 0):
        n_f = fijo.shape[1]
        for j in range(n_f):
            ecm[f'fixed{j}'] = np.asarray(fijo)[:, j]
    dy = y.diff().iloc[hb:]; ecm = ecm.loc[dy.index]
    pval = fit.params; phi = np.array([float(pval[f'balanza.L{i}']) for i in range(1, p + 1)])
    coef = {'const': float(pval.get('const', 0.0)), 'y.L1': float(phi.sum() - 1)}
    for col in x.columns:
        b = float(pval[f'{col}.L0']); coef[col + '.L1'] = b
        coef['D.' + col] = b
    for j in range(1, p):
        coef[f'D.y.L{j}'] = float(-phi[j:].sum())
    extras = [n for n in pval.index if not (n == 'const' or n.startswith('balanza.L') or any((n == f'{col}.L0' for col in x.columns)))]
    for j in range(len(extras)):
        coef[f'fixed{j}'] = float(pval[extras[j]])
    ecm = ecm[list(coef)]; fitted = ecm @ pd.Series(coef)
    res = dy - fitted
    if not np.allclose(np.asarray(res), np.asarray(fit.resid), rtol=1e-07, atol=1e-07):
        raise AssertionError('La reparametrización ECM q=0 no reprodujo los residuos ARDL')
    R = []; q = []
    names = list(pval.index); row = np.zeros(len(names))
    for i in range(1, p + 1):
        row[names.index(f'balanza.L{i}')] = 1
    R.append(row); q.append(1.0)
    for col in x.columns:
        row = np.zeros(len(names)); row[names.index(f'{col}.L0')] = 1
        R.append(row); q.append(0.0)
    ols = sm.OLS(np.asarray(fit.model._y, float), np.asarray(fit.model._x, float)).fit(); f_aux = float(ols.f_test((np.asarray(R), np.asarray(q))).fvalue)
    return ({'max_error_equivalencia': float(np.abs(res - np.asarray(fit.resid)).max()), 'p': p, 'q': 0, 'F_niveles_condicional_auxiliar': f_aux, 'PSS_interpretable': False, 'nota': 'F auxiliar sobre restricciones afines; NO usar valores críticos PSS para q=0.'}, ecm)

def comparar_uecm_candidato(datos, columnas, todo, ajustes, clases):
    cand = todo[(todo.modelo == 'ARDL') & todo['compatible_UECM_statsmodels'] & todo.BIC.notna()]; admis = cand[cand['diagnosticos_ok']]
    eleg = (admis if not admis.empty else cand).sort_values('BIC')
    if eleg.empty:
        return (None, None)
    r = eleg.iloc[0]; fit = ajustes['ARDL', bool(r.meses), int(r.p), int(r.q)]
    u = UECM.from_ardl(fit.model).fit(); names = list(u.params.index)
    hac = sm.OLS(np.asarray(u.model._y, float), np.asarray(u.model._x, float)).fit(cov_type='HAC', cov_kwds={'maxlags': 6, 'use_correction': True})
    tabla = pd.DataFrame({'termino': names, 'coeficiente': np.asarray(u.params), 'p_HAC_aproximado': hac.pvalues, 'ee_HAC': hac.bse})
    x_niveles = ['balanza.L1'] + [f'{x}.L1' for x in columnas]; R = np.zeros((len(x_niveles), len(names)))
    for i, nombre in enumerate(x_niveles):
        if nombre not in names:
            raise RuntimeError('Inconsistencia de términos de niveles: ' + nombre)
        R[i, names.index(nombre)] = 1.0
    F = float(hac.f_test(R).fvalue); F_regresores = float(hac.f_test(R[1:]).fvalue)
    p_y = float(hac.t_test(R[0:1]).pvalue)
    integracion_compatible = all((clases.get(v) in ('I(0) compatible', 'I(1) probable') for v in ['balanza'] + columnas))
    mensaje = 'SIN VALIDAR: integración/condiciones pendientes; NO afirmar cointegración.' if not integracion_compatible else 'Integración clasificada: verificar límites, degeneración y supuestos antes de concluir.'
    info = {'modelo': 'ARDL_UECM_auxiliar_q_mayor_igual_1', 'p': int(r.p), 'q': int(r.q), 'meses': bool(r.meses), 'n': int(r.n), 'BIC': float(r.BIC), 'F_niveles_HAC_descriptiva': F, 'F_regresores_HAC_descriptiva': F_regresores, 'p_OLS_HAC_saldo_L1_auxiliar': p_y, 'estado_cointegracion': mensaje, 'nota': 'F y p anteriores NO sustituyen PSS t-bounds ni prueba de degeneración no estándar.'}
    if integracion_compatible:
        try:
            bounds = u.bounds_test(case=3, asymptotic=True); info['F_PSS_referencia'] = float(bounds.stat)
            cv = bounds.crit_vals; info['I0_5'] = float(cv.loc[95.0, 'lower'])
            info['I1_5'] = float(cv.loc[95.0, 'upper'])
            info['estado_cointegracion'] = 'CONTRASTE REFERENCIAL: aún requieren análisis de degeneración y robustez.'
        except Exception as e:
            info['error_bounds'] = str(e)
    return (info, tabla)

def figura_rezagos(tabla, mostrar, guardar, ruta):
    t = tabla[(tabla.modelo == 'ARDL') & (tabla.meses == True) & tabla.BIC.notna()].sort_values(['p', 'q'])
    if t.empty:
        return
    fig, ax = plt.subplots(2, 1, figsize=(17, 12)); fig.suptitle('Selección de rezagos ARDL · muestra común · 2010–2024')
    xs = np.arange(len(t)); labs = [f'({int(r.p)},{int(r.q)})' for _, r in t.iterrows()]
    for a, c in zip(ax, ['AIC', 'BIC']):
        colors = ['#3E8C78' if bool(x) else '#C57462' for x in t.diagnosticos_ok]; a.bar(xs, t[c], color=colors)
        a.set_xticks(xs, labs, rotation=0); a.set_ylabel(c + ' (menor es mejor)')
        a.set_title('ARDL(p,q) con 11 indicadores de mes'); lo = t[c].min()
        hi = t[c].max(); a.set_ylim(lo - 27, hi + 28)
        for i, v in enumerate(t[c]):
            a.annotate(f'{v:.1f}', (i, v), ha='center', xytext=(0, 7), textcoords='offset points', fontsize=12)
        a.grid(axis='y', alpha=0.15)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color='#3E8C78', label='BG(6,12,24), BP y CUSUM aprobados'), Patch(color='#C57462', label='Algún diagnóstico pendiente')], loc='lower center', ncol=2, fontsize=13)
    presentar_figura(fig, 'Figura_06_AIC_BIC_REZAGOS', mostrar, guardar, ruta)

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--archivo', default=ARCHIVO)
    parser.add_argument('--criterio', choices=['bic', 'aic'], default='bic')
    parser.add_argument('--familia', choices=['garch_comun', 'auto'], default='garch_comun')
    parser.add_argument('--guardar', action='store_true', help='Guarda resultados junto al Excel, sin crear carpeta')
    parser.add_argument('--sin-mostrar', action='store_true'); parser.add_argument('--sin-graficas', action='store_true')
    parser.add_argument('--sin-sensibilidad', action='store_true', help='Omitir análisis adicional p=1..6')
    if argv is None and ('ipykernel' in sys.modules or 'ipykernel_launcher' in Path(sys.argv[0]).name):
        argv = []
    arg = parser.parse_args(argv); path = resolver_archivo(arg.archivo)
    ruta = path.parent; mostrar = not arg.sin_mostrar
    guardar = arg.guardar; TABLAS.clear()
    FIGURAS.clear(); print('Excel:', path.name, '| Retornos mensuales: 100·Δln(P)', '| familia:', arg.familia)
    print('SHA256 de la base:', hashlib.sha256(path.read_bytes()).hexdigest()); d, imputados = leer_datos(path)
    print('Muestra: 2010–2024 (180 meses) | Saldo=X−M | imputaciones cardamomo:', int(imputados.sum()) if imputados is not None else 'no verificadas')
    emitir('01_Estacionalidad', estacionalidad(d), mostrar, guardar, ruta); comp = []
    params = []; sol = {}
    for p in PRODUCTOS:
        sol[p] = {}; y = d['r_' + p].dropna()
        y.name = 'r_' + p; medias = [1, 2, 3] if p == 'cafe' else [1]
        for m in ['GARCH', 'EGARCH']:
            for k in medias:
                info, sigma, param = garch_mle(y, m, mean_lags=k, burn_in=3 if p == 'cafe' else 1); comp.append(info)
                sol[p][m, k] = (info, sigma); param['producto'] = p
                param['modelo'] = m; param['media_ar'] = k
                params.append(param)
    candidatos = pd.DataFrame(comp); emitir('02_GARCH_EGARCH_Comparacion', candidatos, mostrar, guardar, ruta)
    emitir('03_GARCH_EGARCH_Parametros', pd.concat(params, ignore_index=True), mostrar, guardar, ruta); sel = {}
    dec = []
    for p in PRODUCTOS:
        ci = candidatos[candidatos.producto == p]
        if arg.familia == 'garch_comun':
            cc = ci[(ci.modelo == 'GARCH') & ci.valido_numerico]
            if cc.empty:
                cc = ci[ci.modelo == 'GARCH']
            ok = cc[cc.diagnosticos_5pct]; fila = (ok if not ok.empty else cc).sort_values('BIC').iloc[0]
        else:
            admis = ci[ci.valido_numerico & ci.diagnosticos_5pct]
            if admis.empty:
                admis = ci[ci.valido_numerico]
            if admis.empty:
                admis = ci[ci.converge]
            if admis.empty:
                raise RuntimeError('No hay candidato convergente para ' + p)
            fila = admis.sort_values('BIC').iloc[0]
        if not fila.valido_numerico:
            print('ADVERTENCIA: modelo escogido sin validación numérica plena:', p, fila.modelo)
        sel[p] = fila.modelo; d['vol_' + p] = sol[p][fila.modelo, int(fila.media_ar)][1]
        dec.append({'producto': p, 'modelo': fila.modelo, 'media_ar': int(fila.media_ar), 'BIC': fila.BIC, 'ARCH_LM_antes_p': fila.arch_pre_p, 'LB_residuos_12_p': fila.lb12_p, 'LB_cuadrados_12_p': fila.lb2_12_p, 'ARCH_LM_despues_p': fila.arch_post_12_p, 'nu_t': fila.nu, 'diagnosticos_ok': fila.diagnosticos_5pct, 'precaucion_ARCH_antes': bool(fila.arch_pre_p >= 0.05)})
    emitir('04_Seleccion_GARCH', pd.DataFrame(dec), mostrar, guardar, ruta)
    if imputados is not None and int(imputados.sum()) > 0:
        imp = imputados.astype(bool); alternativas = np.exp(np.log(d.cardamomo).mask(imp).interpolate(method='time', limit_direction='both'))
        yy = (100 * np.log(alternativas).diff()).dropna(); yy.name = 'r_cardamomo'
        infoalt, sigalt, _ = garch_mle(yy, 'GARCH', mean_lags=1, burn_in=3); ind = d['vol_cardamomo'].dropna().index.intersection(sigalt.index)
        orig = d.loc[ind, 'vol_cardamomo']; alt = sigalt.loc[ind]
        emitir('04b_Sensibilidad_cardamomo', pd.DataFrame([{'n_meses_imputados': int(imp.sum()), 'interpolacion': 'lineal en log-precio', 'BIC_alternativo': infoalt['BIC'], 'media_sigma_original': orig.mean(), 'media_sigma_alternativa': alt.mean(), 'correlacion': orig.corr(alt), 'diferencia_absoluta_media': (orig - alt).abs().mean()}]), mostrar, guardar, ruta)
    vols = ['vol_' + p for p in PRODUCTOS]; s = d.vol_petroleo
    delta = s.diff(); delta.loc[s.first_valid_index()] = 0.0
    d['vol_petroleo_pos'] = delta.clip(lower=0).fillna(0).cumsum(); d['vol_petroleo_neg'] = delta.clip(upper=0).fillna(0).cumsum()
    colsint = ['balanza', 'itcer', 'tcn', 'remesas'] + vols + ['vol_petroleo_pos', 'vol_petroleo_neg']; t, inttab, clas = pruebas_integracion(d, colsint)
    emitir('05_ADF_KPSS_y_PP_si_disponible', t, mostrar, guardar, ruta); emitir('06_Integracion', inttab, mostrar, guardar, ruta)
    idx = ['balanza', 'itcer', 'tcn', 'remesas'] + vols + ['vol_petroleo_pos', 'vol_petroleo_neg']; ad = d.dropna(subset=idx).copy()
    ad.index = pd.DatetimeIndex(ad.index, freq='MS'); mat = sm.add_constant(ad[['itcer', 'tcn', 'remesas']].astype(float))
    emitir('06b_VIF_controles', pd.DataFrame({'variable': ['itcer', 'tcn', 'remesas'], 'VIF': [variance_inflation_factor(mat.values, j) for j in (1, 2, 3)]}), mostrar, guardar, ruta)
    cols = {'ARDL': vols + ['itcer', 'tcn', 'remesas'], 'NARDL_exploratorio': [x for x in vols if x != 'vol_petroleo'] + ['vol_petroleo_pos', 'vol_petroleo_neg', 'itcer', 'tcn', 'remesas']}
    tablas = []; fits = {}
    for nombre, xx in cols.items():
        a, f = evaluar_grid(ad, nombre, xx, maxp=3, maxq=2, hold_back=3); tablas.append(a)
        fits.update(f)
    grid = pd.concat(tablas, ignore_index=True)
    for mod in cols:
        n = grid.loc[grid.modelo == mod, 'n'].dropna().unique()
        if len(n) != 1:
            raise AssertionError('Muestras distintas dentro del grid ' + mod)
    emitir('07_Todos_rezagos_muestra_comun', grid, mostrar, guardar, ruta); mejores = mejores_candidatos(grid)
    emitir('07b_Mejores_AIC_BIC', mejores, mostrar, guardar, ruta); seleccion = []
    coeficientes = []; elegido_fits = {}
    for mod in cols:
        c = grid[(grid.modelo == mod) & grid.BIC.notna()]; usar = c[c.diagnosticos_ok]
        eleg = (usar if not usar.empty else c).sort_values(arg.criterio.upper()).iloc[0]; fit = fits[mod, bool(eleg.meses), int(eleg.p), int(eleg.q)]
        elegido_fits[mod] = fit; tab, h = resumen_coef_niveles(fit, mod)
        coeficientes.append(tab)
        seleccion.append({'modelo': mod, 'p': int(eleg.p), 'q': int(eleg.q), 'n': int(eleg.n), 'meses': bool(eleg.meses), 'AIC': float(eleg.AIC), 'BIC': float(eleg.BIC), 'BG6': float(eleg.BG6), 'BG12': float(eleg.BG12), 'BG24': float(eleg.BG24), 'BP': float(eleg.BP), 'CUSUM': float(eleg.CUSUM), 'pasa_diagnosticos': bool(eleg.diagnosticos_ok), 'permite_UECM_desde_ARDL': bool(eleg.q >= 1), 'cointegracion_confirmada': False})
    resumen = pd.DataFrame(seleccion); emitir('08_ARDL_vs_NARDL_exploratorio', resumen, mostrar, guardar, ruta)
    emitir('09_Coeficientes_en_niveles_HAC', pd.concat(coeficientes, ignore_index=True), mostrar, guardar, ruta)
    try:
        f = elegido_fits['NARDL_exploratorio']; _, h = resumen_coef_niveles(f, 'NARDL_exploratorio')
        names = list(f.params.index); R = np.zeros((1, len(names)))
        R[0, names.index('vol_petroleo_pos.L0')] = 1; R[0, names.index('vol_petroleo_neg.L0')] = -1
        w = h.f_test(R)
        emitir('09c_Wald_simetria_WTI_exploratoria', pd.DataFrame([{'F_HAC': float(w.fvalue), 'p_HAC': float(w.pvalue), 'H0': 'coeficientes positivos y negativos iguales', 'alcance': 'exploratorio; NO establece cointegracion ni asimetria de largo plazo'}]), mostrar, guardar, ruta)
    except Exception as exc:
        print('Wald exploratorio no disponible:', repr(exc))
    if int(resumen.loc[resumen.modelo == 'ARDL', 'q'].iloc[0]) == 0:
        try:
            eq, ecm = reparametrizar_ardl_q0(elegido_fits['ARDL']); emitir('09b_Equivalencia_ECM_q0', pd.DataFrame([eq]), mostrar, guardar, ruta)
        except Exception as exc:
            print('ADVERTENCIA: no se pudo verificar ECM restringido q=0:', repr(exc))
    u_info, u_tab = comparar_uecm_candidato(ad, cols['ARDL'], grid, fits, clas)
    if u_info:
        emitir('10_UECM_auxiliar_y_advertencias', pd.DataFrame([u_info]), mostrar, guardar, ruta)
        emitir('10b_UECM_auxiliar_coeficientes', u_tab, mostrar, guardar, ruta)
    extendida = None
    if not arg.sin_sensibilidad:
        piezas = []
        for nombre, xx in cols.items():
            a, _ = evaluar_grid(ad, nombre, xx, maxp=6, maxq=2, hold_back=6); piezas.append(a)
        extendida = pd.concat(piezas, ignore_index=True); emitir('11_Sensibilidad_p1_a6_n172', extendida, mostrar, guardar, ruta)
        emitir('11b_Mejores_sensibilidad', mejores_candidatos(extendida), mostrar, guardar, ruta)
    print('\n=== DICTAMEN AUTOMÁTICO (DESCRIPTIVO, NO CAUSAL) ==='); print(resumen.to_string(index=False))
    pendientes = [x for x in ['balanza', 'itcer', 'tcn', 'remesas'] + vols if clas.get(x) not in ('I(0) compatible', 'I(1) probable')]
    print('\nIntegración sin resolver:', pendientes)
    if pendientes:
        print('No interpretar Bounds/cointegración ni velocidad de corrección del error.')
    print('NOTA: la suma parcial NARDL de una volatilidad I(0) puede producir tendencias artificiales; no afirmar asimetría de largo plazo.')
    print('NOTA: la base comercial contiene ajustes mensuales a referencias anuales; confirmar fuente. Cardamomo: valor unitario, con imputaciones.')
    print('NOTA: GARCH utiliza ML SciPy; sus errores estándar numéricos son aproximados y no se propagan automáticamente a HAC.')
    print('NOTA: diferencias ADF(12)/ADF(14), KPSS y rupturas implican integración PROVISIONAL; no se fuerza I(2).')
    print('NOTA: F-PSS del UECM q>=1 se reporta solo como referencia; NO pertenece al ARDL(q=0) principal.')
    if not arg.sin_graficas:
        graficar(d, candidatos, sel, resumen, mostrar, guardar, ruta); figura_rezagos(grid, mostrar, guardar, ruta)
    if guardar:
        (ruta / 'NOTAS_CRITICAS_MODELO.txt').write_text('No hay cointegración confirmada. La selección es por BIC/AIC con muestra común.\nModelo ARDL(3,0) no entra directamente en UECM.from_ardl; consultar alternativa q>=1.\nLa exploración p=1..6 usa n=172: no comparar su BIC con n=175.\nCambios NARDL de sigma WTI son exploratorios, no asimetría de precio ni causalidad.\nCardamomo valor unitario con imputaciones; comercio mensual requiere conciliación.\n', encoding='utf-8')
    return {'datos': d, 'GARCH': candidatos, 'familias': pd.DataFrame(dec), 'integracion': inttab, 'rezagos': grid, 'mejores': mejores, 'seleccion': resumen, 'sensibilidad6': extendida, 'tablas': TABLAS}

def comparar_nardl_precios_wti(datos, mostrar=True, guardar=False, salida=None, criterio='BIC'):
    if salida is None:
        salida = Path.cwd()
    salida = Path(salida)
    if criterio.upper() not in ('AIC', 'BIC'):
        raise ValueError('criterio debe ser AIC o BIC')
    d = datos.copy(); d['log_wti'] = np.log(d['petroleo'])
    cambios = d['log_wti'].diff().fillna(0.0); d['wti_pos'] = cambios.clip(lower=0).cumsum()
    d['wti_neg'] = cambios.clip(upper=0).cumsum(); error_identidad = float((d['wti_pos'] + d['wti_neg'] - (d['log_wti'] - d['log_wti'].iloc[0])).abs().max())
    if error_identidad >= 1e-09:
        raise AssertionError('No se verifica la identidad de la descomposición del precio WTI.')
    vols = ['vol_' + p for p in PRODUCTOS]; controles = ['itcer', 'tcn', 'remesas']
    candidatos = {'ARDL_original': vols + controles, 'ARDL_precio_WTI': vols + controles + ['log_wti'], 'NARDL_precio_WTI': vols + controles + ['wti_pos', 'wti_neg']}
    requeridas = list(dict.fromkeys(['balanza'] + [v for vv in candidatos.values() for v in vv])); muestra = d[requeridas].dropna().copy()
    muestra.index = pd.DatetimeIndex(muestra.index, freq='MS'); allrows = []
    fits = {}
    for estacional in (False, True):
        fija = meses_dummies(muestra.index) if estacional else None
        for nombre, cols in candidatos.items():
            for p in (1, 2, 3):
                for q in (0, 1, 2):
                    clave = (nombre, estacional, p, q)
                    try:
                        r = ARDL(muestra['balanza'], lags=p, exog=muestra[cols], order=q, fixed=fija, trend='c', hold_back=3, missing='raise').fit()
                        Y = np.asarray(r.model._y, dtype=float); X = np.asarray(r.model._x, dtype=float)
                        ols = sm.OLS(Y, X).fit(); np.testing.assert_allclose(ols.params, np.asarray(r.params), rtol=1e-07, atol=1e-07)
                        bh = sm.OLS(Y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 6, 'use_correction': True}); nombres = list(r.params.index)
                        diagnosticos = {f'BG{h}': float(acorr_breusch_godfrey(ols, nlags=h)[1]) for h in (6, 12, 24)}
                        diagnosticos['BP'] = float(het_breuschpagan(ols.resid, X)[1]); diagnosticos['JB'] = float(jarque_bera(ols.resid)[1])
                        diagnosticos['CUSUM'] = float(breaks_cusumolsresid(ols.resid, ddof=X.shape[1])[1])
                        fila = {'modelo': nombre, 'meses': estacional, 'p': p, 'q': q, 'n': int(r.nobs), 'k': len(r.params), 'AIC': float(r.aic), 'BIC': float(r.bic), **diagnosticos, 'BG_ok': min((diagnosticos[f'BG{i}'] for i in (6, 12, 24))) >= 0.05, 'identidad_wti_error': error_identidad}
                        if nombre == 'NARDL_precio_WTI':
                            R = np.zeros((q + 1, len(nombres)))
                            for lag in range(q + 1):
                                R[lag, nombres.index(f'wti_pos.L{lag}')] = 1.0; R[lag, nombres.index(f'wti_neg.L{lag}')] = -1.0
                            fila['Wald_simetria_todos_HAC_p'] = float(bh.wald_test(R, scalar=True, use_f=True).pvalue)
                            fila['Wald_simetria_suma_HAC_p'] = float(bh.wald_test(R.sum(axis=0, keepdims=True), scalar=True, use_f=True).pvalue)
                            fila['Wald_contemporaneo_HAC_p'] = float(bh.wald_test(R[:1], scalar=True, use_f=True).pvalue)
                        allrows.append(fila); fits[clave] = (r, ols, bh)
                    except Exception as exc:
                        allrows.append({'modelo': nombre, 'meses': estacional, 'p': p, 'q': q, 'error': repr(exc)})
    tabla = pd.DataFrame(allrows)
    for nombre in candidatos:
        n = tabla[(tabla.modelo == nombre) & tabla.AIC.notna()].n.unique()
        if len(n) != 1:
            raise AssertionError(f'La búsqueda de {nombre} no tiene muestra común: {n}')
    emitir('12_NARDL_precio_WTI_todos_los_modelos', tabla, mostrar, guardar, salida); res = []
    for nombre in candidatos:
        datos_m = tabla[(tabla.modelo == nombre) & tabla.AIC.notna()]
        for filtro, subset in [('mínimo sin filtro', datos_m), ('BG6_BG12_BG24_no_rechaza', datos_m[datos_m.BG_ok])]:
            if subset.empty:
                continue
            for ic in ('AIC', 'BIC'):
                m = subset.sort_values(ic).iloc[0].to_dict(); m['criterio'] = ic
                m['restriccion'] = filtro; res.append(m)
    mejores = pd.DataFrame(res); emitir('13_NARDL_precio_WTI_seleccion', mejores, mostrar, guardar, salida)
    anidados = []
    for estacional in (False, True):
        for p in (1, 2, 3):
            for q in (0, 1, 2):
                parte = tabla[(tabla.meses == estacional) & (tabla.p == p) & (tabla.q == q)]
                if len(parte) != 3 or parte.AIC.isna().any():
                    continue
                b = parte.set_index('modelo'); a = b.loc['ARDL_precio_WTI']
                n = b.loc['NARDL_precio_WTI']
                anidados.append({'meses': estacional, 'p': p, 'q': q, 'delta_AIC_NARDL_menos_ARDL_precio': n.AIC - a.AIC, 'delta_BIC_NARDL_menos_ARDL_precio': n.BIC - a.BIC, 'wald_todos_HAC_p': n.get('Wald_simetria_todos_HAC_p', np.nan), 'wald_suma_HAC_p': n.get('Wald_simetria_suma_HAC_p', np.nan), 'BG24_ARDL_precio': a.BG24, 'BG24_NARDL_precio': n.BG24})
    anidados = pd.DataFrame(anidados); emitir('14_NARDL_precio_WTI_comparaciones_anidadas', anidados, mostrar, guardar, salida)
    coefs = []
    for nombre in candidatos:
        sub = mejores[(mejores.modelo == nombre) & (mejores.criterio == 'BIC') & (mejores.restriccion == 'BG6_BG12_BG24_no_rechaza')]
        if sub.empty:
            sub = mejores[(mejores.modelo == nombre) & (mejores.criterio == 'BIC') & (mejores.restriccion == 'mínimo sin filtro')]
        if sub.empty:
            continue
        m = sub.iloc[0]; rr, oo, hac = fits[nombre, bool(m.meses), int(m.p), int(m.q)]
        for j, par in enumerate(rr.params.index):
            coefs.append({'modelo': nombre, 'p': int(m.p), 'q': int(m.q), 'meses': bool(m.meses), 'regresor': par, 'coeficiente': float(rr.params.iloc[j]), 'error_HAC': float(hac.bse[j]), 'p_HAC': float(hac.pvalues[j])})
    coef = pd.DataFrame(coefs); emitir('15_NARDL_precio_WTI_coeficientes_HAC', coef, mostrar, guardar, salida)
    print('\n=== COMPARACIÓN ADICIONAL DE PRECIOS WTI ==='); print('No equivale a una prueba de causalidad ni demuestra cointegración.')
    print('NARDL del precio WTI amplía la pregunta respecto al estudio original de volatilidad.')
    print('La selección con filtros se separa del mínimo AIC/BIC global para evitar sesgo oculto.')
    return {'todos': tabla, 'mejores': mejores, 'anidados': anidados, 'coeficientes': coef, 'muestra': muestra, 'identidad_wti': error_identidad}

def ejecutar_todo(archivo=ARCHIVO, guardar=False, mostrar=True, familia='garch_comun', criterio='bic', sensibilidad=True, graficas=True):
    opciones = ['--archivo', str(archivo), '--familia', familia, '--criterio', criterio]
    if guardar:
        opciones.append('--guardar')
    if not mostrar:
        opciones.append('--sin-mostrar')
    if not sensibilidad:
        opciones.append('--sin-sensibilidad')
    if not graficas:
        opciones.append('--sin-graficas')
    principal = main(opciones); archivo_local = resolver_archivo(archivo)
    adicional = comparar_nardl_precios_wti(principal['datos'], mostrar=mostrar, guardar=guardar, salida=archivo_local.parent, criterio=criterio)
    return {'principal': principal, 'NARDL_precios_WTI': adicional, 'tablas': TABLAS, 'archivo': str(archivo_local)}
from statsmodels.stats.multitest import multipletests; from scipy.stats import t as student_t

def benchmark_varianza_constante(serie, lags=1, quemado=None):
    x = np.asarray(serie.dropna(), float); b = lags if quemado is None else quemado
    n = len(x) - b; X = np.column_stack([np.ones(n)] + [x[b - i:len(x) - i] for i in range(1, lags + 1)])
    beta = np.linalg.lstsq(X, x[b:], rcond=None)[0]; sig = max(np.std(x[b:] - X @ beta), 0.001)

    def nll(par):
        e = x[b:] - X @ par[:lags + 1]; s = np.exp(np.clip(par[-2], -8.0, 12.0))
        nu = 2.05 + np.exp(np.clip(par[-1], -10.0, 7.0))
        return -float(np.sum(student_t.logpdf(e / s * np.sqrt(nu / (nu - 2)), df=nu) - 0.5 * np.log((nu - 2) / nu) - np.log(s)))
    opt = min([minimize(nll, np.r_[beta, np.log(sig), np.log(v - 2.05)], method='L-BFGS-B', options={'maxiter': 350}) for v in (5.0, 10.0, 25.0)], key=lambda v: v.fun)
    k = lags + 3
    return dict(n=n, media_ar=lags, converge=bool(opt.success), AIC=2 * k + 2 * opt.fun, BIC=k * np.log(n) + 2 * opt.fun, nu=2.05 + np.exp(opt.x[-1]), volatilidad_constante=np.exp(opt.x[-2]))

def _fit_ardl_robustez(datos, columnas, p=3, q=0, covariables_fijas=(), temporal=None, hold_back=3):
    if temporal is not None:
        datos = datos.loc[temporal].copy()
    ind = ['balanza'] + list(columnas) + list(covariables_fijas); datos = datos.dropna(subset=ind).copy()
    fix = meses_dummies(datos.index)
    for col in covariables_fijas:
        fix[col] = datos[col]
    fit = ARDL(datos.balanza, lags=p, exog=datos[list(columnas)], order=q, fixed=fix, trend='c', hold_back=hold_back, missing='raise').fit()
    diag, ols = diagnosticos_niveles(fit)
    res = {'modelo': '', 'n': fit.nobs, 'p': p, 'q': q, 'AIC': fit.aic, 'BIC': fit.bic, 'BG6': diag['BG6'], 'BG12': diag['BG12'], 'BG24': diag['BG24'], 'BP': diag['BP'], 'JB': diag['JB'], 'CUSUM': diag['CUSUM'], 'diagnosticos_ok': diag['diagnosticos_ok']}
    return (fit, res)

def _extraer_term(fit, maxlag_hac=6):
    ols = sm.OLS(np.asarray(fit.model._y, float), np.asarray(fit.model._x, float))
    h = ols.fit(cov_type='HAC', cov_kwds={'maxlags': maxlag_hac, 'use_correction': True}); names = list(fit.params.index)
    return {key: {'coef': float(fit.params.iloc[i]), 'p': float(h.pvalues[i]), 'ee': float(h.bse[i])} for i, key in enumerate(names)}

def verificar_series_integration(datos, columnas):
    rows = []
    for col in columnas:
        s = datos[col].dropna()
        for dord in (0, 1, 2):
            ss = s.diff(dord).dropna() if dord == 1 else s.diff().diff().dropna() if dord == 2 else s
            for trend in ('c', 'ct'):
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore'); a = adfuller(ss, regression=trend, maxlag=12, autolag='BIC')
                    a14 = adfuller(ss, regression=trend, maxlag=14, autolag='AIC'); kp = kpss(ss, regression=trend, nlags='auto')
                rows.append(dict(variable=col, diferencias=dord, tendencia=trend, n=len(ss), ADF_12_BIC_p=a[1], ADF_12_BIC_lags=a[2], ADF_14_AIC_p=a14[1], ADF_14_AIC_lags=a14[2], KPSS_p=kp[1], ADF_acuerdo_5pct=bool((a[1] < 0.05) == (a14[1] < 0.05))))
    return pd.DataFrame(rows)

def analizar_coincidencia_uecm(datos, columnas, ardl_p=3):
    df = datos.dropna(subset=['balanza'] + list(columnas)).copy(); fijo = meses_dummies(df.index)
    ar = ARDL(df['balanza'], lags=ardl_p, exog=df[list(columnas)], order=1, fixed=fijo, trend='c', hold_back=3).fit(); u = UECM.from_ardl(ar.model).fit()
    summary = {'especificacion': 'UECM auxiliar p=%d, q=1' % ardl_p, 'observaciones': int(u.nobs), 'AIC_ARDL_auxiliar': float(ar.aic), 'BIC_ARDL_auxiliar': float(ar.bic), 'F_PSS': np.nan, 'I0_5': np.nan, 'I1_5': np.nan, 'PSS_supera_I1_5': False, 'coef_y_L1': float(u.params.get('balanza.L1', np.nan)), 'notas': 'Los contrastes del q=1 NO se atribuyen al ARDL q=0 seleccionado.'}
    try:
        b = u.bounds_test(case=3, asymptotic=True)
        summary.update(F_PSS=float(b.stat), I0_5=float(b.crit_vals.loc[95.0, 'lower']), I1_5=float(b.crit_vals.loc[95.0, 'upper']), PSS_supera_I1_5=bool(b.stat > b.crit_vals.loc[95.0, 'upper']))
    except Exception as ex:
        summary['notas'] += ' Bounds: ' + repr(ex)
    U = sm.OLS(np.asarray(u.model._y, float), np.asarray(u.model._x, float)).fit(cov_type='HAC', cov_kwds={'maxlags': 6, 'use_correction': True})
    ix = list(u.params.index); R = np.zeros((len(columnas), len(ix)))
    for j, col in enumerate(columnas):
        R[j, ix.index(col + '.L1')] = 1
    summary['F_niveles_X_HAC_auxiliar'] = float(U.f_test(R).fvalue); summary['p_niveles_X_HAC_auxiliar'] = float(U.f_test(R).pvalue)
    summary['p_y_L1_HAC_auxiliar'] = float(U.pvalues[ix.index('balanza.L1')]); summary['t_y_L1_HAC_auxiliar'] = float(U.tvalues[ix.index('balanza.L1')])
    return (pd.DataFrame([summary]), ar, u)

def auditoria_robustez(datos, muestras='ambas', familia_alt=False, mostrar=True, guardar=False, directorio=None):
    direct = Path.cwd() if directorio is None else Path(directorio); vols = ['vol_' + p for p in PRODUCTOS]
    cols = vols + ['itcer', 'tcn', 'remesas']; df = datos.dropna(subset=['balanza'] + cols).copy()
    df.index = pd.DatetimeIndex(df.index, freq='MS'); filas = []
    pval = []; fitbase, base = _fit_ardl_robustez(df, cols)
    base.update(modelo='ARDL_base_p3q0'); filas.append(base)
    p_hac6 = _extraer_term(fitbase, 6)
    for col in vols:
        term = col + '.L0'; pval.append(p_hac6[term]['p'])
    adj = multipletests(pval, method='holm')[1]
    tabla_multiples = pd.DataFrame({'producto': PRODUCTOS, 'p_sin_ajustar': pval, 'p_ajustado_Holm_5_pruebas': adj, 'rechaza_al_5pct_Holm': adj < 0.05})
    emitir('16_Ajuste_por_cinco_pruebas_Holm', tabla_multiples, mostrar, guardar, direct)
    variantes = [('COVID_2020_2021', cols, ['covid']), ('Sin_cafe', [x for x in cols if x != 'vol_cafe'], []), ('Sin_azucar', [x for x in cols if x != 'vol_azucar'], []), ('Sin_banano', [x for x in cols if x != 'vol_banano'], []), ('Sin_cardamomo', [x for x in cols if x != 'vol_cardamomo'], []), ('Sin_WTI', [x for x in cols if x != 'vol_petroleo'], []), ('Solo_2010_2019', cols, []), ('Sin_2020_2021', cols, []), ('Desde_2016', cols, [])]
    coef = []
    for nombre, subcols, extra in [('ARDL_base_p3q0', cols, [])] + variantes:
        try:
            temp = df
            if nombre == 'Solo_2010_2019':
                temp = df[df.index < '2020-01-01']
            if nombre == 'Sin_2020_2021':
                temp = df[(df.index < '2020-01-01') | (df.index >= '2022-01-01')]
            if nombre == 'Desde_2016':
                temp = df[df.index >= '2016-01-01']
            if nombre == 'Sin_2020_2021':
                filas.append({'modelo': nombre, 'estado': 'NO ESTIMADO: salto entre 2019 y 2022 invalida rezagos contiguos'}); continue
            if nombre == 'ARDL_base_p3q0':
                fit, rec = (fitbase, dict(base))
            else:
                fit, rec = _fit_ardl_robustez(temp, subcols, covariables_fijas=extra)
            rec.update(modelo=nombre, estado='estimado'); filas.append(rec) if nombre != 'ARDL_base_p3q0' else None
            for ventana in (3, 6, 12):
                coefmap = _extraer_term(fit, ventana)
                for var in vols + ['covid']:
                    key = var + '.L0' if var in vols else var
                    if key not in coefmap:
                        if var == 'covid':
                            matches = [k for k in coefmap if 'covid' in k.lower()]; key = matches[0] if matches else None
                    if key in coefmap:
                        v = coefmap[key]
                        coef.append(dict(modelo=nombre, variable=var, HAC_maxlags=ventana, coeficiente=v['coef'], error_estandar=v['ee'], p_HAC=v['p'], n=int(fit.nobs)))
        except Exception as e:
            filas.append({'modelo': nombre, 'estado': 'ERROR: ' + repr(e)})
    tabla_sens = pd.DataFrame(filas); tabla_coef = pd.DataFrame(coef)
    emitir('17_Robustez_COVID_LeaveOneOut_Subperiodos', tabla_sens, mostrar, guardar, direct)
    emitir('18_Robustez_Coeficientes_HAC_3_6_12', tabla_coef, mostrar, guardar, direct); alternativas = []
    if familia_alt:
        replaced = datos.copy()
        for p in PRODUCTOS:
            y = datos['r_' + p].dropna(); y.name = 'r_' + p
            m = 2 if p == 'cafe' else 1; info, sigma, _ = garch_mle(y, 'EGARCH', mean_lags=m, burn_in=3 if p == 'cafe' else 1)
            alternativas.append(dict(producto=p, EGARCH_BIC=info['BIC'], EGARCH_valido=info['valido_numerico'], EGARCH_diag=info['diagnosticos_5pct']))
            if info['valido_numerico'] and info['diagnosticos_5pct']:
                replaced['vol_' + p] = sigma
        try:
            altfit, altres = _fit_ardl_robustez(replaced, cols); altres['modelo'] = 'ARDL_volatilidades_EGARCH_validas'
            alternativas.append(altres); mm = _extraer_term(altfit)
            for c in vols:
                coef.append(dict(modelo='ARDL_volatilidades_EGARCH_validas', variable=c, HAC_maxlags=6, coeficiente=mm[c + '.L0']['coef'], error_estandar=mm[c + '.L0']['ee'], p_HAC=mm[c + '.L0']['p'], n=altfit.nobs))
            tabla_coef = pd.DataFrame(coef); emitir('18_Robustez_Coeficientes_HAC_3_6_12', tabla_coef, mostrar, guardar, direct)
        except Exception as e:
            alternativas.append(dict(error=repr(e)))
    emitir('19_Sensibilidad_GARCH_EGARCH_En_ARDL', pd.DataFrame(alternativas) if alternativas else pd.DataFrame([{'nota': 'No ejecutado. Activar familia_alt=True para comparar las segundas etapas.'}]), mostrar, guardar, direct)
    return {'base': fitbase, 'multiples': tabla_multiples, 'modelos': tabla_sens, 'coeficientes': tabla_coef, 'alternativas': pd.DataFrame(alternativas)}

def ejecutar_revision_terna(archivo=ARCHIVO, mostrar=True, guardar=False, graficas=True, sensibilidad_6=False, alternativos_egarch=True, precios_wti=True):
    args = ['--archivo', str(archivo), '--criterio', 'bic', '--familia', 'garch_comun']
    if not mostrar:
        args += ['--sin-mostrar']
    if guardar:
        args += ['--guardar']
    if not graficas:
        args += ['--sin-graficas']
    if not sensibilidad_6:
        args += ['--sin-sensibilidad']
    base = main(args); df = base['datos']
    path = resolver_archivo(archivo); out = path.parent
    columnas = ['balanza', 'itcer', 'tcn', 'remesas'] + ['vol_' + p for p in PRODUCTOS]; panel = verificar_series_integration(df, columnas)
    emitir('20_Integracion_sensibilidad_detallada', panel, mostrar, guardar, out)
    aux, fitq1, u = analizar_coincidencia_uecm(df, ['vol_' + p for p in PRODUCTOS] + ['itcer', 'tcn', 'remesas'])
    emitir('21_Bounds_PSS_SOLO_modelo_auxiliar_q1', aux, mostrar, guardar, out); benchmark = []
    for p in PRODUCTOS:
        m = 2 if p == 'cafe' else 1; c = benchmark_varianza_constante(df['r_' + p], lags=m, quemado=3 if p == 'cafe' else 1)
        g = base['GARCH']; sel = base['familias'].query('producto == @p').iloc[0]
        benchmark.append(dict(producto=p, model_constant='AR-t varianza constante', BIC_constante=c['BIC'], BIC_GARCH_elegido=float(sel.BIC), Delta_BIC_GARCH_menos_constante=float(sel.BIC - c['BIC']), GARCH_favorecido_por_BIC=bool(sel.BIC < c['BIC']), ARCH_LM_pre_p=float(sel.ARCH_LM_antes_p)))
    emitir('22_GARCH_vs_varianza_constante_tStudent', pd.DataFrame(benchmark), mostrar, guardar, out)
    robusto = auditoria_robustez(df, mostrar=mostrar, guardar=guardar, directorio=out, familia_alt=alternativos_egarch); comp = None
    if precios_wti:
        comp = comparar_nardl_precios_wti(df, mostrar=mostrar, guardar=guardar, salida=out)
    emit = pd.DataFrame([{'revision': 'Diagnósticos de niveles del modelo principal', 'estado': 'Estimado: no confirma causalidad/cointegración'}, {'revision': 'AIC/BIC y muestras comunes', 'estado': 'Ejecutado'}, {'revision': 'Contraste PSS de ARDL seleccionado q=0', 'estado': 'NO INTERPRETAR con los valores críticos del UECM auxiliar q=1'}, {'revision': 'Posibles I(2) y rupturas', 'estado': 'Clasificación condicionada por ADF/KPSS/PP y sensibilidad'}, {'revision': 'Prueba de asimetría NARDL', 'estado': 'Exploratoria de WTI; no implica causalidad'}, {'revision': 'Incertidumbre primera etapa GARCH', 'estado': 'NO propagada a HAC; sensibilidad entre familias no la sustituye'}, {'revision': 'Cardamomo 23 imputaciones', 'estado': 'Mantener advertencia y sensibilidad'}, {'revision': 'Fuente comercio mensual pre2016', 'estado': 'Verificación externa de fuentes pendiente'}])
    emitir('23_DICTAMEN_PENDIENTES_Y_ALCANCES', emit, mostrar, guardar, out)
    return {'base': base, 'integracion': panel, 'bounds_auxiliar': aux, 'benchmark_varianza': pd.DataFrame(benchmark), 'robustez': robusto, 'NARDL_precio': comp, 'tablas': TABLAS}

def prueba_volatilidad_movil(datos, mostrar=True, guardar=False, destino=None):
    destino = Path.cwd() if destino is None else Path(destino); cuadros = []
    coef = []
    for ventana in (6, 12):
        df = datos.copy(); colsG = ['vol_' + p for p in PRODUCTOS] + ['itcer', 'tcn', 'remesas']
        for p in PRODUCTOS:
            df['mov_' + p] = df['r_' + p].rolling(ventana, min_periods=ventana).std(ddof=1)
        colsR = ['mov_' + p for p in PRODUCTOS] + ['itcer', 'tcn', 'remesas']; comun = df.dropna(subset=['balanza'] + colsG + colsR).copy()
        for nom, cols in [('GARCH', colsG), ('Vol_movil_' + str(ventana), colsR)]:
            fit, reg = _fit_ardl_robustez(comun, cols); reg.update(tipo=nom, ventana_meses=ventana, fecha_inicio=str(comun.index.min().date()))
            cuadros.append(reg); c = _extraer_term(fit, 6)
            for p in PRODUCTOS:
                k = 'vol_' + p + '.L0' if nom == 'GARCH' else 'mov_' + p + '.L0'
                coef.append(dict(medida=nom, ventana_meses=ventana, producto=p, coeficiente=c[k]['coef'], p_HAC=c[k]['p'], n=fit.nobs))
    t1 = pd.DataFrame(cuadros); t2 = pd.DataFrame(coef)
    emitir('24_Volatilidad_movil_vs_GARCH_misma_muestra', t1, mostrar, guardar, destino)
    emitir('25_Coeficientes_volatilidad_movil', t2, mostrar, guardar, destino); return {'comparacion': t1, 'coeficientes': t2}
_ejecutar_revision_original = ejecutar_revision_terna

def ejecutar_revision_terna(archivo=ARCHIVO, mostrar=True, guardar=False, graficas=True, sensibilidad_6=False, alternativos_egarch=True, precios_wti=True):
    r = _ejecutar_revision_original(archivo=archivo, mostrar=mostrar, guardar=guardar, graficas=graficas, sensibilidad_6=sensibilidad_6, alternativos_egarch=alternativos_egarch, precios_wti=precios_wti)
    r['volatilidad_movil'] = prueba_volatilidad_movil(r['base']['datos'], mostrar=mostrar, guardar=guardar, destino=resolver_archivo(archivo).parent)
    return r

def bootstrap_condicional_ardl_q0(datos, replicas=299, semilla=2026, block=6, mostrar=True, guardar=False, destino=None):
    if not 100 <= replicas <= 5000:
        raise ValueError('Usar 100..5000 replicaciones.')
    destino = Path.cwd() if destino is None else Path(destino); cols = ['vol_' + p for p in PRODUCTOS] + ['itcer', 'tcn', 'remesas']
    frame = datos[['balanza'] + cols].dropna().copy(); T = len(frame)
    y = frame.balanza.to_numpy(float); exog = frame[cols].to_numpy(float)
    meses = meses_dummies(frame.index).to_numpy(float); p = 3
    h = 3; n = T - h

    def diseno_alt(yy):
        return np.column_stack([np.ones(n)] + [yy[h - j:T - j] for j in range(1, p + 1)] + [exog[h:], meses[h:]])
    alt = diseno_alt(y); R = np.zeros((1 + len(cols), alt.shape[1]))
    q = np.zeros(len(R)); R[0, 1:1 + p] = 1.0
    q[0] = 1.0
    for k in range(len(cols)):
        R[k + 1, 1 + p + k] = 1.0

    def estadistico_f(yy):
        xx = diseno_alt(yy); resp = yy[h:]
        b = np.linalg.lstsq(xx, resp, rcond=None)[0]; resid = resp - xx @ b
        inv = np.linalg.pinv(xx.T @ xx); z = R @ b - q
        cov = R @ inv @ R.T; F = float(z @ np.linalg.pinv(cov) @ z / len(R) / (resid @ resid / (len(resp) - xx.shape[1])))
        return F
    Fobs = estadistico_f(y); dy = np.r_[np.nan, np.diff(y)]
    zz = np.column_stack([np.ones(n), dy[h - 1:T - 1], dy[h - 2:T - 2], meses[h:]]); bc = np.linalg.lstsq(zz, dy[h:], rcond=None)[0]
    errors = dy[h:] - zz @ bc; errors = errors - errors.mean()
    rng = np.random.default_rng(semilla); Fs = []
    for i in range(replicas):
        blocks = []
        while len(blocks) < n:
            inicio = int(rng.integers(0, n)); blocks.extend(errors.take((inicio + np.arange(block)) % n).tolist())
        shock = np.array(blocks[:n]); yb = y.copy()
        dyb = dy.copy()
        for t in range(h, T):
            D = np.r_[1.0, dyb[t - 1], dyb[t - 2], meses[t]]; dyb[t] = float(D @ bc + shock[t - h])
            yb[t] = yb[t - 1] + dyb[t]
        f = estadistico_f(yb)
        if np.isfinite(f):
            Fs.append(f)
    Fs = np.array(Fs)
    res = pd.DataFrame([{'contraste': 'Bootstrap condicional H0 niveles ARDL(3,0)', 'F_muestra': Fobs, 'replicas_validas': len(Fs), 'p_bootstrap_condicional': float((1 + sum(Fs >= Fobs)) / (1 + len(Fs))), 'F_simulada_95pct': float(np.quantile(Fs, 0.95)), 'F_simulada_99pct': float(np.quantile(Fs, 0.99)), 'longitud_bloque': block, 'semilla': semilla, 'limitacion': 'NO es prueba PSS formal; X fijo, supuesto de exogeneidad y proceso Δy bajo H0.'}])
    emitir('26_Bootstrap_condicional_q0_NO_PSS', res, mostrar, guardar, destino); return res
_ejecutar_final_sin_bootstrap = ejecutar_revision_terna

def ejecutar_revision_terna(archivo=ARCHIVO, mostrar=True, guardar=False, graficas=True, sensibilidad_6=False, alternativos_egarch=True, precios_wti=True, bootstrap_q0=True, replicas_bootstrap=299):
    x = _ejecutar_final_sin_bootstrap(archivo, mostrar, guardar, graficas, sensibilidad_6, alternativos_egarch, precios_wti)
    if bootstrap_q0:
        x['bootstrap_q0'] = bootstrap_condicional_ardl_q0(x['base']['datos'], replicas=replicas_bootstrap, mostrar=mostrar, guardar=guardar, destino=resolver_archivo(archivo).parent)
    return x

def toda_yamamoto_tesis(datos, mostrar=True, guardar=False, graficas=True, destino=None, criterio='BIC', p_principal_max=3, p_sensibilidad_max=10, dmax_principal=2):
    from statsmodels.stats.multitest import multipletests; from scipy.stats import chi2
    if criterio.upper() not in ('AIC', 'BIC'):
        raise ValueError('criterio debe ser AIC o BIC')
    if not 1 <= p_principal_max <= p_sensibilidad_max <= 12:
        raise ValueError('La búsqueda principal y sensibilidad de p son incompatibles')
    if not 1 <= dmax_principal <= 2:
        raise ValueError('La sensibilidad está definida para dmax=1 y 2')
    cols = ['balanza'] + [f'vol_{p}' for p in PRODUCTOS]; comun = datos[cols].dropna().copy()
    if len(comun) < 100:
        raise ValueError('Datos mensuales insuficientes para TY')
    if not comun.index.equals(pd.date_range(comun.index[0], periods=len(comun), freq='MS')):
        raise ValueError('Los sistemas TY requieren meses consecutivos sin huecos')
    dmax_comparacion = 2; hold_back = p_sensibilidad_max + dmax_comparacion
    nobs = len(comun) - hold_back
    if nobs <= 70:
        raise ValueError('Muestra TY demasiado pequeña tras los rezagos')
    if not set(PRODUCTOS).issubset(set(NOMBRES)):
        raise RuntimeError('Listado de productos inconsistente')
    m = meses_dummies(comun.index).to_numpy(dtype=float); series = {}
    for c in cols:
        vals = comun[c].to_numpy(dtype=float); sd = np.std(vals, ddof=1)
        if not (np.isfinite(sd) and sd > 0):
            raise ValueError('Varianza cero: ' + c)
        series[c] = (vals - np.mean(vals)) / sd
    modelos = {}; resumen = []
    tolerancia = np.finfo(float).eps

    def evaluar(producto, p, dmax):
        y = series['balanza']; x = series[f'vol_{producto}']
        lag_total = p + dmax; terminos = [np.ones(nobs)]
        cols_design = ['const']
        for j in range(1, lag_total + 1):
            terminos.extend((y[hold_back - j:len(y) - j], x[hold_back - j:len(x) - j])); cols_design.extend((f'balanza_L{j}', f'vol_L{j}'))
        for k in range(11):
            terminos.append(m[hold_back:, k]); cols_design.append(f'mes_{k + 2}')
        X = np.column_stack(terminos); Y = np.column_stack((y[hold_back:], x[hold_back:]))
        if np.linalg.matrix_rank(X) != X.shape[1]:
            raise np.linalg.LinAlgError(f'Rango insuficiente VAR {producto}, p={p}, dmax={dmax}')
        fit_s = sm.OLS(Y[:, 0], X).fit(); fit_v = sm.OLS(Y[:, 1], X).fit()
        resid = np.column_stack((fit_s.resid, fit_v.resid)); cov = np.asarray(resid.T @ resid / nobs, float)
        sign, logdet = np.linalg.slogdet(cov)
        if sign <= 0:
            raise np.linalg.LinAlgError('Covarianza VAR no definida positiva')
        X_p = np.column_stack([np.ones(nobs)] + [arr[hold_back - j:len(arr) - j] for j in range(1, p + 1) for arr in (y, x)] + [m[hold_back:, k] for k in range(11)])
        fit_p_y = sm.OLS(Y[:, 0], X_p).fit(); fit_p_x = sm.OLS(Y[:, 1], X_p).fit()
        res_p = np.column_stack((fit_p_y.resid, fit_p_x.resid)); signo, ld = np.linalg.slogdet(res_p.T @ res_p / nobs)
        if signo <= 0:
            raise np.linalg.LinAlgError('Covarianza VAR(p) no positiva')
        k_ols = X_p.shape[1] * 2; aic = ld + 2 * k_ols / nobs
        bic = ld + np.log(nobs) * k_ols / nobs; R = np.zeros((p, X.shape[1]))
        for j in range(1, p + 1):
            R[j - 1, cols_design.index(f'vol_L{j}')] = 1.0
        w = fit_s.wald_test(R, use_f=False, scalar=True); f_rob = fit_s.get_robustcov_results(cov_type='HC3')
        wr = f_rob.wald_test(R, use_f=False, scalar=True); bg_y = acorr_breusch_godfrey(fit_s, nlags=12)
        bg_x = acorr_breusch_godfrey(fit_v, nlags=12); bg_y6 = acorr_breusch_godfrey(fit_s, nlags=6)
        bg_x6 = acorr_breusch_godfrey(fit_v, nlags=6)
        return dict(producto=producto, p=int(p), dmax=int(dmax), lag_total=int(lag_total), fecha_inicio=str(comun.index[hold_back].date()), fecha_final=str(comun.index[-1].date()), n=int(nobs), k_ecuacion=int(X.shape[1]), AIC_VAR=float(aic), BIC_VAR=float(bic), criterio_calculado_en='VAR(p) no aumentado; misma muestra', MWALD_chi2=float(np.asarray(w.statistic).item()), grados_libertad_wald=p, p_convencional=float(w.pvalue), MWALD_HC3_chi2=float(np.asarray(wr.statistic).item()), p_HC3=float(wr.pvalue), BG12_saldo_p=float(bg_y[1]), BG12_vol_p=float(bg_x[1]), BG6_saldo_p=float(bg_y6[1]), BG6_vol_p=float(bg_x6[1]), BG12_ambas_ok=bool(bg_y[1] >= 0.05 and bg_x[1] >= 0.05), termino_extra_excluido_de_Wald=True)
    filas = []
    for producto in PRODUCTOS:
        for dmax in [dmax_principal, 1] if dmax_principal != 1 else [1, 2]:
            for p in range(1, p_sensibilidad_max + 1):
                try:
                    filas.append(evaluar(producto, p, dmax))
                except (ValueError, np.linalg.LinAlgError) as e:
                    filas.append({'producto': producto, 'p': p, 'dmax': dmax, 'error': str(e)})
    grid = pd.DataFrame(filas); errores = grid[grid.get('error', pd.Series(index=grid.index, dtype=object)).notna()]
    if not errores.empty:
        print('ATENCIÓN: Algunas especificaciones TY no estimables:\n', errores.to_string(index=False))
    valid = grid[grid['BIC_VAR'].notna()].copy()
    if valid.empty:
        raise RuntimeError('No hubo especificaciones TY válidas')
    for dmax in (dmax_principal, 1 if dmax_principal != 1 else 2):
        for pr in PRODUCTOS:
            c = valid[(valid.producto == pr) & (valid.dmax == dmax) & (valid.p <= p_principal_max)].copy()
            if c.empty:
                raise RuntimeError(f'Sin candidato TY para {pr}, dmax={dmax}')
            chosen = c.sort_values([criterio.upper() + '_VAR', 'p']).iloc[0].copy()
            chosen['tipo'] = 'principal' if dmax == dmax_principal else 'sensibilidad_dmax'; resumen.append(chosen)
    sel = pd.DataFrame(resumen); principal = sel[sel.tipo == 'principal'].copy()
    for col in ('p_convencional', 'p_HC3'):
        padj = multipletests(principal[col].astype(float).to_numpy(), alpha=0.05, method='holm')[1]
        principal['p_Holm_' + ('convencional' if col == 'p_convencional' else 'HC3')] = padj
    principal['interpretacion'] = np.where(principal['BG12_ambas_ok'], 'Contraste reportable con BG12 favorable', 'ADVERTENCIA: autocorrelación en al menos una ecuación; no concluir precedencia sin corregir')
    assert principal.n.nunique() == 1, 'TY muestra diferente por producto'; assert principal.termino_extra_excluido_de_Wald.all()
    d1 = sel[sel.tipo == 'sensibilidad_dmax'].copy()
    for col in ('p_convencional', 'p_HC3'):
        d1['p_Holm_' + ('convencional' if col == 'p_convencional' else 'HC3')] = multipletests(d1[col].astype(float).to_numpy(), method='holm')[1]
    destino = Path.cwd() if destino is None else Path(destino); emitir('27_TY_grid_AIC_BIC_muestra_comun', grid, mostrar, guardar, destino)
    emitir('28_Toda_Yamamoto_principal_Holm', principal, mostrar, guardar, destino)
    emitir('29_Toda_Yamamoto_sensibilidad_dmax', d1, mostrar, guardar, destino)
    obs = 'TY: cinco sistemas bivariados (saldo, sigma GARCH); 11 dummies de mes; ' + f'muestra común n={nobs}; p seleccionado por {criterio.upper()} del VAR(p) no aumentado entre 1 y {p_principal_max}; ' + f'comparaciones hasta p={p_sensibilidad_max} en la misma muestra; dmax={dmax_principal} ' + 'y sensibilidad dmax=1. Wald chi-cuadrado restringe solo los p primeros ' + 'rezagos de volatilidad; aplica Holm a las 5 pruebas. Si BG12 rechaza ' + 'en alguna ecuación, el p-valor requiere cautela. La cota dmax=2 es ' + 'un supuesto conservador que NO demuestra que las series sean I(2). ' + 'Inferencia condicional a las volatilidades previamente estimadas. ' + 'No establece causalidad económica ni cointegración.'
    emitir('30_Toda_Yamamoto_notas_alcance', pd.DataFrame([{'nota': obs}]), mostrar, guardar, destino)
    print('\n=== TODA–YAMAMOTO EN EL ALCANCE DE LA TESIS ===')
    print(principal[['producto', 'p', 'dmax', 'n', 'p_convencional', 'p_HC3', 'p_Holm_convencional', 'p_Holm_HC3', 'BG12_ambas_ok']].to_string(index=False))
    print(obs)
    if graficas:
        xx = np.arange(len(PRODUCTOS)); fig, ax = plt.subplots(figsize=(9, 4.6))
        ax.bar(xx - 0.18, principal.set_index('producto').loc[PRODUCTOS, 'p_Holm_convencional'], width=0.36, label='Wald convencional + Holm')
        ax.bar(xx + 0.18, principal.set_index('producto').loc[PRODUCTOS, 'p_Holm_HC3'], width=0.36, label='Wald HC3 + Holm')
        ax.axhline(0.05, color='black', ls='--', lw=1, label='Nivel de significancia 5%')
        ax.set_ylim(0, max(0.2, float(principal[['p_Holm_convencional', 'p_Holm_HC3']].max().max()) * 1.12))
        ax.set_xticks(xx, [NOMBRES[p] for p in PRODUCTOS]); ax.set_ylabel('p-valor ajustado Holm')
        ax.set_title('Toda–Yamamoto: contenido predictivo de las volatilidades'); ax.legend(loc='upper right', fontsize=8)
        fig.tight_layout(); FIGURAS['Figura_09_Toda_Yamamoto_tesis'] = fig
        if guardar:
            fig.savefig(destino / 'Figura_09_Toda_Yamamoto_tesis.png', dpi=180, bbox_inches='tight')
        if mostrar:
            plt.show()
        else:
            plt.close(fig)
    return {'grid': grid, 'principal': principal, 'sensibilidad': d1, 'metodologia': obs, 'muestra': comun}
_ejecutar_sin_toda_yamamoto = ejecutar_revision_terna

def ejecutar_revision_terna(archivo=ARCHIVO, mostrar=True, guardar=False, graficas=True, sensibilidad_6=False, alternativos_egarch=True, precios_wti=False, bootstrap_q0=False, replicas_bootstrap=299, toda_yamamoto=True, ty_criterio='BIC', ty_p_principal_max=3, ty_p_sensibilidad_max=10):
    x = _ejecutar_sin_toda_yamamoto(archivo=archivo, mostrar=mostrar, guardar=guardar, graficas=graficas, sensibilidad_6=sensibilidad_6, alternativos_egarch=alternativos_egarch, precios_wti=precios_wti, bootstrap_q0=bootstrap_q0, replicas_bootstrap=replicas_bootstrap)
    if toda_yamamoto:
        x['Toda_Yamamoto'] = toda_yamamoto_tesis(x['base']['datos'], mostrar=mostrar, guardar=guardar, graficas=graficas, destino=resolver_archivo(archivo).parent, criterio=ty_criterio, p_principal_max=ty_p_principal_max, p_sensibilidad_max=ty_p_sensibilidad_max)
    return x
if __name__ == '__main__':
    RESULTADOS = ejecutar_revision_terna()
