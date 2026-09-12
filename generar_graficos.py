"""
Genera todos los graficos del notebook con estilo custom, los guarda en plots/.

Uso:
    python generar_graficos.py

Para exportar con fondo transparente en vez del violeta, poner TRANSPARENT = True.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.stats as st

from sklearn.datasets import load_diabetes
from sklearn.linear_model import (
    LinearRegression, Ridge, BayesianRidge, ARDRegression
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures


# ==========================================================================
# Config de estilo
# ==========================================================================
BG        = '#1a0e79'   # violeta profundo, fondo
FG        = '#ffffff'   # blanco, texto
YELLOW    = '#ffc857'   # amarillo, dato principal
CELESTE   = '#c7c2ef'   # celeste, secundario y bandas

TRANSPARENT = False     # True para fondo transparente en el PNG

FONT_FAMILY = [
    'ITC Avant Garde Gothic Std',
    'ITC Avant Garde Gothic',
    'ITCAvantGardeStd-Bk',
    'Avant Garde',
    'Century Gothic',
    'Futura',
    'Helvetica',
    'Arial',
    'sans-serif',
]

mpl.rcParams.update({
    'figure.facecolor':   BG,
    'axes.facecolor':     BG,
    'savefig.facecolor':  BG,
    'font.family':        FONT_FAMILY,
    'font.size':          12,
    'axes.edgecolor':     FG,
    'axes.labelcolor':    FG,
    'axes.titlecolor':    FG,
    'axes.titlesize':     16,
    'axes.titleweight':   'bold',
    'axes.labelsize':     13,
    'axes.linewidth':     0.8,
    'xtick.color':        FG,
    'ytick.color':        FG,
    'text.color':         FG,
    'legend.labelcolor':  FG,
    'legend.facecolor':   (0, 0, 0, 0),
    'legend.frameon':     False,
    'legend.fontsize':    11,
    'grid.color':         FG,
    'grid.alpha':         0.12,
    'axes.grid':          True,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'figure.dpi':         110,
    'savefig.dpi':        160,
    'savefig.bbox':       'tight',
})

OUT = 'plots'
os.makedirs(OUT, exist_ok=True)


def save(fig, name):
    fig.savefig(os.path.join(OUT, f'{name}.png'), transparent=TRANSPARENT)
    plt.close(fig)


def clean(ax):
    ax.tick_params(length=3, width=0.6)


# ==========================================================================
# Datos sinteticos, caso cuadratico grado 16
# ==========================================================================
np.random.seed(12345)
num_obs = 20
x = -1.0 + 2 * np.random.rand(num_obs)
w0_true, w1_true, w2_true, sigma_true = -0.5, 1.0, 2.0, 0.1
y = (
    w0_true + w1_true * x + w2_true * x ** 2
    + sigma_true * np.random.randn(num_obs)
)


def poly_phi(z, degree):
    z2 = z[:, None] if z.ndim == 1 else z
    return np.hstack([z2 ** i for i in range(degree + 1)])


degree = 16
Phi = poly_phi(x, degree)

xx = np.linspace(-1, 1, 300)
yy_true = w0_true + w1_true * xx + w2_true * xx ** 2
Phi_xx = poly_phi(xx, degree)

modelos = {
    'OLS':            LinearRegression(fit_intercept=False),
    'Ridge':          Ridge(alpha=1.0, fit_intercept=False),
    'Bayesian Ridge': BayesianRidge(fit_intercept=False),
    'ARD':            ARDRegression(fit_intercept=False),
}
for m in modelos.values():
    m.fit(Phi, y)
coefs = {name: m.coef_ for name, m in modelos.items()}

true_coef = np.zeros(degree + 1)
true_coef[:3] = [w0_true, w1_true, w2_true]


# ==========================================================================
# 1, datos observados
# ==========================================================================
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(x, y, 'x', color=YELLOW, markersize=12, mew=2.2, label='Datos observados')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Datos sinteticos del caso cuadratico')
ax.legend()
clean(ax)
save(fig, '01_datos_observados')


# ==========================================================================
# 2, curvas ajustadas por los cuatro modelos
# ==========================================================================
fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True, sharey=True)
axes = axes.flatten()
for i, (name, m) in enumerate(modelos.items()):
    ax = axes[i]
    ax.plot(xx, yy_true, '--', color=CELESTE, lw=2.2, label='Verdadera')
    ax.plot(xx, m.predict(Phi_xx), '-', color=YELLOW, lw=2.8, label='Prediccion')
    ax.plot(x, y, 'x', color=FG, markersize=9, mew=1.6, label='Observado')
    ax.set_title(name)
    ax.set_ylim(-1.5, 3.5)
    if i >= 2:
        ax.set_xlabel('x')
    if i % 2 == 0:
        ax.set_ylabel('y')
    if i == 0:
        ax.legend(loc='upper left', fontsize=10)
    clean(ax)
plt.tight_layout()
save(fig, '02_curvas_ajustadas')


# ==========================================================================
# 3, coeficientes estimados vs verdaderos
# ==========================================================================
idx_true = np.where(true_coef != 0)[0]
fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
axes = axes.flatten()
for i, (name, coef) in enumerate(coefs.items()):
    ax = axes[i]
    ax.vlines(range(degree + 1), 0, coef, colors=YELLOW, linewidth=1.4, alpha=0.7)
    ax.scatter(range(degree + 1), coef, color=YELLOW, s=45, zorder=4,
               edgecolors=FG, linewidth=0.4, label='Estimado')
    ax.scatter(idx_true, true_coef[idx_true], color=CELESTE, s=110, marker='X',
               edgecolors=FG, linewidth=0.8, zorder=5, label='Verdadero')
    ax.axhline(0, color=FG, lw=0.5, alpha=0.35)
    ax.set_title(name)
    if i >= 2:
        ax.set_xlabel('Grado')
    if i % 2 == 0:
        ax.set_ylabel('Coeficiente')
    if i == 0:
        ax.legend(loc='best', fontsize=10)
    clean(ax)
plt.tight_layout()
save(fig, '03_coeficientes')


# ==========================================================================
# 4, precisiones alpha_j del ARD (sintetico)
# ==========================================================================
model_ard = modelos['ARD']
fig, ax = plt.subplots(figsize=(11, 5))
ax.bar(range(degree + 1), model_ard.lambda_, color=YELLOW,
       edgecolor=FG, linewidth=0.4)
ax.set_yscale('log')
ax.set_xlabel('Grado del polinomio')
ax.set_ylabel(r'$\alpha_j$ (escala log)')
ax.set_title('Precisiones del prior ARD por peso')
clean(ax)
save(fig, '04_precisiones_sintetico')


# ==========================================================================
# 5, posteriors marginales de los primeros 7 pesos
# ==========================================================================
m = model_ard.coef_
S = model_ard.sigma_
n_features = len(m)
threshold = getattr(model_ard, 'threshold_lambda', 1e4)
keep = model_ard.lambda_ < threshold
idx_keep = np.where(keep)[0]

def sigma_of(j):
    if j in idx_keep:
        j_in_S = int(np.where(idx_keep == j)[0][0])
        return float(np.sqrt(S[j_in_S, j_in_S]))
    return 1e-3

ww = np.linspace(-3, 3, 500)
n_show = 7
fig, ax = plt.subplots(figsize=(11, 5.5))
for j in range(n_show):
    color = YELLOW if j < 3 else CELESTE
    lw    = 2.6    if j < 3 else 1.3
    alpha_l = 1.0  if j < 3 else 0.7
    ax.plot(ww, st.norm.pdf(ww, m[j], sigma_of(j)),
            color=color, lw=lw, alpha=alpha_l, label=f'$w_{{{j}}}$')

ax.axvline(w0_true, color=FG, lw=0.6, alpha=0.4)
ax.axvline(w1_true, color=FG, lw=0.6, alpha=0.4)
ax.axvline(w2_true, color=FG, lw=0.6, alpha=0.4)
ax.plot([w0_true, w1_true, w2_true], [0, 0, 0], 'o',
        color=FG, markersize=8, label='Verdaderos')

ax.set_xlabel('$w_j$')
ax.set_ylabel('Densidad')
ax.set_title('Posteriors marginales (primeros 7 pesos)')
ax.legend(loc='upper right', ncol=2, fontsize=10)
clean(ax)
save(fig, '05_posteriors_marginales')


# ==========================================================================
# 6, posterior predictive con incertidumbre epistemica y aleatoria
# ==========================================================================
yy_mean, yy_measured_std = model_ard.predict(Phi_xx, return_std=True)
sigma_noise = np.sqrt(1.0 / model_ard.alpha_)
yy_std = np.sqrt(np.clip(yy_measured_std ** 2 - sigma_noise ** 2, 0, None))

fig, ax = plt.subplots(figsize=(11, 5.5))
ax.fill_between(xx, yy_mean - 2 * yy_measured_std, yy_mean - 2 * yy_std,
                color=CELESTE, alpha=0.3)
ax.fill_between(xx, yy_mean + 2 * yy_std, yy_mean + 2 * yy_measured_std,
                color=CELESTE, alpha=0.3, label='+ aleatoria')
ax.fill_between(xx, yy_mean - 2 * yy_std, yy_mean + 2 * yy_std,
                color=YELLOW, alpha=0.35, label='epistemica')
ax.plot(xx, yy_true, '--', color=CELESTE, lw=2, label='Verdadera')
ax.plot(xx, yy_mean, '-', color=YELLOW, lw=2.6, label='Media del posterior')
ax.plot(x, y, 'x', color=FG, markersize=10, mew=2, label='Observado')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Posterior predictive con incertidumbre')
ax.legend(loc='upper left', fontsize=10, ncol=2)
clean(ax)
save(fig, '06_posterior_predictive')


# ==========================================================================
# 7, muestras del posterior sobre los pesos
# ==========================================================================
m_active = m[keep]
w_post = st.multivariate_normal(mean=m_active,
                                 cov=S + 1e-6 * np.eye(S.shape[0]))

fig, ax = plt.subplots(figsize=(11, 5.5))
for _ in range(20):
    w_active_sample = w_post.rvs()
    w_sample = np.zeros(n_features)
    w_sample[keep] = w_active_sample
    ax.plot(xx, Phi_xx @ w_sample, color=YELLOW, lw=0.9, alpha=0.5)
ax.plot([], [], color=YELLOW, lw=2, label='Muestras del posterior')
ax.plot(xx, yy_true, '--', color=CELESTE, lw=2.5, label='Verdadera')
ax.plot(x, y, 'x', color=FG, markersize=10, mew=2, label='Observado')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Muestras funcionales del posterior')
ax.legend(loc='upper left', fontsize=10)
clean(ax)
save(fig, '07_posterior_samples')


# ==========================================================================
# Datos medicos, diabetes con features polinomicos
# ==========================================================================
data = load_diabetes()
poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
X_med_poly = poly.fit_transform(data.data)
poly_names = poly.get_feature_names_out(data.feature_names)

X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
    X_med_poly, data.target, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train_ms = scaler.fit_transform(X_train_m)
X_test_ms  = scaler.transform(X_test_m)

modelos_med = {
    'OLS':            LinearRegression(),
    'Ridge':          Ridge(alpha=1.0),
    'Bayesian Ridge': BayesianRidge(),
    'ARD':            ARDRegression(),
}
for m in modelos_med.values():
    m.fit(X_train_ms, y_train_m)
coefs_med = {name: m.coef_ for name, m in modelos_med.items()}


# ==========================================================================
# 8, sparsity ridge vs ard en diabetes
# ==========================================================================
n_features_med = len(poly_names)
n_ridge = int(np.sum(np.abs(coefs_med['Ridge']) > 1e-3))
n_ard   = int(np.sum(np.abs(coefs_med['ARD'])   > 1e-3))

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
axes[0].bar(range(n_features_med), coefs_med['Ridge'],
            color=CELESTE, edgecolor=FG, linewidth=0.2)
axes[0].set_title(f'Ridge, {n_ridge} activas de {n_features_med}')
axes[1].bar(range(n_features_med), coefs_med['ARD'],
            color=YELLOW, edgecolor=FG, linewidth=0.2)
axes[1].set_title(f'ARD, {n_ard} activas de {n_features_med}')
for a in axes:
    a.axhline(0, color=FG, lw=0.4, alpha=0.4)
    a.set_xlabel('Variable')
    clean(a)
axes[0].set_ylabel('Coeficiente')
plt.tight_layout()
save(fig, '08_sparsity_diabetes')


# ==========================================================================
# 9, precisiones alpha_j en diabetes
# ==========================================================================
ard_med = modelos_med['ARD']
fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(range(n_features_med), ard_med.lambda_,
       color=YELLOW, edgecolor=FG, linewidth=0.2)
ax.set_yscale('log')
ax.set_xlabel('Variable')
ax.set_ylabel(r'$\alpha_j$ (escala log)')
ax.set_title('Precisiones ARD en dataset de diabetes')
clean(ax)
save(fig, '09_precisiones_diabetes')


# ==========================================================================
# 10, parity plot diabetes
# ==========================================================================
y_pred_med, y_std_med = ard_med.predict(X_test_ms, return_std=True)

fig, ax = plt.subplots(figsize=(7, 7))
ax.errorbar(y_test_m, y_pred_med, yerr=1.96 * y_std_med,
            fmt='o', color=YELLOW, ecolor=CELESTE, elinewidth=0.9,
            markersize=6, alpha=0.9, mec=FG, mew=0.4,
            label='ARD, media e IC 95%')
lo = float(min(y_test_m.min(), y_pred_med.min()))
hi = float(max(y_test_m.max(), y_pred_med.max()))
ax.plot([lo, hi], [lo, hi], '--', color=FG, lw=1, alpha=0.6,
        label='prediccion perfecta')
ax.set_xlabel('Progresion real')
ax.set_ylabel('Progresion predicha')
ax.set_title('ARD en Diabetes, predicho vs real')
ax.legend(loc='upper left', fontsize=10)
ax.set_aspect('equal', adjustable='box')
clean(ax)
plt.tight_layout()
save(fig, '10_parity_diabetes')


# ==========================================================================
# 11, evolucion iterativa de lambda_j y w_j en el caso sintetico
# Muestra como durante el ajuste ARD sube lambda_j (precision del prior)
# de las variables irrelevantes, empujando su peso hacia cero, hasta
# converger o hasta cruzar el threshold_lambda que sklearn usa como cutoff.
# ==========================================================================
iters = [1, 2, 3, 5, 8, 15, 30, 60, 120, 250, 500]
lambdas_traj = []
coefs_traj = []
for max_it in iters:
    m_it = ARDRegression(fit_intercept=False, max_iter=max_it,
                         threshold_lambda=1e4)
    m_it.fit(Phi, y)
    lambdas_traj.append(m_it.lambda_.copy())
    coefs_traj.append(m_it.coef_.copy())
lambdas_traj = np.array(lambdas_traj)
coefs_traj = np.array(coefs_traj)

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# Panel izquierdo, trayectoria de lambda_j
axL = axes[0]
for j in range(degree + 1):
    if j < 3:
        axL.plot(iters, lambdas_traj[:, j], color=YELLOW, lw=2.6,
                 marker='o', markersize=6, mec=FG, mew=0.5, alpha=1.0,
                 label=f'$w_{{{j}}}$ verdadero' if j == 0 else None)
    else:
        axL.plot(iters, lambdas_traj[:, j], color=CELESTE, lw=1.1,
                 marker='.', markersize=5, alpha=0.55,
                 label='$w_j$ irrelevante' if j == 3 else None)
axL.axhline(1e4, color=FG, lw=1.2, ls='--', alpha=0.8,
            label='threshold_lambda (cutoff)')
axL.set_xscale('log')
axL.set_yscale('log')
axL.set_xlabel('Iteraciones del ajuste ARD')
axL.set_ylabel(r'$\lambda_j$ (log)')
axL.set_title('$\\lambda_j$ sube en las variables irrelevantes')
axL.legend(loc='center right', fontsize=10)
clean(axL)

# Panel derecho, trayectoria de w_j
axR = axes[1]
for j in range(degree + 1):
    if j < 3:
        axR.plot(iters, coefs_traj[:, j], color=YELLOW, lw=2.6,
                 marker='o', markersize=6, mec=FG, mew=0.5, alpha=1.0)
    else:
        axR.plot(iters, coefs_traj[:, j], color=CELESTE, lw=1.1,
                 marker='.', markersize=5, alpha=0.55)
axR.axhline(0, color=FG, lw=0.5, alpha=0.4)
axR.set_xscale('log')
axR.set_xlabel('Iteraciones del ajuste ARD')
axR.set_ylabel('$w_j$')
axR.set_title('los pesos irrelevantes convergen a cero')
clean(axR)

fig.suptitle(
    'Dinamica del ARD, cada iteracion sube $\\lambda_j$ de las variables '
    'irrelevantes hasta apagarlas',
    color=FG, fontsize=15, y=1.03
)
plt.tight_layout()
save(fig, '11_dinamica_ard')


# ==========================================================================
# 12, tabla grafica de la dinamica del ARD para la presentacion
# Muestra, en cada celda, el par (lambda_j, w_j) al cortar el ajuste en
# distintas iteraciones. Amarillo = peso verdadero, celeste = peso
# irrelevante. Se ve como lambda sube y w cae a cero para los irrelevantes.
# ==========================================================================
iter_sel = [1, 2, 3, 4, 5, 10, 30]
cols_sel = [0, 1, 2, 5, 10, 15]
relev = [True, True, True, False, False, False]

lambdas_sel = []
coefs_sel = []
for it in iter_sel:
    m_it = ARDRegression(fit_intercept=False, max_iter=it,
                         threshold_lambda=1e4)
    m_it.fit(Phi, y)
    lambdas_sel.append(m_it.lambda_[cols_sel].copy())
    coefs_sel.append(m_it.coef_[cols_sel].copy())
lambdas_sel = np.array(lambdas_sel)
coefs_sel = np.array(coefs_sel)


def fmt_l(v):
    if v < 10:
        return f'{v:.2f}'
    if v < 1e5:
        return f'{v:.0f}'
    return f'{v:.0e}'


def fmt_w(v):
    if abs(v) < 1e-3:
        return f'{v:+.0e}'
    return f'{v:+.3f}'


cell_text = []
cell_colors = []
for i in range(len(iter_sel)):
    row_text = []
    row_col = []
    for k in range(len(cols_sel)):
        lam = lambdas_sel[i, k]
        w   = coefs_sel[i, k]
        row_text.append(f'$\\lambda$ = {fmt_l(lam)}\n$w$ = {fmt_w(w)}')
        if lam >= 1e4:
            row_col.append(CELESTE + '40')
        elif relev[k]:
            row_col.append(YELLOW + '55')
        else:
            row_col.append(YELLOW + '20')
    cell_text.append(row_text)
    cell_colors.append(row_col)

col_labels = [f'$w_{{{j}}}$' for j in cols_sel]
row_labels = [f'iter {it}' for it in iter_sel]
col_colors = [YELLOW if r else CELESTE for r in relev]

fig, ax = plt.subplots(figsize=(14, 7))
ax.axis('off')

table = ax.table(
    cellText=cell_text,
    cellColours=cell_colors,
    rowLabels=row_labels,
    colLabels=col_labels,
    colColours=col_colors,
    loc='center',
    cellLoc='center',
)
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 3.2)

for (i, j), cell in table.get_celld().items():
    cell.set_edgecolor(FG)
    cell.set_linewidth(0.4)
    if i == 0:
        cell.get_text().set_color(BG)
        cell.get_text().set_fontweight('bold')
        cell.get_text().set_fontsize(13)
    elif j == -1:
        cell.get_text().set_color(FG)
        cell.get_text().set_fontweight('bold')
        cell.set_facecolor(BG)
        cell.set_edgecolor(FG)
    else:
        cell.get_text().set_color(FG)

fig.suptitle(
    'Dinamica del ARD, $\\lambda_j$ sube y $w_j$ tiende a cero',
    color=FG, fontsize=16, fontweight='bold', y=0.97,
)
save(fig, '12_tabla_dinamica')


print(f'Listo. {len(os.listdir(OUT))} graficos guardados en la carpeta {OUT}/')
