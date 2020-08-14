#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from core.dataloader import TimeSeries
from core.model import Hierarchy
from transformer import senticnet


stock = '0005.HK'
cols = ['Date', 'pos', 'neu', 'neg', 'ple', 'att', 'sen',
        'apt', 'Volume', 'ma10', 'ptc_chg', 'rsi12', 'macd', 'mfi']
data = TimeSeries('data/%s.csv' % stock, usecols=cols)

# %%
'''
taus = np.arange(1, 8)
n_class = np.arange(2, 10)

score = []
for tau in taus:
    score_ = []
    for n in n_class:
        X = data.weekly(senticnet,tau)[0]
        X = StandardScaler().fit_transform(X)
        clster = Hierarchy(n, method='ward', criterion='maxclust').fit(X)
        score_.append(clster.silhouette_score_)
    score.append(score_)
score = np.array(score).T
fig = plt.figure()
ax = Axes3D(fig)
grid = np.meshgrid(taus, n_class)
ax.plot_surface(grid[0], grid[1], score, rstride=1, cstride=1, cmap='rainbow')
ax.set_xlabel(r'Data period $\tau$')
ax.set_ylabel('Style number $n$')
ax.set_zlabel('Silhouette score')
plt.show()
'''
# %%
def class_change(X, y, center, radius, rank=1):
    change = abs(np.sign(np.append(0, np.diff(y,rank)))) # change[0] === 0
    dist = []
    for i in range(len(change)):
        if change[i] == 0:
            dist.append(0)
        else:
            precenter = center[y[i-rank]]
            preradius = radius[y[i-rank]]
            dist.append(abs((np.linalg.norm(X[i] - precenter) - preradius) / preradius))
    return dist

# %% [markdown]
# 风格的变换有没有周期性：风格自身的周期研究，离散ACD

# %%
tau = 4
n_class = np.arange(2,6)

X = data.weekly(senticnet,tau)[0]
X = StandardScaler().fit_transform(X)
fig, ax = plt.subplots(2, 2, figsize=(9, 7))
for i, n in enumerate(n_class):
    cluster = Hierarchy(n, method='ward', criterion='maxclust').fit(X)
    dist = class_change(X, cluster.predict(), cluster.center, cluster.radius)
    # dist = [abs(np.linalg.norm(X[i+1] - X[i])) for i in range(len(X)-1)]
    ax[i//2, i%2].bar(np.arange(len(dist)), dist)
    ax[i//2, i%2].set_xlabel('Periods', fontsize=10)
    ax[i//2, i%2].set_ylabel('Distance between styles', fontsize=10)
    ax[i//2, i%2].set_title('$n = %d$' % n, fontsize=10)
    ax[i//2, i%2].set_ylim(0, 1.9)
plt.tight_layout()
plt.show()

# %%
taus = np.arange(1,5)
n = 3

fig, ax = plt.subplots(2, 2, figsize=(9, 7))
for i, tau in enumerate(taus):
    X = data.weekly(senticnet,tau)[0]
    X = StandardScaler().fit_transform(X)
    cluster = Hierarchy(n, method='ward', criterion='maxclust').fit(X)
    dist = class_change(X, cluster.predict(), cluster.center, cluster.radius)
    # dist = [abs(np.linalg.norm(X[i+1] - X[i])) for i in range(len(X)-1)]
    ax[i//2, i%2].bar(np.arange(len(dist)), dist)
    ax[i//2, i%2].set_xlabel('Periods', fontsize=10)
    ax[i//2, i%2].set_ylabel('Distance between styles', fontsize=10)
    ax[i//2, i%2].set_title(r'$\tau = %d$' % tau, fontsize=10)
    ax[i//2, i%2].set_ylim(0, 0.9)
plt.tight_layout()
plt.show()

# %%
tau, n = 4, 3
threshold = 0.5

X = data.weekly(senticnet,tau)[0]
X = StandardScaler().fit_transform(X)

cluster = Hierarchy(n, method='ward', criterion='maxclust').fit(X)
center = cluster.center
fig, ax = plt.subplots(2, 1, figsize=(8, 4))
sns.heatmap(center,-4, 4, annot=True, square=True, cmap='RdYlGn', fmt='.2f', xticklabels=cols[1:], ax=ax[0])
ax[0].tick_params(labelsize=10)
ax[0].set_ylim(3, 0)
# take the sign of each center.
for i in range(center.shape[0]):
    for j in range(center.shape[1]):
        center[i,j] = np.sign(center[i,j]) if abs(center[i,j]) > threshold else 0
sns.heatmap(center,-1, 1, annot=True, square=True, cmap='RdYlGn', fmt='.0f', xticklabels=cols[1:], ax=ax[1])
ax[1].tick_params(labelsize=10)
ax[1].set_ylim(3, 0)
plt.tight_layout()
plt.show()

