#!/usr/bin/env python
# coding: utf-8


import pathlib


# In[3]:


PATH = pathlib.Path("./output/figures/featsel_threshold")
import os

os.makedirs(PATH, exist_ok=True)

# In[4]:


import numpy as np

from sklearn.preprocessing import scale

import seaborn as sns
import matplotlib.pyplot as plt


from arfs_gen import genClassificationData


def data(informative=5, redundant=10, d=17, n=300):
    STATE = np.random.RandomState(seed=1231241)

    X, y = genClassificationData(
        n_features=d,
        n_redundant=redundant,
        n_strel=informative,
        n_samples=n,
        random_state=STATE,
    )
    X = scale(X)
    print()
    return X, y


# In[21]:


import sys

sys.path.append("../runner/")

import squamish

STATE = np.random.RandomState(seed=1231241)
model = squamish.Main(random_state=STATE, n_resampling=50, fpr=1e-4)

X, y = data()

model.fit(X, y)
shadw_bounds = model.stat_.shadow_stat

imps = model.rfmodel.importances()

relevant = imps > shadw_bounds[1]


sns.barplot(x=np.arange(len(imps)), y=imps, hue=relevant)


plt.axhline(shadw_bounds[1])
plt.title("Feature selection with statistical based threshold")
plt.savefig(PATH / "selection_with_statmethod.pdf")

