#%%
%load_ext autoreload
%autoreload 2
import sys
import lightgbm
import numpy as np
from sklearn.preprocessing import scale

import arfs_gen
sys.path.append("./runner/")

import experiment_pipeline
import fsmodel
import squamish
from squamish.models import RF,MyBoruta
from boruta import BorutaPy
# %% [python]


X,y = arfs_gen.genClassificationData(linear=False,n_samples=1000,n_features=80,
                                        n_strel=10,n_redundant=10)
X = scale(X)

lm = fsmodel.LM()
lm.fit(X,y)
lm.score(X,y)

# %%
s = squamish.Main()
s.fit(X,y)
#%%
print(s.borutamodel)
print(s.support_)
print(s.borutamodel.estimator.support_)
#%%
BEST_PARAMS = {
        #"num_leaves": 32,
        "max_depth": 10,
        "boosting_type": "rf",
        "bagging_fraction": 0.632,
        "bagging_freq": 1,
        "feature_fraction": 0.9,  # We force low feature fraction to reduce overshadowing of better redundant features
        "subsample": None,
        "subsample_freq": None,
        "verbose": -1,
        "colsample_bytree": None,
        "importance_type": "gain",
}
rf = RF("classification",**BEST_PARAMS)
rf.fit(X,y)
assert rf.score(X,y) == rf.estimator.score(X,y)
rf.score(X,y)
#%%
state = np.random.RandomState(123)
lgbm = lightgbm.LGBMClassifier(randomstate=state.randint(1e6),n_jobs=-1)
lgbm.fit(X,y)
lgbm.score(X,y)
#%%

bor = BorutaPy(lgbm)
bor.fit(X,y)
print("Support with LGBM GBT")
print(bor.support_)
#%%
rf = RF("classification",n_jobs=6)
bor = BorutaPy(rf.estimator,perc=50,n_estimators='auto',alpha=0.05)
bor.fit(X,y)
print("Support with LGBM RF")
print(bor.support_)

print(bor.support_weak_)
print(bor.ranking_)
# %%
X,y = arfs_gen.genClassificationData(linear=False,n_samples=1000,n_features=80,
                                        n_strel=10,n_redundant=10)
X = scale(X)
#%%
mb = MyBoruta("classification",perc=70,feature_fraction=0.2,max_depth=5)
mb.fit(X,y)
# %%
print(mb)
pred = mb.fset()[:20]
noise= mb.fset()[20:]
print(pred)
print(noise)
from sklearn.metrics import accuracy_score

print("recall", accuracy_score(np.ones(20),pred))
print("fprate", accuracy_score(np.ones(60),noise))

# # %%
# from sklearn.ensemble import RandomForestClassifier
# rf = RandomForestClassifier(n_jobs=-1)
# bor = BorutaPy(rf,perc=50,n_estimators='auto',alpha=0.05)
# bor.fit(X,y)

# # %%
# pred = bor.support_[:20]
# noise= bor.support_[20:]
# print(pred)
# print(noise)
# from sklearn.metrics import accuracy_score

# print("recall", accuracy_score(np.ones(20),pred))
# print("fprate", accuracy_score(np.ones(60),noise))

# # %%


# %%
