#!/usr/bin/env python
# coding: utf-8
import pathlib

from sklearn.utils import check_random_state



import dill
import numpy as np
import pandas as pd

from sklearn.metrics import precision_score, roc_auc_score, recall_score, f1_score

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

path = "./runner/results/"
path = pathlib.Path(path)

import sys
sys.path.append("./runner/")

import experiment_pipeline
import paper_output

toy_set_params = experiment_pipeline.toy_set_params
    

def get_truth(params):
    strong=params["strong"]
    weak=params["weak"]
    irrel=params["irr"]
    truth = [True] * (strong + weak) + [False] * irrel
    return truth

def get_truthAR(params):
    strong=params["strong"]
    weak=params["weak"]
    irrel=params["irr"]
    truth = [2] * strong + [1]*weak + [0] * irrel
    return truth



# In[97]:



toy_set_params


# In[98]:


filepath = path / "paper.dat"
results = []
toy = {}

with open(filepath,"rb") as file:
    content = dill.load(file=file)

    for set_and_model, results in content.items():
        if len(results) > 0:
            toy.setdefault(set_and_model, []).append(results)


# In[99]:


ARModels = ["FRI", "SQ"] 

only_ARModels = {k:v for (k,v) in toy.items() if k[1] in ARModels}



# # Precision and Recall per Relevance Type
# questions:
# how sensitive is each method on different types of relevance class?

# In[101]:


# We only take truth if relevance type (1 or 2) 
def get_truth_onetype(params, reltype):
    truth = get_truthAR(params)
    truth = np.array(truth)
    if reltype==0:
        raise Exception("0 makes no sense")
    return truth == reltype



def rowfunc(row, reltype, scorefunc):
    ix = row.name
    data = ix[0]
    method = ix[1]

    features = row[0]

    truth = get_truth_onetype(toy_set_params[data], reltype)
    pred = features == reltype
    score  = scorefunc(truth, pred,zero_division=0)
    
    if score == 0:
        score = np.nan
    return score


def convert_frame(series, name):
    return series.to_frame().rename(columns={0:name}).groupby(level=[0,1]).mean()


# In[102]:


unpacked_lists = pd.DataFrame(only_ARModels).T.explode(0)
feature_frame = unpacked_lists.applymap(lambda x: x["features"])


# In[103]:


weakly_precision = convert_frame(feature_frame.apply(rowfunc, axis=1,args=[1,precision_score]), "precision")
weakly_recall = convert_frame(feature_frame.apply(rowfunc, axis=1,args=[1,recall_score]), "recall")

weakly = pd.concat([weakly_precision,weakly_recall],axis=1).unstack(level=0).T


# In[104]:


weakly


# In[105]:


weakly.loc["precision"].mean()


# In[106]:


weakly.groupby(level=0).mean()


# # Strongly

# In[107]:


strongly_precision = convert_frame(feature_frame.apply(rowfunc, axis=1,args=[2,precision_score]), "precision")
strongly_recall = convert_frame(feature_frame.apply(rowfunc, axis=1,args=[2,recall_score]), "recall")

strongly = pd.concat([strongly_precision,strongly_recall],axis=1).unstack(level=0).T


# In[108]:


strongly


# In[109]:


strongly.groupby(level=0).mean()


# In[110]:


combined = pd.concat([weakly,strongly],axis=1,keys=["Weakly","Strongly"]).round(decimals=2)


# In[111]:


combined


# In[116]:


combined.replace(np.nan,"-")


# In[114]:


def print_df_astable(df, filename=None):
    output = df.to_latex(multicolumn=False, bold_rows=True)
    if filename is not None:
        with open("./output/tables/{}.tex".format(filename), "w") as f:
            f.write(output)
    return output
# In[115]:


print_df_astable(combined, "prec_rec_ARFS")


# In[117]:


combined_mean = combined.groupby(level=0).mean()


# In[118]:


print_df_astable(combined_mean,"prec_rec_ARFS_mean")


# In[ ]:




