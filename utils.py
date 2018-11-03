import numpy as np
import pandas as pd
#import torch
#from torch.nn.functional import cosine_similarity
from sklearn.metrics import roc_curve, auc

def euc_dist(a, b, dim):
    return np.linalg.norm(a-b, axis=dim)

def euc_dist_sim(a, b, dim=0):
    return 1/(1+euc_dist(a, b, dim))

def cos_dist_sim(a, b, dim):
    # N x M,  M x K --> NxK scores
    a = a / np.linalg.norm(a, axis=dim, keepdims=True)
    b = b / np.linalg.norm(b, axis=dim, keepdims=True)
    return np.dot(a, b.T)

def cos_dist_batch(a, b):
    # N x M,  N x M --> N scores
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return (a * b).sum(1)

def cos_dist_sim_torch(a, b, dim=0):
    a = torch.from_numpy(a).float()
    b = torch.from_numpy(b).float()

    return cosine_similarity(a, b, dim=dim).numpy()

def key2df(keys, delimeter="-"):
	key_df = pd.DataFrame(keys, columns=['key'])
	key_df['spk'] = key_df.key.apply(lambda x: x.split(delimeter)[0])
	key_df['label'] = key_df.groupby('spk').ngroup()
	key_df['origin'] = key_df.spk.apply(lambda x: 'voxc2' if x.startswith('id') else 'voxc1')
	key_df = key_df.set_index('key')

	return key_df

def compute_eer(pos_scores, neg_scores):
    score_vector = np.concatenate([pos_scores, neg_scores])
    label_vector = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    fpr, tpr, thres = roc_curve(label_vector, score_vector, pos_label=1)
    eer = np.min([fpr[np.nanargmin(np.abs(fpr - (1 - tpr)))],
                 1-tpr[np.nanargmin(np.abs(fpr - (1 - tpr)))]])
    thres = thres[np.nanargmin(np.abs(fpr - (1 - tpr)))]

    return eer, thres
