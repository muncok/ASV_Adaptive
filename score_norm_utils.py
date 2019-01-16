import numpy as np
import pandas as pd
import torch
from torch.nn.functional import cosine_similarity

def key2df(keys, delimeter="-"):
    key_df = pd.DataFrame(keys, columns=['key'])
    key_df['spk'] = key_df.key.apply(lambda x: x.split(delimeter)[0])
    key_df['label'] = key_df.groupby('spk').ngroup()
    key_df['origin'] = key_df.spk.apply(lambda x: 'voxc2' if x.startswith('id') else 'voxc1')
    key_df = key_df.set_index('key')
    
    return key_df

def uttrkey_to_id(key2id, uttr_keys):
    return [key2id[key] for key in uttr_keys]

def batch_cosine_score(x, y):
    if not isinstance(x, torch.Tensor):
        x = torch.from_numpy(x)
    if not isinstance(y, torch.Tensor):
        y = torch.from_numpy(y)
    score = cosine_similarity(x.unsqueeze(1), y.unsqueeze(0), dim=2)
    return score

from sklearn.metrics import roc_curve

def compute_eer(pos_scores, neg_scores):                                                 
    score_vector = np.concatenate([pos_scores, neg_scores])                              
    label_vector = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))]) 
    fpr, tpr, thres = roc_curve(label_vector, score_vector, pos_label=1)                 
    eer = np.min([fpr[np.nanargmin(np.abs(fpr - (1 - tpr)))],                            
              1-tpr[np.nanargmin(np.abs(fpr - (1 - tpr)))]])                          
    thres = thres[np.nanargmin(np.abs(fpr - (1 - tpr)))]                                 

    return eer, thres                                                                    

def compute_z_scores(imposter_scores, scores):
    # scores: (n_enr_embeds, n_test_embeds), 2D Tensor
    z_mu = imposter_scores.mean(dim=1, keepdim=True)
    z_std = imposter_scores.std(dim=1, keepdim=True)
    
    return (scores - z_mu) / z_std

def compute_z_threshold(imposter_scores, threshold):
    # scores: (n_enr_embeds, n_test_embeds), 2D Tensor
    z_mu = imposter_scores.mean(dim=1, keepdim=True)
    z_std = imposter_scores.std(dim=1, keepdim=True)
    
    return (threshold - z_mu.mean()) / z_std.mean()

def eval_cosine_score(enr_embeds, test_embeds, test_label, threshold):
    """
     Returns:
         score: score for each enr_embeds
    """
    score = batch_cosine_score(enr_embeds, test_embeds)
    pred = score.mean(0) > threshold
    pred_acc = pred.eq(test_label).sum().item() / len(test_label)
    eer, _ = compute_eer(score.mean(0)[test_label==1], score.mean(0)[test_label==0])
    #print(f"pred_acc: {pred_acc}")
    #print(f"eer: {eer}")
    return score, pred_acc, eer

def eval_z_score(enr_embeds, test_embeds, test_label, threshold, imposter_embeds):
    score, _, _ = eval_cosine_score(enr_embeds, test_embeds, test_label, threshold)
    imp_score = batch_cosine_score(enr_embeds, imposter_embeds)
    z_threshold = compute_z_threshold(imp_score, threshold)
    z_score = compute_z_scores(imp_score, score)
    z_pred = z_score.mean(0) > z_threshold
    z_pred_acc = z_pred.eq(test_label).sum().item() / len(test_label)
    z_eer, _ = compute_eer(z_score.mean(0)[test_label==1], z_score.mean(0)[test_label==0])
    #print(f"z_pred_acc: {z_pred_acc}")
    #print(f"z_eer: {z_eer}")
    return z_score, z_pred_acc, z_eer
    
def eval_s_score(enr_embeds, test_embeds, test_label, threshold, imposter_embeds):
    target_imp_score = batch_cosine_score(enr_embeds, imposter_embeds)
    test_imp_score = batch_cosine_score(test_embeds, imposter_embeds)
    target_std = target_imp_score.std(dim=1)
    test_std = test_imp_score.std(dim=1)

    imp_embeds_mean = imposter_embeds.mean(axis=0, keepdims=True)
    target_norm_embeds = (enr_embeds - imp_embeds_mean)
    test_norm_embeds = (test_embeds - imp_embeds_mean)
    norm_score = np.matmul(target_norm_embeds, test_norm_embeds.T)
    scale_factor = target_std.unsqueeze(1) * test_std

    s_score = norm_score / scale_factor
    s_threshold = threshold / scale_factor.mean().item()

    s_pred = s_score.mean(0) > s_threshold
    s_pred_err = 1 - (s_pred==test_label).sum().item() / len(test_label)
#     s_eer, _ = compute_eer(s_score.mean(0)[test_label==1], s_score.mean(0)[test_label==0])
    #print(f"s_pred_acc: {s_pred_acc}")
    #print(f"s_eer: {s_eer}")
    return s_pred_err