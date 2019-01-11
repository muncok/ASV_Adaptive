import numpy as np
import pandas as pd
import torch
from torch.nn.functional import cosine_similarity
from sklearn.metrics import roc_curve

def euc_dist(a, b, dim):

    return np.linalg.norm(a-b, axis=dim)

def euc_dist_sim(a, b, dim):

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

def cos_dist_sim_torch(a, b, dim):
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

def find_best_threshold(y_train_true, y_train_prob):
    fpr_train, tpr_train, thresholds_train = roc_curve(y_train_true,
            y_train_prob, pos_label =True)
    sum_sensitivity_specificity_train = tpr_train + (1-fpr_train)
    best_threshold_id_train = np.argmax(sum_sensitivity_specificity_train)
    best_threshold = thresholds_train[best_threshold_id_train]

    return best_threshold, fpr_train, thresholds_train

def set_threshold(config, embeds, val_trial_pth):
    trial_for_thresh = pd.read_pickle(val_trial_pth)
    if ('Cos' in config['sim']) or ('cos' in config['sim']):
        train_score_vector = cos_dist_batch(embeds[trial_for_thresh.enrolment_id],
                embeds[trial_for_thresh.test_id])
    elif 'euc' in config['sim']:
        train_score_vector =  euc_dist_sim(embeds[trial_for_thresh.enrolment_id],
                embeds[trial_for_thresh.test_id], dim=1)

    train_label_vector = trial_for_thresh.label.tolist()
    accept_thres, fpr_, thres_ = find_best_threshold(
            train_label_vector, train_score_vector)

    if config["thresh_type"] == "normal":
        config['accept_thres'] = accept_thres
        config['enroll_thres'] = thres_[np.where(fpr_ < 0.001)[0][-1]]
    elif config["thresh_type"] == "extreme":
        config['accept_thres'] = thres_[np.where(fpr_ > 0.2)[0][0]]
        config['enroll_thres'] = thres_[np.where(fpr_ < 0.01)[0][-1]]

def sort_trials(config, key2id, trial):
    enr_spks, enr_uttr_keys, pos_trial_keys, neg_trial_keys = trial
    n_trials = len(pos_trial_keys) + len(neg_trial_keys)
    enr_ids = np.array([key2id[k] for k in enr_uttr_keys])
    trial_idxs = np.array([key2id[k]
        for k in pos_trial_keys + neg_trial_keys])
    label = np.array([1]*len(pos_trial_keys) + [0]*len(neg_trial_keys))

    # sorting trials
    if config['trial_sort'] == 'random':
        permu_idx = np.random.permutation(range(n_trials))
        trial_idxs = trial_idxs[permu_idx]
        label = label[permu_idx]
    elif config['trial_sort'] == 'sortedPos':
        sessions = list(map(lambda x: x[8:19], pos_trial_keys))
        df = pd.DataFrame.from_dict(dict( utters = pos_trial_keys,
            session = sessions ))
        unique_session = np.unique(sorted(df.session.values))
        session_cnt = df.session.value_counts()

        n_unique_sess = len(unique_session)
        n_sess_trials = len(neg_trial_keys)+n_unique_sess

        pos_sess_idx_ = sorted(np.random.choice(range(n_sess_trials),
            size=n_unique_sess, replace=False))

        pos_seat_idx_ = []
        for i, sess in enumerate(unique_session):
            l = session_cnt[sess]
            pos_sess_idx_[i+1:] += l-1
            for j in range(l):
                pos_seat_idx_.append(j+pos_sess_idx_[i])

        neg_seat_idx_ = list(set(range(n_trials)) - set(pos_seat_idx_))

        pos_trial_id = [key2id[k] for k in sorted(pos_trial_keys)]
        neg_trial_id = [key2id[k] for k in neg_trial_keys]
        trial_idxs = np.zeros(n_trials)
        trial_idxs[pos_seat_idx_] = pos_trial_id
        trial_idxs[neg_seat_idx_] = neg_trial_id
        trial_idxs = trial_idxs.astype(np.int64)

        label = np.zeros(n_trials)
        label[pos_seat_idx_] = [1]*len(pos_trial_keys)
        label[neg_seat_idx_] = [0]*len(neg_trial_keys)

    return enr_spks, enr_ids, trial_idxs, label

def interpret_trace(trace_record):
    pass

