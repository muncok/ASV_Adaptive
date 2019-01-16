import numpy as np
from spk_model import spk_model
from sv_system import sv_system
from utils import cos_dist_sim
from utils import read_trials


def evaluation_base(config, embeds, enr_idx, trial_idxs, label):
    enr_uttrs_embeds = embeds[enr_idx]
    trial_uttrs_embeds = embeds[trial_idxs]
    scores = cos_dist_sim(enr_uttrs_embeds, trial_uttrs_embeds, dim=1)
    scores = scores.mean(0).astype(np.float32)
    preds = np.array(scores > config['accept_thres'])

    trial_trace = np.stack([trial_idxs, scores, label, preds], axis=0)

    return trial_trace

def adapt_sv_system(config, embeds, keys, enr_spk, enr_idx, trial_idxs, label):
    spk_models = []
    enroll_utters = embeds[enr_idx]
    enr_keys = [keys[id_] for id_ in enr_idx]
    spk_models.append(spk_model(config, enr_spk, enr_keys, enroll_utters))

    system = sv_system(spk_models, config)
    preds = []
    enroll_pred = []
    scores = []
    for idx in trial_idxs:
        # evaluate trial samples step by step
        key = keys[idx]
        if config['sv_mode'] == 'inc':
            accept, enroll, cfid = system.verify_adapt(key,
                    embeds[idx])
        elif config['sv_mode'] == 'inc_update':
            accept, enroll, cfid = system.verify_adapt(key,
                    embeds[idx])
        elif config['sv_mode'] == 'inc_update_neg':
            accept, enroll, cfid = system.verify_adapt_neg(key,
                    embeds[idx])
        else:
            print("Not available mode")
            raise NotImplementedError

        preds.append(accept)
        enroll_pred.append(enroll)
        scores.append(cfid)

    # stack trial results
    trial_trace = np.stack([trial_idxs, scores, label, preds, enroll_pred], axis=0)

    # system.show_enrolls()
    # system.print_benefits()

    return system, trial_trace

def test_sv_system(embeds, system, trial_idxs, label):
    preds = []
    max_scores = []
    all_scores = []
    for idx in trial_idxs:
        # evaluate trial samples step by step
        accept, max_score, scores =  system.verify(embeds[idx])
        preds.append(accept)
        max_scores.append(max_score)
        all_scores.append(scores)

    # stack trial results
    trial_trace = np.stack([trial_idxs, max_scores, label, preds], axis=0)

    return trial_trace, all_scores

from score_norm_utils import eval_s_score
import torch

def s_score_norm(sv_embeds, enr_idxs, adapt_tr, test_tr, threshold):
    init_enr_embeds = sv_embeds[enr_idxs]
    adapt_trial_idxs, _, _, adapt_pred, adapt_enroll_pred = adapt_tr
    adapt_embeds = sv_embeds[adapt_trial_idxs.astype(np.int64)]
    adapt_enr_embeds = adapt_embeds[adapt_enroll_pred == 1]
    total_enr_embeds = np.concatenate([init_enr_embeds, adapt_enr_embeds])

    test_trial_idxs, _, test_label, _ = test_tr
    test_embeds = sv_embeds[test_trial_idxs.astype(np.int64)]
    test_label = torch.from_numpy(test_label).byte()
    
    enr_embeds = total_enr_embeds
    imposter_embeds = adapt_embeds[adapt_pred==0]
    
    s_err = eval_s_score(
                          enr_embeds, test_embeds,
                          test_label, threshold,
                          imposter_embeds)
    
    return s_err

def base_s_score_norm(sv_embeds, enr_idxs, test_tr, ood_tr, threshold):
    enr_embeds = sv_embeds[enr_idxs]
    
    test_trial_idxs, _, test_label, _ = test_tr
    test_embeds = sv_embeds[test_trial_idxs.astype(np.int64)]
    test_label = torch.from_numpy(test_label).byte()
    
    ood_trial_idxs, _, _, _ = ood_tr
    imposter_embeds =  sv_embeds[np.array(ood_trial_idxs).astype(np.int64)]
    
    s_err = eval_s_score(
                          enr_embeds, test_embeds,
                          test_label, threshold,
                          imposter_embeds)
    
    return s_err

def eval_wrapper(config, embeds, keys, trial, metaInfo_l, trace_l):
    enr_spks, enr_idxs, adapt_trial, test_trial, ood_trial \
            = read_trials(config, keys, trial)

    n_trials = len(adapt_trial[0])
    if config['sv_mode'] == 'base':
        adapt_trace = evaluation_base(config, embeds, enr_idxs, *adapt_trial)
        test_trace = evaluation_base(config, embeds, enr_idxs, *test_trial)
        ood_trace = evaluation_base(config, embeds, enr_idxs, *ood_trial)
        test_s_err = base_s_score_norm(
                    embeds, enr_idxs, test_trace, 
                    ood_trace, config['accept_thres'])
        meta_info = {'enr_spks':enr_spks, 'enr_idxs':enr_idxs, 'n_trials':n_trials,
                    'test_s_err':test_s_err, 'ood_s_err':1}
    elif 'inc' in config['sv_mode']:
        system, adapt_trace = adapt_sv_system(config, embeds, keys,
                               enr_spks, enr_idxs, *adapt_trial)
        test_trace, test_scores = test_sv_system(embeds, system, *test_trial)
        test_s_err = s_score_norm(
                    embeds, enr_idxs, adapt_trace, 
                    test_trace, config['accept_thres'])
        ood_trace, ood_scores = test_sv_system(embeds, system, *ood_trial)
        ood_s_err = s_score_norm(
                    embeds, enr_idxs, adapt_trace, 
                    ood_trace, config['accept_thres'])
        meta_info = {'enr_spks':enr_spks, 'enr_idxs':enr_idxs, 'n_trials':n_trials, 
                     'test_s_err':test_s_err, 'ood_s_err':ood_s_err} 
#                      'test_scores':test_scores,
#                      'ood_scores':ood_scores}
    else:
        raise NotImplemented

    metaInfo_l.put(meta_info)
    trace_l.put((adapt_trace, test_trace, ood_trace))

