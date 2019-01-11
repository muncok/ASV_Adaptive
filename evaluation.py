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
        elif config['sv_mode'] == 'inc_neg':
            accept, enroll, cfid = system.verify_adapt_neg(key,
                    embeds[idx])

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
    scores = []
    for idx in trial_idxs:
        # evaluate trial samples step by step
        accept, cfid =  system.verify(embeds[idx])

        preds.append(accept)
        scores.append(cfid)

    # stack trial results
    trial_trace = np.stack([trial_idxs, scores, label, preds], axis=0)

    return trial_trace

def eval_wrapper(config, embeds, keys, trial, metaInfo_l, trace_l):
    enr_spks, enr_idxs, adapt_trial, test_trial, ood_trial \
            = read_trials(config, keys, trial)

    n_trials = len(adapt_trial[0])
    if config['sv_mode'] == 'base':
        adapt_trace = evaluation_base(config, embeds, enr_idxs, *adapt_trial)
        test_trace = evaluation_base(config, embeds, enr_idxs, *test_trial)
        ood_trace = evaluation_base(config, embeds, enr_idxs, *ood_trial)
    elif 'inc' in config['sv_mode']:
        system, adapt_trace = adapt_sv_system(config, embeds, keys,
                               enr_spks, enr_idxs, *adapt_trial)
        test_trace = test_sv_system(embeds, system, *test_trial)
        ood_trace = test_sv_system(embeds, system, *ood_trial)
    else:
        raise NotImplemented

    meta_info = {'enr_spks':enr_spks, 'enr_idxs':enr_idxs, 'n_trials':n_trials}
    metaInfo_l.put(meta_info)
    trace_l.put((adapt_trace, test_trace, ood_trace))
