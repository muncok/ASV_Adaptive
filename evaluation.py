import numpy as np
from spk_model import spk_model
from sv_system import sv_system
from utils import cos_dist_sim


def evaluation_base(config, embeds, enr_idx, trial_idxs, label):
    enr_uttrs_embeds = embeds[enr_idx]
    trial_uttrs_embeds = embeds[trial_idxs]
    scores = cos_dist_sim(enr_uttrs_embeds, trial_uttrs_embeds, dim=1)
    scores = scores.mean(0).astype(np.float32)
    pred = np.array(scores > config['accept_thres']).astype(np.int32)
    trial_trace = np.stack([trial_idxs, scores, label, pred], axis=0)

    return trial_trace

def run_inc_sv_system(config, system, embeds, keys, trial_idxs, label):
    pred = []
    enroll_pred = []
    scores = []
    for idx in trial_idxs:
        # evaluate trial samples step by step
        key = keys[idx]
        if config['sv_mode'] == 'inc':
            accept, enroll, cfid = system.verify_and_enroll(key,
                    embeds[idx])
        elif config['sv_mode'] == 'inc_update':
            accept, enroll, cfid = system.verify_and_enroll(key,
                    embeds[idx])
        elif config['sv_mode'] == 'inc_neg':
            accept, enroll, cfid = system.verify_and_enroll_neg(key,
                    embeds[idx])

        pred.append(accept)
        enroll_pred.append(enroll)
        scores.append(cfid)

    # stack trial results
    trial_trace = np.stack([trial_idxs, scores, label, pred, enroll_pred], axis=0)

    return trial_trace

def evaluation_inc(config, embeds, keys, enr_spk, enr_idx, trial_idxs, label):
    spk_models = []
    enroll_utters = embeds[enr_idx]
    enr_keys = [keys[id_] for id_ in enr_idx]
    spk_models.append(spk_model(config, enr_spk, enr_keys, enroll_utters))

    system = sv_system(spk_models, config)
    trial_trace = run_inc_sv_system(config, system, embeds, keys, trial_idxs, label)

    # system.show_enrolls()
    # system.print_benefits()

    return trial_trace

def eval_wrapper(config, embeds, keys, enr_spk, enr_idx, trials_ids, label,
        metaInfo_l, trace_l):
    n_trials = len(trials_ids)
    if config['sv_mode'] == 'base':
        trial_trace = evaluation_base(config, embeds, enr_idx, trials_ids, label)
    elif 'inc' in config['sv_mode']:
        trial_trace = evaluation_inc(config, embeds, keys,
                               enr_spk, enr_idx, trials_ids, label)
    else:
        raise NotImplemented

    meta_info = {'enr_spk':enr_spk, 'enr_idx':enr_idx, 'n_trials':n_trials}

    metaInfo_l.put(meta_info)
    trace_l.put(trial_trace)
