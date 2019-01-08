import numpy as np
from spk_model import spk_model
from sv_system import sv_system
from utils import cos_dist_sim


def evaluation_base(config, embeds, enr_ids, trials_ids, label):
    enr_uttrs_embeds = embeds[enr_ids]
    trial_uttrs_embeds = embeds[trials_ids]
    scores = cos_dist_sim(enr_uttrs_embeds, trial_uttrs_embeds, dim=1)
    scores = scores.mean(0).astype(np.float32)
    pred = np.array(scores > config['accept_thres']).astype(np.int32)
    trial_trace = np.stack([pred, label, scores], axis=0)

    return trial_trace

def run_inc_sv_system(config, system, embeds, keys, trial_ids, label):
    pred = []
    enroll_pred = []
    scores = []
    for trial_id in trial_ids:
        # evaluate trial samples step by step
        if config['sv_mode'] == 'inc':
            accept, enroll, cfid = system.verify_and_enroll(keys[trial_id],
                    embeds[trial_id])
        elif config['sv_mode'] == 'inc_update':
            accept, enroll, cfid = system.verify_and_enroll(keys[trial_id],
                    embeds[trial_id])
        elif config['sv_mode'] == 'inc_neg':
            accept, enroll, cfid = system.verify_and_enroll_neg(keys[trial_id],
                    embeds[trial_id])

        pred.append(accept)
        enroll_pred.append(enroll)
        scores.append(cfid)

    trial_trace = np.stack([pred, label, scores, enroll_pred], axis=0)

    return trial_trace

def evaluation_inc(config, embeds, keys, enr_spk, enr_ids, trial_ids, label):
    spk_models = []
    enroll_utters = embeds[enr_ids]
    enr_keys = [keys[id_] for id_ in enr_ids]
    spk_models.append(spk_model(config, enr_spk, enr_keys, enroll_utters))

    system = sv_system(spk_models, config)
    trial_trace = run_inc_sv_system(config, system, embeds, keys, trial_ids, label)

    # system.show_enrolls()
    # system.print_benefits()

    return trial_trace

def eval_wrapper(config, embeds, keys, enr_spk, enr_ids, trials_ids, label,
        metaInfo_l, trace_l):
    n_trials = len(trials_ids)
    if config['sv_mode'] == 'base':
        trace = evaluation_base(config, embeds, enr_ids, trials_ids, label)
        result = [enr_spk, enr_ids, n_trials]
    elif 'inc' in config['sv_mode']:
        trace = evaluation_inc(config, embeds, keys,
                               enr_spk, enr_ids, trials_ids, label)
        result = [enr_spk, enr_ids, n_trials]
    else:
        raise NotImplemented

    metaInfo_l.put(result)
    trace_l.put(trace)
