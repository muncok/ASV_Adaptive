from multiprocessing import Process, Manager
from tqdm import tqdm
import pickle
import os
import pandas as pd
import numpy as np
from sv_system import sv_system
from spk_model import spk_model
from utils import cos_dist_sim_torch, euc_dist_sim, key2df, cos_dist_sim
from plot_ROC import plot_ROC
import argparse



keys = np.array(pickle.load(open("../voxc2_fbank64_embeds/sv_keys.pkl", "rb")))
key_df = key2df(keys)
key2id = {k:v for v, k in enumerate(keys)}
embeds = np.load("../voxc2_fbank64_precise_untied_embeds/sv_embeds.npy")

### Return: Accuracy, Enroll_accuracy, FPR, FNR, Scores
def get_accuracy_system(sv_system, trial_ids, label):
    accepts = []
    enrolls = []
    scores = []
    for trial_id in trial_ids:
        accept, enroll, cfid = sv_system.verify_and_enroll(keys[trial_id], embeds[trial_id])
        accepts.append(accept)
        enrolls.append(enroll)
        scores.append(cfid)
    n_total_enrolls = np.count_nonzero(np.array(enrolls) != -1)

    ### Accuracy ###
    acc = np.count_nonzero(np.array(accepts) == label) / len(label)

    ### Enroll Accuracy ###
    if n_total_enrolls == 0:
        enr_acc = 1
    else:
        enr_acc = np.count_nonzero(np.array(enrolls) == 1) / n_total_enrolls

    ### FPR and FNR
    fpr = np.count_nonzero((np.array(accepts) == 1) & (label == 0)) / np.count_nonzero(label == 0)
    fnr = np.count_nonzero((np.array(accepts) == 0) & (label == 1)) / np.count_nonzero(label == 1)

    return acc, enr_acc, fpr, fnr, enrolls, scores

def evaluation(enr_spk, enr_id, trials_id, label, config):
    spk_models = []
    enroll_utters = embeds[enr_id]
    spk_models.append(spk_model(enr_spk, keys, enroll_utters, config))

    system = sv_system(spk_models, config)
    accuracy, enr_accuracy, fpr, fnr, enrolls, scores = get_accuracy_system(system, trials_id, label)
    pos_ratio = np.count_nonzero(label == 1) / np.count_nonzero(label == 0)
    # eer, _ = compute_eer(scores[label==1], scores[label==0])

    return [accuracy, enr_accuracy, fpr, fnr, enrolls, scores, pos_ratio]

def evaluation_base(enr_spk, enr_id, trials_id, label, config):
    enr_uttrs_embeds = embeds[[key2id[k] for k in enr_uttrs]]
    trial_uttrs_embeds = embeds[trials_id]
    scores = cos_dist_sim(enr_uttrs_embeds, trial_uttrs_embeds, dim=1)
    mean_scores = scores.mean(0)
    accepts = mean_scores > config['accept_thres']
    acc = np.count_nonzero(np.array(accepts) == label) / len(label)
    fpr = np.count_nonzero((np.array(accepts) == True) & (label == 0)) / np.count_nonzero(label == 0)
    fnr = np.count_nonzero((np.array(accepts) == False) & (label == 1)) / np.count_nonzero(label == 1)
    pos_ratio = np.count_nonzero(label == 1) / np.count_nonzero(label == 0)
    # eer, _ = compute_eer(scores[label==1], scores[label==0])

    return [acc, fpr, fnr, scores, pos_ratio]

def eval_wrapper(enr_spk, enr_id, trials_id, label, config, res_lst):
    config['mod'] = 'base'
    base_stat = evaluation_base(enr_spk, enr_id, trials_id, label, config)
    config['mod'] = 'inc'
    inc_stat = evaluation(enr_spk, enr_id, trials_id, label, config)

    b_acc, b_fpr, b_fnr, b_scores, b_pos_ratio = base_stat
    i_acc, i_enr_acc, i_fpr, i_fnr, i_enrolls, i_scores, i_pos_ratio = inc_stat

    result = (enr_spk, len(enr_id), label,
            b_acc, b_fpr, b_fnr,  b_scores, b_pos_ratio,
            i_acc, i_enr_acc, i_fpr, i_fnr, i_enrolls, i_scores, i_pos_ratio)

    res_lst.put(result)

if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-n_enr',
                        type=str,
                        help='number of enrollments',
                        default='full')

    parser.add_argument('-n_process',
                        type=int,
                        help='number of processes',
                        default=40)

    parser.add_argument('-thresh_type',
                        type=str,
                        help='type of threshold',
                        choices=['normal', 'extreme'],
                        default='normal'
                        )

    parser.add_argument('-update',
                        help='use of thresh update',
                        action='store_true')

    parser.add_argument('-incl_init',
                        help='include the init enrollment',
                        action='store_true')

    args = parser.parse_args()

    config = {
            # sim: cosMean, meanCos, euc
            'sim': 'meanCos',
            'accept_thres_update': args.update,
            'enroll_thres_update': args.update,
            # trial_tpye: sortedPos, random, posNeg
            'trial_type': 'random',
            # n_use_enroll: 'full' or 'N' (decimal in string type such as '5')
            'n_use_enroll': args.n_enr,
            'include_init': args.incl_init,
            'cfid_coef': 0.0005,
            'mean_coef': 0.01,
            'c_multiplier': 1,
            'm_multiplier': 1,
            'ord': 2,
            # normal, extreme
            'thresh_type': args.thresh_type
            }
    print(config)

    # trial for finding best threshold
    trial_for_thresh = pd.read_pickle('../cases/enr306_uttr1/trial_for_thresh.pkl')
    if ('Cos' in config['sim']) or ('cos' in config['sim']):
        train_score_vector = cos_dist_sim_torch(embeds[trial_for_thresh.enrolment_id],
                embeds[trial_for_thresh.test_id], dim=1)
    elif 'euc' in config['sim']:
        train_score_vector =  euc_dist_sim(embeds[trial_for_thresh.enrolment_id],
                embeds[trial_for_thresh.test_id], dim=1)

    train_label_vector = trial_for_thresh.label.tolist()
    accept_thres, fpr_, thres_ = plot_ROC(train_label_vector, train_score_vector)

    if config["thresh_type"] == "normal":
        config['accept_thres'] = accept_thres
        config['enroll_thres'] = thres_[np.where(fpr_ < 0.001)[0][-1]]
    elif config["thresh_type"] == "extreme":
        config['accept_thres'] = thres_[np.where(fpr_ > 0.2)[0][0]]
        config['enroll_thres'] = thres_[np.where(fpr_ < 0.01)[0][-1]]
    print('Accept Thres: {:.5f}, Enroll Thres: {:.5f}'.format(config['accept_thres'], config['enroll_thres']))


    enroll_id_sets = []
    trials_id_sets = []
    trials_label_sets = []

    # neg_trials = pickle.load(open('../cases/enr10_pos300/test/neg_trials.pkl', 'rb'))
    # neg_trials = [list(s) for s in neg_trials]
    # combinations = pickle.load(open('../cases/enr10_pos300/test/combination.pkl', 'rb'))
    # combinations = sorted(combinations, key=lambda x: len(x[1]))
    trial_set = pickle.load(open("../cases/enr306_uttr1/12240_trials.pkl", "rb"))

    output_dir = "results/n_enr_{}".format(config['n_use_enroll'])
    if config["include_init"]:
        output_dir += "_initEnr"
    if config["accept_thres_update"]:
        output_dir += "_thresUpdt"
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    pickle.dump(config, open(output_dir+'/config.pkl', "wb"))

    n_parallel = args.n_process
    for i, idx in enumerate(range(0,len(trial_set), n_parallel)):
        results = []
        procs = []
        manager = Manager()
        res_lst = manager.Queue()

        print('Starting jobs [{}/{}]'.format(i, len(trial_set)//n_parallel))
        for j, trial in enumerate(trial_set[idx:idx+n_parallel]):
            enr_spks, enr_uttrs, pos_trial, neg_trial = trial
            n_trials = len(pos_trial) + len(neg_trial)
            permu_idx = np.random.permutation(range(n_trials))

            ### Make trials with pre-defined(sorted pos trial) indices ###
            enr_id = np.array([key2id[k] for k in enr_uttrs])
            trials_id = np.array([key2id[k] for k in pos_trial + neg_trial])
            label = np.array([1]*len(pos_trial) + [0]*len(neg_trial))
            trials_id = trials_id[permu_idx]
            label = label[permu_idx]

            # record traces
            # enroll_id_sets += [enr_id]
            # trials_id_sets += [trials_id]
            # trials_label_sets += [label]

            proc = Process(target=eval_wrapper, args=(enr_spks, enr_id, trials_id, label, config, res_lst,))
            procs.append(proc)
            proc.start()

        print('Joining jobs [{}/{}]'.format(i, len(trial_set)//n_parallel))
        for p in tqdm(procs):
            results.append(res_lst.get())
            p.join()


        result_df = pd.DataFrame(results,
                                columns=['spk', 'n_enr_uttr', 'label',
                                        'base_acc', 'base_fpr', 'base_fnr', 'base_score', 'base_pos_ratio',
                                        'inc_acc', 'inc_enr', 'inc_fpr', 'inc_fnr', 'inc_enroll', 'inc_score',
                                        'inc_pos_ratio']
                                )
        result_df.to_pickle("{}/result{}.pkl".format(output_dir, i))

    print('Done')
