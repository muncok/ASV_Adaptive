from multiprocessing import Process, Manager
from tqdm import tqdm
import pickle
import os
import pandas as pd
import numpy as np
from sv_system import sv_system
from spk_model import spk_model
from utils import euc_dist_sim, key2df, cos_dist_batch, cos_dist_sim
from plot_ROC import plot_ROC
import argparse



keys = np.array(pickle.load(open("sv_keys.pkl", "rb")))
key_df = key2df(keys)
key2id = {k:v for v, k in enumerate(keys)}
embeds = np.load("sv_embeds.npy")


from sklearn.metrics import roc_curve

def compute_eer(pos_scores, neg_scores):
    score_vector = np.concatenate([pos_scores, neg_scores])
    label_vector = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    fpr, tpr, thres = roc_curve(label_vector, score_vector, pos_label=1)
    eer = np.min([fpr[np.nanargmin(np.abs(fpr - (1 - tpr)))],
                 1-tpr[np.nanargmin(np.abs(fpr - (1 - tpr)))]])
    thres = thres[np.nanargmin(np.abs(fpr - (1 - tpr)))]

    return eer

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
    correct = np.count_nonzero(np.array(accepts) == label)
    wrong = len(label) - correct

    ### Enroll Accuracy ###
    if n_total_enrolls == 0:
        enr_acc = 1
    else:
        enr_acc = np.count_nonzero(np.array(enrolls) == 1) / n_total_enrolls

    ### FPR and FNR
    fpr = np.count_nonzero((np.array(accepts) == 1) & (label == 0)) / np.count_nonzero(label == 0)
    fnr = np.count_nonzero((np.array(accepts) == 0) & (label == 1)) / np.count_nonzero(label == 1)

    return acc, enr_acc, fpr, fnr, enrolls, scores, correct, wrong

def evaluation_inc(enr_spk, enr_id, trials_id, label, config):
    spk_models = []
    enroll_utters = embeds[enr_id]
    spk_models.append(spk_model(enr_spk, keys, enroll_utters, config))

    system = sv_system(spk_models, config)
    accuracy, enr_accuracy, fpr, fnr, enrolls, scores, correct, wrong = get_accuracy_system(system, trials_id, label)
    scores = np.array(scores)
    pos_scores = scores[label==1]
    neg_scores = scores[label==0]
    eer = compute_eer(pos_scores, neg_scores)

    return [accuracy, eer, fpr, fnr, enrolls, enr_accuracy, pos_scores.tolist(), neg_scores.tolist(), correct, wrong]

def evaluation_base(enr_spk, enr_id, trials_id, label, config):
    enr_uttrs_embeds = embeds[[key2id[k] for k in enr_uttrs]]
    trial_uttrs_embeds = embeds[trials_id]
    scores = cos_dist_sim(enr_uttrs_embeds, trial_uttrs_embeds, dim=1)
    scores = scores.mean(0).astype(np.float32)
    accepts = scores > config['accept_thres']
    acc = np.count_nonzero(np.array(accepts) == label) / len(label)
    correct = np.count_nonzero(np.array(accepts) == label)
    wrong = len(label) - correct

    fpr = np.count_nonzero((np.array(accepts) == True) & (label == 0)) / np.count_nonzero(label == 0)
    fnr = np.count_nonzero((np.array(accepts) == False) & (label == 1)) / np.count_nonzero(label == 1)
    pos_scores = scores[label==1]
    neg_scores = scores[label==0]
    eer = compute_eer(pos_scores, neg_scores)

    return [acc, eer, fpr, fnr, pos_scores.tolist(), neg_scores.tolist(), correct, wrong]

def eval_wrapper(enr_spk, enr_id, trials_id, label, config, res_lst):
    n_trials = len(trials_id)
    if config['mod'] == 'base':
        base_stat = evaluation_base(enr_spk, enr_id, trials_id, label, config)
        b_acc, b_eer, b_fpr, b_fnr, pos_scores, neg_scores, correct, wrong = base_stat
        result = (enr_spk, enr_id[0], n_trials, b_acc, b_eer, b_fpr, b_fnr, n_trials)
    elif config['mod'] == 'inc':
        inc_stat = evaluation_inc(enr_spk, enr_id, trials_id, label, config)
        i_acc, i_eer, i_fpr, i_fnr, i_enrolls, i_enr_acc, pos_scores, neg_scores, correct, wrong = inc_stat
        result = (enr_spk, enr_id[0], i_acc, i_eer, i_fpr, i_fnr, i_enrolls, i_enr_acc, n_trials)
    else:
        raise NotImplemented

    res_lst.put(result)
    posScore_lst.put(pos_scores)
    negScore_lst.put(neg_scores)
    correct_lst.put(correct)
    wrong_lst.put(wrong)




if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-n_enr',
                        type=str,
                        help='number of enrollments',
                        default='full')

    parser.add_argument('-out_dir',
                        type=str,
                        help='output_dir base',
                        required=True
                        )

    parser.add_argument('-n_process',
                        type=int,
                        help='number of processes',
                        default=40)

    parser.add_argument('-sv_mode',
                        type=str,
                        help='type of sv_system',
                        choices=['base', 'inc'],
                        required=True
                        )

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
            'enroll_thres_update': False,
            # trial_tpye: sortedPos, random, posNeg
            'trial_type': 'random',
            # n_use_enroll: 'full' or 'N' (decimal in string type such as '5')
            'n_use_enroll': args.n_enr,
            'include_init': args.incl_init,
            'cfid_coef': 1.2,  #alpha 0.0005
            'mean_coef': 1,    #beta 0.01
            'c_multiplier': 1,
            'm_multiplier': 1,
            'ord': 2,
            # normal, extreme
            'thresh_type': args.thresh_type,
            'mod': args.sv_mode
            }
    print(config)

    # trial for finding best threshold
    trial_for_thresh = pd.read_pickle('enr306_uttr1/trial_for_thresh.pkl')
    if ('Cos' in config['sim']) or ('cos' in config['sim']):
        train_score_vector = cos_dist_batch(embeds[trial_for_thresh.enrolment_id],
                embeds[trial_for_thresh.test_id])
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

    for p_ratio in [0.01, 0.1, 0.5, 0.9]:
        print("="*100)
        print("p_ratio: {}".format(p_ratio))
        trial_set = pickle.load(open("enr306_uttr1/trials_ratio_{}.pkl".format(str(p_ratio)), "rb"))
        output_dir = "{}/n_enr_{}_pRatio_{}".format(args.out_dir, config['n_use_enroll'], p_ratio)
        if config["include_init"]:
            output_dir += "_initEnr"
        if config["accept_thres_update"]:
            output_dir += "_thresUpdt"
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        pickle.dump(config, open(output_dir+'/config.pkl', "wb"))

        n_parallel = args.n_process
        results = []
        posScores = []
        negScores = []
        total_correct = 0
        total_wrong = 0
        for i, idx in enumerate(range(0,len(trial_set), n_parallel)):
            procs = []
            manager = Manager()
            res_lst = manager.Queue()
            posScore_lst = manager.Queue()
            negScore_lst = manager.Queue()
            correct_lst = manager.Queue()
            wrong_lst = manager.Queue()

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

                proc = Process(target=eval_wrapper, args=(enr_spks, enr_id, trials_id, label, config, res_lst,))
                procs.append(proc)
                proc.start()

            print('Joining jobs [{}/{}]'.format(i, len(trial_set)//n_parallel))
            for p in tqdm(procs):
                results.append(res_lst.get())
                posScores += posScore_lst.get()
                negScores += negScore_lst.get()
                total_correct += correct_lst.get()
                total_wrong += wrong_lst.get()
                p.join()

        #pickle.dump(results, open("{}/result.pkl".format(output_dir), "wb"))
        pickle.dump(posScores, open("{}/posScores.pkl".format(output_dir), "wb"))
        pickle.dump(negScores, open("{}/negScores.pkl".format(output_dir), "wb"))
        pickle.dump([total_correct, total_wrong], open("{}/answers.pkl".format(output_dir), "wb"))

    print('Done')
