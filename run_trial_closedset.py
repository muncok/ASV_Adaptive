import os
import pickle
import pandas as pd
import numpy as np
import sys
import pprint

from multiprocessing import Process, Manager
from tqdm import tqdm

from utils import key2df, read_trials
from evaluation import eval_wrapper
from parser import get_args


if __name__=='__main__':

# ===================== Set Args =====================
    args = get_args()

    if args.sv_mode == 'inc_update':
        args.ths_update = True
        args.incl_init = True

    config = {
                'sim': 'meanCos', # cosMean, meanCos, euc
                'accept_thres_update': args.ths_update,
                'enroll_thres_update': False,
                'neg_thres_update': False,
                'trial_sort': args.trial_sort, # sortedPos, random
                'n_use_enroll': args.n_enr,
                'include_init': args.incl_init,
                'alpha': 0.0005,  # alpha 0.0005
                'beta': 0.01,  # beta 0.01
                'sv_mode': args.sv_mode
             }
    config['accept_thres'] = 0.56792
    config['enroll_thres'] = 0.68013
    config['neg_thres'] = 0.45

    print(">>>>>Set Configurations<<<<<")
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(config)

    # load key, embed
    keys = np.array(pickle.load(open("xvector_embeds/sv_keys.pkl", "rb")))
    embeds = np.load("xvector_embeds/sv_embeds.npy")
    key_df = key2df(keys)


# ===================== Run trials =====================
    print(">>>>>Run Trials<<<<<")
    n_parallel = args.n_process
    trial_set = pickle.load(open(args.trial_in,"rb"))

    # ===== Output Dir =====
    output_dir = "{}/{}/n_enr_{}/".format(
            args.out_dir, args.sv_mode, config['n_use_enroll'])
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    pickle.dump(config, open(output_dir+'/config.pkl', "wb"))

    metaInfo_l = []
    trace_l = []
    for i, idx in tqdm(enumerate(range(0, len(trial_set), n_parallel)),
                       ascii=True, desc="Jobs", file=sys.stdout,
                       total= len(trial_set)//n_parallel):
        procs = []
        manager = Manager()
        metaInfo_q = manager.Queue()
        trace_q = manager.Queue()

        for j, trial in enumerate(trial_set[idx:idx+n_parallel]):
            # enr_spks, enr_idxs, trial_idxs, label = read_trials(config, keys, trial)
            proc = Process(target=eval_wrapper,
                    args=(config, embeds, keys,
                        trial, metaInfo_q, trace_q))
            procs.append(proc)
            proc.start()

        for p in procs:
            metaInfo_l.append(metaInfo_q.get())
            trace_l.append(trace_q.get())
            p.join()

    meta_df = pd.DataFrame(metaInfo_l)
    meta_df.to_pickle("{}/meta_info_df.pkl".format(output_dir))
    pickle.dump(trace_l, open("{}/trace.pkl".format(output_dir), "wb"))

print('Done')
