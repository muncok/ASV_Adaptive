import os
import sys
import pickle
import pprint

import pandas as pd
import numpy as np
from multiprocessing import Process, Manager
from tqdm import tqdm

from evaluation import eval_wrapper
from parser import get_args


if __name__=='__main__':

# ===================== Set Args =====================
    print(">>>>>Set Configurations<<<<<")
    args = get_args()


    config = {
                'sim': 'meanCos', # cosMean, meanCos, euc
                'accept_thres_update': False,
                'enroll_thres_update': False,
                'neg_thres_update': False,
                'include_init': False,
                'trial_sort': args.trial_sort, # sortedPos, random
                'queue_size': args.q_size,
                'n_enrs': args.n_enrs,
                'alpha': 0.0005,  # alpha 0.0005
                'beta': 0.01,  # beta 0.01
             }
    config['accept_thres'] = 0.56792
    config['enroll_thres'] = 0.68013
    config['neg_thres'] = 0.45

    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(config)

    # load key, embed, trials
    keys = np.array(pickle.load(open("xvector_embeds/sv_keys.pkl", "rb")))
    embeds = np.load("xvector_embeds/sv_embeds.npy")
    trial_set = pickle.load(open(args.trial_in+"/trials.pkl","rb"))

# ===================== Run trials =====================
    print(">>>>>Run Trials<<<<<")
    for mode in args.sv_mode:
        config['sv_mode'] = mode
        # set threshold update setting
        if "update" in mode:
            config['accept_thres_update'] = True
            config['enroll_thres_update'] = True
            config['include_init'] = True

        print(">>> {} mode  with {}".format(mode, args.trial_in))
        # ===== Output Dir =====
        trial_info = pickle.load(open(args.trial_in+"/trial_info.pkl","rb"))
        upper_n_enrs = trial_info['n_enrs']
        assert config['n_enrs'] <= upper_n_enrs
        output_dir = "{}/FS{}_ENR{}_QUE{}/{}/".format(
                args.out_dir, trial_info['set_size'], config['n_enrs'],
                config['queue_size'], mode)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        with open(output_dir+"config.txt", "w") as file:
            file.write(pprint.pformat(config))

        metaInfo_l = []
        trace_l = []
        n_parallel = args.n_process
        for i, idx in tqdm(enumerate(range(0, len(trial_set), n_parallel)),
                           ascii=True, desc="Jobs", file=sys.stdout,
                           total= len(trial_set)//n_parallel):
            procs = []
            manager = Manager()
            metaInfo_q = manager.Queue()
            trace_q = manager.Queue()

            for j, trial in enumerate(trial_set[idx:idx+n_parallel]):
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
