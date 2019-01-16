import os
import sys
import itertools

trial = "trials/enr306/enr306_closedset"
out_dir = "enr306_closedset_out"
FSs = [3, 5, 7, 9]
n_enrs = [1, 2, 3]
modes = "base inc inc_update inc_update_neg"

for family_size, n_enr in itertools.product(FSs, n_enrs):
    cmd = "python run_trial_closedset.py -trial_in {trial}/FS_{f_size}/ -out_dir {out_dir}/ -sv_mode {mode} -n_enr {n_enr}"\
            .format(f_size=family_size, trial=trial, mode=modes,
                    out_dir=out_dir, n_enr=n_enr)
    print(cmd)
    try:
        os.system(cmd)
    except:
        sys.exit()
