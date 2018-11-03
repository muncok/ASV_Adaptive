#! /bin/sh
echo "n_enr, out_dir"

n_enr=$1
out_dir=$2

python run_trial.py -n_enr $n_enr -out_dir "$out_dir/base" -sv_mode base -n_process 40
python run_trial.py -n_enr $n_enr -out_dir "$out_dir/inc" -sv_mode inc -n_process 40
python run_trial.py -n_enr $n_enr -out_dir "$out_dir/inc" -sv_mode inc -n_process 40 -incl_init
python run_trial.py -n_enr $n_enr -out_dir "$out_dir/inc" -sv_mode inc -n_process 40 -update
python run_trial.py -n_enr $n_enr -out_dir "$out_dir/inc" -sv_mode inc -n_process 40 -incl_init -update


