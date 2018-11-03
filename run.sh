#! /bin/sh
echo "n_enr, out_dir"

n_enr=$1
out_dir=$2

python run_trial.py -n_enr $n_enr -out_dir "$out_dir/base" -sv_mode base -n_process 80

for i in 1 2 3 4 5
do
    python run_trial.py -n_enr $n_enr -out_dir "$out_dir/inc/$1" -sv_mode inc -n_process 80
    python run_trial.py -n_enr $n_enr -out_dir "$out_dir/inc/$1" -sv_mode inc -n_process 80 -incl_init
    python run_trial.py -n_enr $n_enr -out_dir "$out_dir/inc/$1" -sv_mode inc -n_process 80 -update
    python run_trial.py -n_enr $n_enr -out_dir "$out_dir/inc/$1" -sv_mode inc -n_process 80 -incl_init -update
done

