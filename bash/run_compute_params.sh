#!/bin/sh
#SBATCH -o ./slurm/%j-train.out # STDOUT

python compute_params.py\
    -p "data/preds/raw_selection_48h_cot A-C-D-E_A_BIOM"\
    -cc "data/preds/raw_selection_48h_cot A-C-D-E_cc_A_BIOM"\
    -s 0.1032 0.1032 0.2\
    -o "params/raw_selection_48h_cot A-C-D-E_params.csv"

python compute_params.py\
    -p "data/preds/raw_selection_48h24hL_cot_A_BIOM"\
    -cc "data/preds/raw_selection_48h24hL_cot_cc_A_BIOM"\
    -s 0.1032 0.1032 0.2\
    -o "params/raw_selection_48h24hL_cot_params.csv"

python compute_params.py\
    -p "data/preds/raw_selection_48h48hL_cot_A_BIOM"\
    -cc "data/preds/raw_selection_48h48hL_cot_cc_A_BIOM"\
    -s 0.1032 0.1032 0.2\
    -o "params/raw_selection_48h48hL_cot_params.csv"

python compute_params.py\
    -p "data/preds/selection_8T_cot_A_B_A_BIOM"\
    -cc "data/preds/selection_8T_cot_A_B_cc_A_BIOM"\
    -s 0.1032 0.1032 0.2\
    -o "params/selection_8T_cot_A_B_params.csv"

python compute_params.py\
    -p "data/preds/selection_col_cot_A_B_A_BIOM"\
    -cc "data/preds/selection_col_cot_A_B_cc_A_BIOM"\
    -s 0.1032 0.1032 0.2\
    -o "params/selection_col_cot_A_B_params.csv"

python compute_params.py\
    -p "data/preds/raw_selection_48h72hL_cot A-B-C_A_BIOM"\
    -cc "data/preds/raw_selection_48h72hL_cot A-B-C_cc_A_BIOM"\
    -s 0.1032 0.1032 0.2\
    -o "params/raw_selection_48h72hL_cot A-B-C_params.csv"