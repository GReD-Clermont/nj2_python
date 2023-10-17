#!/bin/sh
#SBATCH -o ./slurm/%j-train.out # STDOUT

python compute_params.py\
    -p "/home/gumougeot/all/codes/python/biom3d/data/nucleus/preds/raw_selection_24h_cot_A_BIOM/20230914-164235-nucleus_aline_all_fold0"\
    -cc "/home/gumougeot/all/codes/python/biom3d/data/nucleus/preds/raw_selection_24h_cot_cc_A_BIOM/20230915-004004-chromo_aline_all_fold0"\
    -s 0.1032 0.1032 0.2\
    -o "params/raw_selection_24h_cot_params_v2"

python compute_params.py\
    -p "/home/gumougeot/all/codes/python/biom3d/data/nucleus/preds/raw_selection_48h_cot A-C-D-E_A_BIOM/20230914-164235-nucleus_aline_all_fold0"\
    -cc "/home/gumougeot/all/codes/python/biom3d/data/nucleus/preds/raw_selection_48h_cot A-C-D-E_cc_A_BIOM/20230915-004004-chromo_aline_all_fold0"\
    -s 0.1032 0.1032 0.2\
    -o "params/raw_selection_48h_cot A-C-D-E_params_v2"

python compute_params.py\
    -p "/home/gumougeot/all/codes/python/biom3d/data/nucleus/preds/raw_selection_48h24hL_cot_A_BIOM/20230914-164235-nucleus_aline_all_fold0"\
    -cc "/home/gumougeot/all/codes/python/biom3d/data/nucleus/preds/raw_selection_48h24hL_cot_cc_A_BIOM/20230915-004004-chromo_aline_all_fold0"\
    -s 0.1032 0.1032 0.2\
    -o "params/raw_selection_48h24hL_cot_params_v2"

python compute_params.py\
    -p "/home/gumougeot/all/codes/python/biom3d/data/nucleus/preds/raw_selection_48h48hL_cot_A_BIOM/20230914-164235-nucleus_aline_all_fold0"\
    -cc "/home/gumougeot/all/codes/python/biom3d/data/nucleus/preds/raw_selection_48h48hL_cot_cc_A_BIOM/20230915-004004-chromo_aline_all_fold0"\
    -s 0.1032 0.1032 0.2\
    -o "params/raw_selection_48h48hL_cot_params_v2"

python compute_params.py\
    -p "/home/gumougeot/all/codes/python/biom3d/data/nucleus/preds/raw_selection_48h72hL_cot A-B-C_A_BIOM/20230914-164235-nucleus_aline_all_fold0"\
    -cc "/home/gumougeot/all/codes/python/biom3d/data/nucleus/preds/raw_selection_48h72hL_cot A-B-C_cc_A_BIOM/20230915-004004-chromo_aline_all_fold0"\
    -s 0.1032 0.1032 0.2\
    -o "params/raw_selection_48h72hL_cot A-B-C_params_v2"

# python compute_params.py\
#     -p "/home/gumougeot/all/codes/python/biom3d/data/nucleus/preds/raw_selection_24h_cot_A_BIOM/20230914-164235-nucleus_aline_all_fold0"\
#     -cc "/home/gumougeot/all/codes/python/biom3d/data/nucleus/preds/raw_selection_24h_cot_cc_A_BIOM/20230915-004004-chromo_aline_all_fold0"\
#     -s 0.1032 0.1032 0.2\
#     -o "params/raw_selection_8T_dry_cot A-B_params_v2"

# python compute_params.py\
#     -p "/home/gumougeot/all/codes/python/biom3d/data/nucleus/preds/raw_selection_48h_cot A-C-D-E_A_BIOM/20230914-164235-nucleus_aline_all_fold0"\
#     -cc "/home/gumougeot/all/codes/python/biom3d/data/nucleus/preds/raw_selection_48h_cot A-C-D-E_cc_A_BIOM/20230915-004004-chromo_aline_all_fold0"\
#     -s 0.1032 0.1032 0.2\
#     -o "params/raw_selection_Col_dry_cot A-B-C_params_v2"

# python compute_params.py\
#     -p "/home/gumougeot/all/codes/python/biom3d/data/nucleus/preds/raw_selection_48h24hL_cot_A_BIOM/20230914-164235-nucleus_aline_all_fold0"\
#     -cc "/home/gumougeot/all/codes/python/biom3d/data/nucleus/preds/raw_selection_48h24hL_cot_cc_A_BIOM/20230915-004004-chromo_aline_all_fold0"\
#     -s 0.1032 0.1032 0.2\
#     -o "params/raw_selection_dry_cot A-C-D-E_params_v2"

# python compute_params.py\
#     -p "/home/gumougeot/all/codes/python/biom3d/data/nucleus/preds/raw_selection_48h48hL_cot_A_BIOM/20230914-164235-nucleus_aline_all_fold0"\
#     -cc "/home/gumougeot/all/codes/python/biom3d/data/nucleus/preds/raw_selection_48h48hL_cot_cc_A_BIOM/20230915-004004-chromo_aline_all_fold0"\
#     -s 0.1032 0.1032 0.2\
#     -o "params/raw_selection_hira8T_dry_cot A-B_params_v2"

# python compute_params.py\
#     -p "/home/gumougeot/all/codes/python/biom3d/data/nucleus/preds/raw_selection_48h72hL_cot A-B-C_A_BIOM/20230914-164235-nucleus_aline_all_fold0"\
#     -cc "/home/gumougeot/all/codes/python/biom3d/data/nucleus/preds/raw_selection_48h72hL_cot A-B-C_cc_A_BIOM/20230915-004004-chromo_aline_all_fold0"\
#     -s 0.1032 0.1032 0.2\
#     -o "params/raw_selection_hira_dry_cot A-B-C_params_v2"
