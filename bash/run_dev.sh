#!/bin/sh
#SBATCH -o ./slurm/%j-train.out # STDOUT

python compute_params.py\
    -p "/home/gumougeot/all/codes/python/biom3d/data/nucleus/preds/raw_selection_8T_dry_cot A-B_A_BIOM/20230913-231641-nucleus_dry_fold0/20220119_cot_8T_dry_A5_432.tif"\
    -cc "/home/gumougeot/all/codes/python/biom3d/data/nucleus/preds/raw_selection_8T_dry_cot A-B_cc_A_BIOM/20230914-015946-chromo_dry_fold0/20220119_cot_8T_dry_A5_432.tif"\
    -s 0.1032 0.1032 0.2\
    -o "params/dev"\
    -cv 2\
    -v
