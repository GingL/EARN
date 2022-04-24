#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python ./tools/train.py \
--dataset refcoco \
--splitBy unc \
--exp_id exp_time \
--checkpoint_path output \
--max_iters 50000 \
--att_res_weight 1.0 \
--lang_res_weight 1.0 \
--vis_res_weight 0.01 \
--loss_combined 5.0 \
--loss_divided 1.0 \
--sub_filter_thr 0.6