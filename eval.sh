#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python ./tools/eval_refclef.py \
--dataset refclef \
--splitBy unc \
--split val \
--id exp_adp

CUDA_VISIBLE_DEVICES=1 python ./tools/eval_refclef.py \
--dataset refclef \
--splitBy unc \
--split val \
--id exp_wo_adp
