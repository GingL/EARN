#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python ./tools/eval.py \
--dataset refcoco \
--splitBy unc \
--split val \
--id exp0
