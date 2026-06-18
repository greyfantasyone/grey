#!/usr/bin/env bash
set -e
python train_recurrent_ppo_floating_grasp.py \
  --xml phy3.02_schemeA_grasp.xml \
  --algo recurrentppo \
  --assist-mode decay \
  --assist-max-strength 0.35 \
  --assist-decay-end 0.35 \
  --timesteps 1200000 \
  --n-envs 4 \
  --save checkpoints/floating_grasp_decay_assist
