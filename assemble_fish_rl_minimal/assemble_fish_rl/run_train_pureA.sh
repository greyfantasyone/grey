#!/usr/bin/env bash
set -e
python train_recurrent_ppo_floating_grasp.py \
  --xml phy3.02_schemeA_grasp.xml \
  --algo recurrentppo \
  --assist-mode off \
  --timesteps 1500000 \
  --n-envs 4 \
  --save checkpoints/floating_grasp_pureA
