#!/usr/bin/env bash
set -e
python play_recurrent_ppo_floating_grasp.py \
  --model checkpoints/floating_grasp_pureA.zip \
  --deterministic
