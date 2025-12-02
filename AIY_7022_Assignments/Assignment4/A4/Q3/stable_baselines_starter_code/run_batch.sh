#!/bin/bash
# Batch training + evaluation script for A2C and PPO on MuJoCo environments

# SET MUJOCO ENVIRONMENT VARIABLES
export MUJOCO_PATH=/home/scai/msr/aiy257584/anaconda3/envs/mujoco-env/lib/python3.10/site-packages/mujoco
export MUJOCO_PLUGIN_PATH=$MUJOCO_PATH
export LD_LIBRARY_PATH=$MUJOCO_PATH:$LD_LIBRARY_PATH
export MUJOCO_GL=egl

# EXPERIMENT CONFIG
ENVS=("InvertedPendulum-v4" "Hopper-v4" "HalfCheetah-v4")
SEEDS=(42 123 456)
ALGOS=("a2c" "ppo")

echo "Starting Batch Training and Evaluation for RL Experiments"
echo ""

# TRAIN FUNCTION
train_config() {
    local env=$1
    local algo=$2
    local seed=$3

    model_dir="models/${algo}/${env}/seed_${seed}"

    echo ""
    echo "TRAINING: $algo on $env (seed: $seed)"
    echo "Saving model to: $model_dir"

    python3 scripts/train_sb.py \
        --env_name "$env" \
        --algo "$algo" \
        --seed "$seed" \
        --mode train \
        --save_dir "$model_dir"

    if [ $? -eq 0 ]; then
        echo "Training SUCCESS: $algo | $env | seed $seed"
    else
        echo "Training FAILED: $algo | $env | seed $seed"
    fi

    echo ""
}

# EVALUATE FUNCTION
evaluate_config() {
    local env=$1
    local algo=$2
    local seed=$3

    model_dir="models/${algo}/${env}/seed_${seed}"

    echo "EVALUATING: $algo on $env (seed: $seed)"
    echo "Loading model from: $model_dir"

    python3 scripts/train_sb.py \
        --env_name "$env" \
        --algo "$algo" \
        --seed "$seed" \
        --mode evaluate \
        --model_dir "$model_dir"

    if [ $? -eq 0 ]; then
        echo "Evaluation SUCCESS: $algo | $env | seed $seed"
    else
        echo "Evaluation FAILED: $algo | $env | seed $seed"
    fi

    echo ""
}

# MAIN LOOP
for env in "${ENVS[@]}"; do
    echo "Environment: $env"

    for seed in "${SEEDS[@]}"; do

       for algo in "${ALGOS[@]}"; do
            echo ""
            echo " STARTING ${algo^^} RUNS "
            echo ""

            # TRAIN
            train_config "$env" "$algo" "$seed"

            # EVALUATE
            evaluate_config "$env" "$algo" "$seed"

        done
    done
done

echo ""
echo "ALL TRAINING + EVALUATION COMPLETED!"
echo ""
echo "View TensorBoard logs with:"
echo "  tensorboard --logdir logs/"
echo ""
echo "Models saved under:"
echo "  models/<algo>/<env>/seed_<seed>/model.zip"
echo ""