#!/bin/bash
# Batch GIF generation script for all trained models

# SET MUJOCO ENVIRONMENT VARIABLES
export MUJOCO_PATH=/home/scai/msr/aiy257584/anaconda3/envs/mujoco-env/lib/python3.10/site-packages/mujoco
export MUJOCO_PLUGIN_PATH=$MUJOCO_PATH
export LD_LIBRARY_PATH=$MUJOCO_PATH:$LD_LIBRARY_PATH
export MUJOCO_GL=egl

# CONFIGURATION
ENVS=("InvertedPendulum-v4" "Hopper-v4" "HalfCheetah-v4")
ALGOS=("a2c" "ppo")
BEST_SEED=456  # Use seed with best performance for GIF generation

# Create GIF output directory
mkdir -p gifs

echo "=========================================="
echo "Starting Batch GIF Generation"
echo "=========================================="
echo ""

# Generate individual algorithm GIFs
for env in "${ENVS[@]}"; do
    echo "Environment: $env"
    echo "----------------------------------------"
    
    for algo in "${ALGOS[@]}"; do
        model_dir="models/${algo}/${env}/seed_${BEST_SEED}"
        output_path="gifs/${algo}_${env}_seed${BEST_SEED}.gif"
        
        echo "Generating GIF: ${algo} on ${env}"
        
        python3 scripts/generate_gif.py \
            --env_name "$env" \
            --algo "$algo" \
            --model_dir "$model_dir" \
            --output_path "$output_path" \
            --n_episodes 3 \
            --fps 30 \
            --max_steps 1000
        
        if [ $? -eq 0 ]; then
            echo "✓ GIF created: $output_path"
        else
            echo "✗ Failed to create GIF for ${algo} on ${env}"
        fi
        echo ""
    done
    
    # Generate comparison GIF
    echo "Generating comparison GIF for $env"
    
    a2c_dir="models/a2c/${env}/seed_${BEST_SEED}"
    ppo_dir="models/ppo/${env}/seed_${BEST_SEED}"
    output_path="gifs/comparison_${env}_seed${BEST_SEED}.gif"
    
    python3 scripts/generate_gif.py \
        --env_name "$env" \
        --algo comparison \
        --a2c_model_dir "$a2c_dir" \
        --ppo_model_dir "$ppo_dir" \
        --output_path "$output_path" \
        --n_episodes 1 \
        --fps 30 \
        --max_steps 1000
    
    if [ $? -eq 0 ]; then
        echo "✓ Comparison GIF created: $output_path"
    else
        echo "✗ Failed to create comparison GIF for ${env}"
    fi
    
    echo ""
    echo "=========================================="
    echo ""
done

echo "All GIFs generated successfully!"
echo ""
echo "GIFs saved to: gifs/"
echo ""
ls -lh gifs/