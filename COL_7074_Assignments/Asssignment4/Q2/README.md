# Transformer Maze Solver - Complete Guide

## File Structure

Make sure your project folder looks like this:

```
your_project/
├── train.csv                    # Your training data
├── test.csv                     # Your test data
├── transformer_model.py         # File 1: Model architecture
├── dataset_handler.py           # File 2: Dataset & vocabulary
├── train_transformer.py         # File 3: Training script
├── eval.py                      # File 4: Evaluation script
└── README.md                    # This file
```

## Quick Start Guide

### Step 1: Verify Your Data

First, make sure your CSV files have these exact column names:
- `input_sequence`
- `output_path`
- `maze_type`

Run the test script:
```bash
python test_data_loading.py
```

If all tests pass, you're good to go! ✓

### Step 2: Train the Model

Basic training (with default hyperparameters):
```bash
python train_transformer.py --train_csv train.csv --test_csv test.csv
```

With custom parameters:
```bash
python train_transformer.py \
    --train_csv train.csv \
    --test_csv test.csv \
    --batch_size 32 \
    --epochs 20 \
    --learning_rate 0.0001 \
    --d_model 128 \
    --nhead 8 \
    --num_layers 6
```

Training will create a folder like `runs/transformer_20241123_120000/` containing:
- `best_model.pth` - Your trained model
- `vocabulary.json` - Token mappings
- `training_curves.png` - Graphs of training progress
- `results.json` - Final metrics

### Step 3: Evaluate the Model

Create a text file `test_input.txt` with a maze input (as a Python list):
```
['<ADJLIST_START>', '(1,0)', '<-->', '(0,0)', ';', ..., '<PATH_START>']
```

Then run:
```bash
python eval.py runs/transformer_20241123_120000/best_model.pth test_input.txt
```

## 📊 Model Hyperparameters

As specified in your assignment:

| Parameter | Value | Description |
|-----------|-------|-------------|
| D_MODEL | 128 | Embedding dimension |
| NHEAD | 8 | Number of attention heads |
| NUM_LAYERS | 6 | Transformer layers |
| DIM_FEEDFORWARD | 512 | Feedforward dimension |
| DROPOUT | 0.1 | Dropout rate |
| BATCH_SIZE | 32 | Batch size |
| EPOCHS | 20 | Training epochs |
| LEARNING_RATE | 1e-4 | Learning rate |

## 📈 What to Expect

### During Training
You'll see output like:
```
Epoch 1/20
----------------------------------------------------------
  Batch 50/100, Loss: 2.3456, Acc: 0.4521
  Batch 100/100, Loss: 2.1234, Acc: 0.5123

Epoch 1 Results:
  Train - Loss: 2.2345, Token Acc: 0.4822
  Val   - Loss: 2.1567, Token Acc: 0.5034, Seq Acc: 0.1200
  ✓ Saved best model (Val Seq Acc: 0.1200)
```

### After Training
The script will:
1. Save training curves (loss and accuracy plots)
2. Evaluate on test set
3. Save all results to JSON

### Expected Performance
- **Token Accuracy**: Should reach 70-90% by epoch 20
- **Sequence Accuracy**: Should reach 30-70% depending on maze complexity
- **Forked mazes** are harder than forkless mazes

## 🐛 Troubleshooting

### Problem: "FileNotFoundError: train.csv"
**Solution**: Make sure `train.csv` and `test.csv` are in the same folder as your Python scripts.

### Problem: "KeyError: '<PAD>'"
**Solution**: The vocabulary wasn't built correctly. Run `test_data_loading.py` to debug.

### Problem: "CUDA out of memory"
**Solution**: Reduce batch size:
```bash
python train_transformer.py --batch_size 16
```

### Problem: "RuntimeError: shape mismatch"
**Solution**: This is usually a padding issue. Check that all sequences are being padded correctly.

### Problem: Model not learning (accuracy stuck at low values)
**Possible causes**:
- Learning rate too high/low → Try 1e-3 or 1e-5
- Data loading issue → Run `test_data_loading.py`
- Vocabulary issue → Check that all tokens are included

## 📝 For Your Assignment Report

You need to include:

1. **Training Curves**: Found in `runs/transformer_XXXXX/training_curves.png`
   - Loss curve (train and val)
   - Token accuracy curve
   - Sequence accuracy curve

2. **Final Metrics**: Found in `runs/transformer_XXXXX/results.json`
   - Best validation sequence accuracy
   - Test loss, token accuracy, sequence accuracy

3. **Sample Predictions**: Use `eval.py` on 5 random validation mazes

4. **Comparison with RNN**: 
   - Training time
   - Final accuracy
   - Why Transformer performs better

## 🎯 Key Concepts to Understand

### For Transformer:
1. **Self-attention**: How tokens attend to all other tokens
2. **Positional encoding**: Adding position information via sine/cosine
3. **Masked attention**: Preventing decoder from seeing future tokens
4. **Multi-head attention**: Multiple attention mechanisms in parallel

### For Your Report:
- Transformers process sequences in parallel (faster than RNN)
- Better at capturing long-range dependencies
- No vanishing gradient problem like RNN
- Attention allows model to "focus" on relevant parts of maze

## 🎓 Tips for Success

1. **Start small**: Test with a few samples first
2. **Monitor training**: Watch the loss decrease and accuracy increase
3. **Save checkpoints**: The best model is automatically saved
4. **Visualize predictions**: Use `eval.py` to see actual predictions
5. **Compare carefully**: Note differences between forked and forkless mazes

## 📚 Additional Resources

If you want to learn more:
- Original paper: "Attention Is All You Need"
- Blog: "The Illustrated Transformer" by Jay Alammar
- Andrew Ng's Deep Learning Specialization (Sequence Models)

Good luck with your assignment! 🚀