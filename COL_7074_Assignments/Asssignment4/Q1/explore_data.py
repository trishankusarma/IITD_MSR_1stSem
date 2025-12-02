import pandas as pd
import ast

train_data = pd.read_csv('data/train_6x6_mazes.csv')

# Inspect one example quickly
example = train_data.iloc[0]

input_seq = ast.literal_eval(example['input_sequence'])
output_path = ast.literal_eval(example['output_path'])

print("Input tokens:", input_seq[:20], "...")
print("Output tokens:", output_path[:20], "...")
print("Input length:", len(input_seq))
print("Output length:", len(output_path))
