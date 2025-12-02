"""
plotUtils.py
Plotting utilities for training visualization
"""

import matplotlib.pyplot as plt
import os
import numpy as np

def plot_training_curves(history, output_dir):
    """
    Plot comprehensive training curves including loss, accuracy, F1, precision, and recall
    
    Args:
        history: dictionary containing training metrics
        output_dir: directory to save plots
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Create a large figure with multiple subplots
    fig = plt.figure(figsize=(20, 12))
    
    # ============================================================
    # Plot 1: Training and Validation Loss
    # ============================================================
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(epochs, history['train_loss'], 'b-o', label='Train', linewidth=2, markersize=4)
    ax1.plot(epochs, history['val_loss'], 'r-s', label='Validation', linewidth=2, markersize=4)
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([1, len(epochs)])
    
    # ============================================================
    # Plot 2: Token-Level Accuracy
    # ============================================================
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(epochs, history['train_token_acc'], 'b-o', label='Train', linewidth=2, markersize=4)
    ax2.plot(epochs, history['val_token_acc'], 'r-s', label='Validation', linewidth=2, markersize=4)
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax2.set_title('Token-Level Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([1, len(epochs)])
    ax2.set_ylim([0, 1.0])
    
    # ============================================================
    # Plot 3: Sequence-Level Accuracy
    # ============================================================
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(epochs, history['val_seq_acc'], 'g-^', label='Validation', linewidth=2, markersize=5)
    ax3.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax3.set_title('Sequence-Level Accuracy', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([1, len(epochs)])
    ax3.set_ylim([0, 1.0])
    
    # ============================================================
    # Plot 4: F1 Score
    # ============================================================
    ax4 = plt.subplot(2, 3, 4)
    ax4.plot(epochs, history['val_f1'], 'm-D', label='F1 Score', linewidth=2, markersize=4)
    ax4.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax4.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    ax4.set_title('Validation F1 Score', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=11)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim([1, len(epochs)])
    ax4.set_ylim([0, 1.0])
    
    # ============================================================
    # Plot 5: Precision and Recall
    # ============================================================
    ax5 = plt.subplot(2, 3, 5)
    ax5.plot(epochs, history['val_precision'], 'c-o', label='Precision', linewidth=2, markersize=4)
    ax5.plot(epochs, history['val_recall'], 'orange', linestyle='--', marker='s', 
             label='Recall', linewidth=2, markersize=4)
    ax5.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax5.set_title('Validation Precision and Recall', fontsize=14, fontweight='bold')
    ax5.legend(fontsize=11)
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim([1, len(epochs)])
    ax5.set_ylim([0, 1.0])
    
    # ============================================================
    # Plot 6: Training Speed
    # ============================================================
    ax6 = plt.subplot(2, 3, 6)
    ax6.plot(epochs, history['batch_time'], 'purple', linestyle='-', marker='*', 
             label='Avg Batch Time', linewidth=2, markersize=6)
    ax6.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax6.set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax6.set_title('Training Speed', fontsize=14, fontweight='bold')
    ax6.legend(fontsize=11)
    ax6.grid(True, alpha=0.3)
    ax6.set_xlim([1, len(epochs)])
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(output_dir, 'training_curves_complete.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Training curves saved to: {plot_path}")
    plt.close()
    
    # ============================================================
    # Create a second figure: Combined Metrics Overview
    # ============================================================
    fig2 = plt.figure(figsize=(16, 10))
    
    # Plot 1: All Accuracy Metrics Together
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(epochs, history['train_token_acc'], 'b-o', label='Train Token Acc', 
             linewidth=2, markersize=3, alpha=0.8)
    ax1.plot(epochs, history['val_token_acc'], 'r-s', label='Val Token Acc', 
             linewidth=2, markersize=3, alpha=0.8)
    ax1.plot(epochs, history['val_seq_acc'], 'g-^', label='Val Sequence Acc', 
             linewidth=2.5, markersize=4)
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax1.set_title('All Accuracy Metrics', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([1, len(epochs)])
    ax1.set_ylim([0, 1.0])
    
    # Plot 2: F1, Precision, Recall Together
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(epochs, history['val_f1'], 'm-D', label='F1 Score', linewidth=2.5, markersize=4)
    ax2.plot(epochs, history['val_precision'], 'c-o', label='Precision', 
             linewidth=2, markersize=3, alpha=0.8)
    ax2.plot(epochs, history['val_recall'], 'orange', linestyle='--', marker='s', 
             label='Recall', linewidth=2, markersize=3, alpha=0.8)
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax2.set_title('F1, Precision, and Recall', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([1, len(epochs)])
    ax2.set_ylim([0, 1.0])
    
    # Plot 3: Loss Curves (Zoomed)
    ax3 = plt.subplot(2, 2, 3)
    ax3.plot(epochs, history['train_loss'], 'b-o', label='Train Loss', 
             linewidth=2, markersize=4)
    ax3.plot(epochs, history['val_loss'], 'r-s', label='Val Loss', 
             linewidth=2, markersize=4)
    ax3.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax3.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([1, len(epochs)])
    
    # Plot 4: Performance Summary Table
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis('off')
    
    # Get best and final metrics
    best_val_seq_acc = max(history['val_seq_acc'])
    best_epoch = history['val_seq_acc'].index(best_val_seq_acc) + 1
    final_val_seq_acc = history['val_seq_acc'][-1]
    final_val_f1 = history['val_f1'][-1]
    final_train_token_acc = history['train_token_acc'][-1]
    final_val_token_acc = history['val_token_acc'][-1]
    
    summary_text = f"""
    TRAINING SUMMARY
    {'='*40}
    
    Best Validation Sequence Accuracy:
        {best_val_seq_acc:.4f} (Epoch {best_epoch})
    
    Final Epoch Metrics:
        Train Token Acc:  {final_train_token_acc:.4f}
        Val Token Acc:    {final_val_token_acc:.4f}
        Val Seq Acc:      {final_val_seq_acc:.4f}
        Val F1 Score:     {final_val_f1:.4f}
    
    Training Progress:
        Initial Loss:     {history['train_loss'][0]:.4f}
        Final Loss:       {history['train_loss'][-1]:.4f}
        Loss Reduction:   {(1 - history['train_loss'][-1]/history['train_loss'][0])*100:.1f}%
    
    Convergence:
        Best Val Loss:    {min(history['val_loss']):.4f}
        Final Val Loss:   {history['val_loss'][-1]:.4f}
    """
    
    ax4.text(0.1, 0.95, summary_text, transform=ax4.transAxes, 
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    
    # Save the overview plot
    overview_path = os.path.join(output_dir, 'training_overview.png')
    plt.savefig(overview_path, dpi=300, bbox_inches='tight')
    print(f"✓ Training overview saved to: {overview_path}")
    plt.close()
    
    # ============================================================
    # Create metrics summary text file
    # ============================================================
    summary_path = os.path.join(output_dir, 'training_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("TRAINING SUMMARY\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Total Epochs: {len(epochs)}\n\n")
        
        f.write("BEST METRICS:\n")
        f.write("-"*60 + "\n")
        f.write(f"  Best Val Seq Acc:  {best_val_seq_acc:.4f} (Epoch {best_epoch})\n")
        f.write(f"  Best Val F1:       {max(history['val_f1']):.4f} (Epoch {history['val_f1'].index(max(history['val_f1'])) + 1})\n")
        f.write(f"  Best Val Token Acc: {max(history['val_token_acc']):.4f} (Epoch {history['val_token_acc'].index(max(history['val_token_acc'])) + 1})\n")
        f.write(f"  Lowest Val Loss:   {min(history['val_loss']):.4f} (Epoch {history['val_loss'].index(min(history['val_loss'])) + 1})\n\n")
        
        f.write("FINAL EPOCH METRICS:\n")
        f.write("-"*60 + "\n")
        f.write(f"  Train Token Acc:   {final_train_token_acc:.4f}\n")
        f.write(f"  Val Token Acc:     {final_val_token_acc:.4f}\n")
        f.write(f"  Val Seq Acc:       {final_val_seq_acc:.4f}\n")
        f.write(f"  Val F1:            {final_val_f1:.4f}\n")
        f.write(f"  Val Precision:     {history['val_precision'][-1]:.4f}\n")
        f.write(f"  Val Recall:        {history['val_recall'][-1]:.4f}\n")
        f.write(f"  Train Loss:        {history['train_loss'][-1]:.4f}\n")
        f.write(f"  Val Loss:          {history['val_loss'][-1]:.4f}\n\n")
        
        f.write("TRAINING PROGRESS:\n")
        f.write("-"*60 + "\n")
        f.write(f"  Initial Train Loss: {history['train_loss'][0]:.4f}\n")
        f.write(f"  Final Train Loss:   {history['train_loss'][-1]:.4f}\n")
        f.write(f"  Loss Reduction:     {(1 - history['train_loss'][-1]/history['train_loss'][0])*100:.1f}%\n\n")
        
        f.write(f"  Initial Token Acc:  {history['train_token_acc'][0]:.4f}\n")
        f.write(f"  Final Token Acc:    {history['train_token_acc'][-1]:.4f}\n")
        f.write(f"  Accuracy Gain:      {(history['train_token_acc'][-1] - history['train_token_acc'][0])*100:.1f}%\n\n")
        
        f.write("PERFORMANCE STATS:\n")
        f.write("-"*60 + "\n")
        avg_batch_time = np.mean(history['batch_time'])
        f.write(f"  Avg Batch Time:     {avg_batch_time:.4f}s\n")
        f.write(f"  Min Batch Time:     {min(history['batch_time']):.4f}s\n")
        f.write(f"  Max Batch Time:     {max(history['batch_time']):.4f}s\n\n")
        
        f.write("="*60 + "\n")
    
    print(f"✓ Training summary saved to: {summary_path}")
    print()


if __name__ == "__main__":
    # Example usage
    print("This module should be imported, not run directly.")
    print("Use: from utils.plotUtils import plot_training_curves")