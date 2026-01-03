import argparse
import time
import sys
import json
import os
import yaml

# Mocking PyTorch for demonstration. 
# In production, import torch and run your actual loop.
def run_mock_training(data_yaml_path, output_dir, epochs):
    print(f"LOG: Loading dataset config from {data_yaml_path}...", flush=True)
    
    # verify files exist
    if not os.path.exists(data_yaml_path):
        raise FileNotFoundError(f"Data config not found at {data_yaml_path}")
    
    # Create output directory for weights
    weights_dir = os.path.join(output_dir, 'weights')
    os.makedirs(weights_dir, exist_ok=True)
    
    print("LOG: Initializing model...", flush=True)
    time.sleep(1) # Simulate setup time

    print(f"LOG: Starting training for {epochs} epochs...", flush=True)
    
    for epoch in range(1, epochs + 1):
        # Simulate training work
        time.sleep(0.5) 
        
        # Mock metrics
        train_loss = 1.0 / (epoch + 0.1)
        val_loss = train_loss * 1.1
        mAP = epoch / epochs
        
        # ---------------------------------------------------------
        # CRITICAL: Stream progress to Flutter
        # ---------------------------------------------------------
        progress_data = {
            "type": "progress",
            "epoch": epoch,
            "total_epochs": epochs,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "mAP": mAP,
            "message": f"Epoch {epoch}/{epochs}: mAP {mAP:.2f}"
        }
        print(json.dumps(progress_data), flush=True)
        # ---------------------------------------------------------

        # Save a 'checkpoint' occasionally
        if epoch % 5 == 0 or epoch == epochs:
            ckpt_path = os.path.join(weights_dir, f'epoch_{epoch}.pt')
            with open(ckpt_path, 'w') as f:
                f.write("mock_model_data") # Replace with torch.save()
            
            print(f"LOG: Saved checkpoint to {ckpt_path}", flush=True)

    # Final success message
    result = {
        "type": "finish",
        "best_model_path": os.path.join(weights_dir, f'epoch_{epochs}.pt'),
        "message": "Training completed successfully."
    }
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TMAI Pro Training Script')
    
    # Arguments passed by Flutter
    parser.add_argument('--data', type=str, required=True, help='Path to data.yaml file')
    parser.add_argument('--save_dir', type=str, required=True, help='Path to save trained models')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    
    args = parser.parse_args()
    
    try:
        run_mock_training(args.data, args.save_dir, args.epochs)
    except Exception as e:
        # Report crashes to Flutter
        error_data = {
            "type": "error",
            "message": str(e)
        }
        print(json.dumps(error_data), flush=True)
        sys.exit(1)