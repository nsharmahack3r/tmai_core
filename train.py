import argparse
import json
import sys
import os
from ultralytics import YOLO

# --- Callback for Flutter Communication ---
def on_train_epoch_end(trainer):
    """
    This function is called by YOLOv8 at the end of every epoch.
    We extract metrics and print them as a JSON line for the Flutter app.
    """
    try:
        # Extract metrics
        # YOLOv8 stores metrics in trainer.metrics (a dict)
        metrics = trainer.metrics
        
        # Safe extraction of key values
        # Note: Key names can vary slightly between versions, but these are standard
        box_loss = trainer.loss_items[0] if hasattr(trainer, 'loss_items') else 0.0
        
        # map50 is Mean Average Precision at IoU=0.50
        map50 = metrics.get("metrics/mAP50(B)", 0.0)
        
        # Current Epoch (1-based index for UI)
        current_epoch = trainer.epoch + 1
        total_epochs = trainer.epochs

        progress_data = {
            "type": "progress",
            "epoch": current_epoch,
            "total_epochs": total_epochs,
            "train_loss": float(box_loss), 
            "mAP": float(map50),
            "message": f"Epoch {current_epoch}/{total_epochs} completed. mAP: {map50:.3f}"
        }
        
        # flush=True is REQUIRED for Flutter to receive the data immediately
        print(json.dumps(progress_data), flush=True)

    except Exception as e:
        # If logging fails, don't crash the training, just print error safely
        error_log = {"type": "log", "message": f"Error in callback: {str(e)}"}
        print(json.dumps(error_log), flush=True)

def run_training(data_yaml, save_dir, epochs, batch_size, model_name='yolov8n.pt'):
    # 1. Initialize Model
    # 'yolov8n.pt' will download automatically if not found (requires internet first time)
    # You might want to bundle this weight file in your repo to avoid download issues.
    print(json.dumps({"type": "log", "message": "Loading YOLOv8 model..."}), flush=True)
    model = YOLO(model_name)

    # 2. Register Custom Callback
    model.add_callback("on_train_epoch_end", on_train_epoch_end)

    # 3. Start Training
    print(json.dumps({"type": "log", "message": "Starting training loop..."}), flush=True)
    
    # YOLOv8 arguments:
    # project: the folder to save results in
    # name: the subfolder name (we set it to empty so it uses 'project' directly if possible)
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        project=save_dir, 
        name='train_run', # This creates save_dir/train_run/weights/best.pt
        verbose=False,    # Reduce CLI noise so we don't mess up our JSON stream
        exist_ok=True     # Overwrite if exists (optional)
    )

    # 4. Final Success
    # The best model is always saved at /path/to/project/name/weights/best.pt
    best_weight_path = os.path.join(save_dir, 'train_run', 'weights', 'best.pt')
    
    success_msg = {
        "type": "finish",
        "best_model_path": best_weight_path,
        "message": "Training completed successfully."
    }
    print(json.dumps(success_msg), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='Path to data.yaml')
    parser.add_argument('--save_dir', type=str, required=True, help='Path to save results')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='Base model (n, s, m, l, x)')

    args = parser.parse_args()

    try:
        run_training(args.data, args.save_dir, args.epochs, args.batch_size, args.model)
    except Exception as e:
        error_msg = {
            "type": "error",
            "message": str(e)
        }
        print(json.dumps(error_msg), flush=True)
        sys.exit(1)