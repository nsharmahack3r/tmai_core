import argparse
import json
import sys
import os
from ultralytics import YOLO

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

# ... (keep the on_train_epoch_end callback exactly as before) ...

def run_training(data_yaml, project_dir, run_name, epochs, batch_size, model_name):
    # 1. Initialize Model
    # model_name = 'yolov8n.pt' or path to existing .pt file
    print(json.dumps({"type": "log", "message": f"Loading model {model_name}..."}), flush=True)
    model = YOLO(model_name)

    # 2. Register Custom Callback
    model.add_callback("on_train_epoch_end", on_train_epoch_end)

    # 3. Start Training
    print(json.dumps({"type": "log", "message": "Starting training loop..."}), flush=True)
    
    # YOLOv8 Path Logic:
    # Results will be saved at: project_dir / run_name
    # e.g. project_path/models/model_name
    results = model.train(
        data=data_yaml,
        project=project_dir,  # Parent directory
        name=run_name,        # Sub-directory
        epochs=epochs,
        batch=batch_size,
        exist_ok=True,        # If folder exists, overwrite/append inside it
        verbose=False
    )

    # 4. Final Success Path
    # YOLO saves weights in /weights/best.pt inside the run folder
    best_weight_path = os.path.join(project_dir, run_name, 'weights', 'best.pt')
    
    success_msg = {
        "type": "finish",
        "best_model_path": best_weight_path,
        "message": "Training completed successfully."
    }
    print(json.dumps(success_msg), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--project', type=str, required=True, help='Parent folder for models')
    parser.add_argument('--name', type=str, required=True, help='Name of this specific run')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='Architecture (yolov8n.pt, etc)')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16)

    args = parser.parse_args()

    try:
        run_training(
            data_yaml=args.data, 
            project_dir=args.project, 
            run_name=args.name, 
            epochs=args.epochs, 
            batch_size=args.batch_size, 
            model_name=args.model
        )
    except Exception as e:
        error_msg = {
            "type": "error",
            "message": str(e)
        }
        print(json.dumps(error_msg), flush=True)
        sys.exit(1)