from ultralytics import YOLO
import os
import argparse

# Tune on 17R

def train(region, weight):
    dataset_name = region + '_top_salient'
    print("Training on Dataset: {}".format(dataset_name))
    yaml_path = os.path.join('datasets', dataset_name, 'dataset.yaml')
    run_name = 'yolov8s_custom_w' + str(weight) + '_' + dataset_name
    print("Saving as run: {}".format(run_name))

    model = YOLO('yolov8s.pt')

    '''if r == '16T':
        model = YOLO('runs/detect/yolov8s_custom_16T_top_salient/weights/last.pt')
        # Train
        results = model.train(
            data=yaml_path,
            name=run_name,
            degrees=180,
            scale=0.5,
            fliplr=0.0,
            imgsz=1216,
            mosaic=0.5,
            batch=12,
            perspective=0.0001,
            plots=True,
            save=True,
            resume=True,
            epochs=300
        )
    else:'''
    results = model.train(
        data=yaml_path,
        name=run_name,
        degrees=180,
        scale=0.5,
        fliplr=0.0,
        imgsz=1216,
        mosaic=0.5,
        batch=12,
        perspective=0.0001,
        plots=True,
        save=True,
        epochs=300
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run custom LD model with different weights on MSE loss component")
    parser.add_argument("--region", default="17R", help="region name")
    parser.add_argument("--mse_weight", default=1.0, help="proportion of box loss to use as MSE loss coefficient")

    args = parser.parse_args()
    train(args.region, args.mse_weight)