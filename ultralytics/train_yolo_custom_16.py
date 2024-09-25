from ultralytics import YOLO
#from models.yolo.detect import DetectionTrainer
import os

# 16 regions (change this to what datasets are available)
regions = ['32S', '32T', '33S', '33T', '53S', '52S', '54S', '54T']
#regions = ['17R', '12R', '16T', '54S', '10S', '10T', '11R', '32S', '33S', '33T', '53S', '52S', '54T', '32T', '18S']
#regions = ['17R']

for r in regions:
    dataset_name = r + '_top_salient'
    print("Training on Dataset: {}".format(dataset_name))
    yaml_path = os.path.join('datasets', dataset_name, 'dataset.yaml')
    #run_name = 'yolov8l_custom_tuned_' + dataset_name
    run_name = 'yolov8s_custom_' + dataset_name
    print("Saving as run: {}".format(run_name))

    model = YOLO('yolov8s.pt')

    if r == '32S':
        model = YOLO('runs/detect/yolov8s_custom_32S_top_salient/weights/last.pt')
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
    else:
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
        '''results = model.train(
            data=yaml_path,
            name=run_name,
            degrees=180,
            fliplr=0.0,
            imgsz=1216,
            box=4.83901,
            mse=5.90031,
            cls=1.07963,
            dfl=1.85788,
            hsv_h=0.02312,
            hsv_s=0.77631,
            hsv_v=0.50805,
            translate=0.10477,
            scale=0.48706,
            shear=0.0,
            flipud=0.0,
            mosaic=0.92367,
            #perspective=0.0
            perspective=0.0001,
            batch=12,
            plots=True,
            save=True,
            epochs=300
        )'''
        '''# tuned hyperparams for 17R 
        results = model.train(
            data=yaml_path,
            name=run_name,
            degrees=180,
            fliplr=0.0,
            imgsz=1216,
            box=6.95499,
            mse=9.57985,
            cls=1.79619,
            dfl=1.62749,
            hsv_h=0.01264,
            hsv_s=0.50185,
            hsv_v=0.27928,
            translate=0.11025,
            scale=0.45495,
            shear=0.0,
            #perspective=0.0
            perspective=0.0001,
            flipud=0.0,
            mosaic=0.79163,
            batch=8, # originally 12 for s model
            plots=True,
            save=True,
            epochs=300
        )'''