"""
Quick commands:

1) Prepare VisDrone to YOLO format
   py tools/prepare_visdrone.py --src-root datasets/VisDrone2019 --out-root datasets/VisDrone2019-DET-YOLO

2) Baseline
   py train.py --data configs/datasets/visdrone_det.yaml

3) A only
   py train.py --data configs/datasets/visdrone_det.yaml --module-a

4) B only
   py train.py --data configs/datasets/visdrone_det.yaml --module-b

5) C only
   py train.py --data configs/datasets/visdrone_det.yaml --module-c

6) A+B+C
   py train.py --data configs/datasets/visdrone_det.yaml --module-a --module-b --module-c

7) Inference
   py test.py --weights runs/train/yolov8s_visdrone_ABC/weights/best.pt --source your_sample.jpg --save
"""

if __name__ == "__main__":
    print(__doc__)
