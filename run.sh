python train_detection.py \
--weight dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth \
--model-name dinov3_convnext_tiny \
--imgsz 640 640 \
--lr 0.0005  \
--epochs 1 \
--scheduler-epochs 15 \
--workers 8 \
--batch 32 \
--config detection_configs/voc.yaml \
--out-dir trial_runs \
--head ssd

python train_detection.py \
--weight dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth \
--model-name dinov3_convnext_tiny \
--imgsz 640 640 \
--lr 0.0005  \
--epochs 1 \
--scheduler-epochs 15 \
--workers 8 \
--batch 32 \
--config detection_configs/voc.yaml \
--out-dir trial_runs \
--head retinanet

python train_detection.py \
--weight dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth \
--model-name dinov3_convnext_tiny \
--imgsz 640 640 \
--lr 0.0005  \
--epochs 1 \
--scheduler-epochs 15 \
--workers 8 \
--batch 32 \
--config detection_configs/voc.yaml \
--out-dir trial_runs \
--head ssd

python train_detection.py \
--weight dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth \
--model-name dinov3_convnext_tiny \
--imgsz 640 640 \
--lr 0.0005  \
--epochs 1 \
--scheduler-epochs 15 \
--workers 8 \
--batch 32 \
--config detection_configs/voc.yaml \
--out-dir trial_runs \
--head retinanet