cd yolor

python -m torch.distributed.launch --nproc_per_node 8 --master_port 9527 train.py --batch-size 64 --img 1280 1280 --data data/coco.yaml --cfg cfg/yolor_p6.cfg --weights '' --device 0,1,2,3,4,5,6,7 --sync-bn --name yolor_p6 --hyp hyp.scratch.1280.yaml --epochs 300
python -m torch.distributed.launch --nproc_per_node 8 --master_port 9527 tune.py --batch-size 64 --img 1280 1280 --data data/coco.yaml --cfg cfg/yolor_p6.cfg --weights 'runs/train/yolor_p6/weights/last_298.pt' --device 0,1,2,3,4,5,6,7 --sync-bn --name yolor_p6-tune --hyp hyp.finetune.1280.yaml --epochs 450
python -m torch.distributed.launch --nproc_per_node 8 --master_port 9527 train.py --batch-size 64 --img 1280 1280 --data data/coco.yaml --cfg cfg/yolor_p6.cfg --weights 'runs/train/yolor_p6-tune/weights/epoch_424.pt' --device 0,1,2,3,4,5,6,7 --sync-bn --name yolor_p6-fine --hyp hyp.finetune.1280.yaml --epochs 450


cp runs/train/yolor_p6-tune/weights/best.pt yolor_p6.pt

python prune.py

cp yolor_p6_pr20.pt ../model_compression_inference/agc2021.pt