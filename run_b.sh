python3 train.py --name cnn_main --model cnn --epochs 25 --threads 2
python3 train.py --name vit_noaug --epochs 10 --no_aug --threads 2
python3 train.py --name cnn_10pct --model cnn --epochs 30 --train_frac 0.1 --warmup 3 --threads 2
python3 train.py --name vit_main10 --epochs 10 --threads 2
