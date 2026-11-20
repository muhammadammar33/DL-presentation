python3 train.py --name vit_main --epochs 25 --threads 2
python3 train.py --name vit_nopos --epochs 10 --no_pos --threads 2
python3 train.py --name vit_p8 --epochs 10 --patch 8 --threads 2
python3 train.py --name vit_10pct --epochs 30 --train_frac 0.1 --warmup 3 --threads 2
