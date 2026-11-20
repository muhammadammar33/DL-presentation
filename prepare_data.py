"""Download CIFAR-10 and save it as cifar_train.npz / cifar_test.npz, the format train.py,
finetune.py and analysis.py read.

    python prepare_data.py               # via torchvision (official source, ~170 MB download)
    python prepare_data.py --source hf   # via the Hugging Face mirror (needs: pip install pyarrow)

Each .npz holds X: uint8 array (N, 32, 32, 3) and y: int64 labels (N,).
"""
import argparse
import io
import os
import numpy as np


def from_torchvision():
    import torchvision
    out = {}
    for split, train in [('train', True), ('test', False)]:
        ds = torchvision.datasets.CIFAR10('data', train=train, download=True)
        out[split] = (np.asarray(ds.data, dtype=np.uint8), np.asarray(ds.targets, dtype=np.int64))
    return out


def from_hf():
    import urllib.request
    import pyarrow.parquet as pq
    from PIL import Image
    base = 'https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/'
    os.makedirs('data', exist_ok=True)
    out = {}
    for split in ['train', 'test']:
        path = f'data/{split}.parquet'
        if not os.path.exists(path):
            print(f'downloading {split} split ...', flush=True)
            urllib.request.urlretrieve(base + f'{split}-00000-of-00001.parquet', path)
        t = pq.read_table(path).to_pydict()
        X = np.stack([np.array(Image.open(io.BytesIO(d['bytes'])).convert('RGB')) for d in t['img']])
        out[split] = (X.astype(np.uint8), np.asarray(t['label'], dtype=np.int64))
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source', choices=['torchvision', 'hf'], default='torchvision')
    a = p.parse_args()
    data = from_torchvision() if a.source == 'torchvision' else from_hf()
    for split, (X, y) in data.items():
        assert X.shape[1:] == (32, 32, 3), X.shape
        np.savez(f'cifar_{split}.npz', X=X, y=y)
        print(f'cifar_{split}.npz: {X.shape[0]} images, per-class counts {np.bincount(y).tolist()}')
