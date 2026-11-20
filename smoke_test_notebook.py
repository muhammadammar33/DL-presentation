"""Execute every code cell of the Colab notebook on CPU with tiny settings, to catch bugs.
CIFAR10 is swapped for a local npz-backed dataset (same interface) and the epochs are cut to 1."""
import json, re, numpy as np, torch, torchvision
from PIL import Image

class LocalCIFAR(torch.utils.data.Dataset):
    def __init__(self, root, train=True, download=False, transform=None):
        d = np.load('cifar_train.npz' if train else 'cifar_test.npz')
        n = 512 if train else 512
        self.X, self.targets, self.transform = d['X'][:n], d['y'][:n].tolist(), transform
    def __len__(self): return len(self.targets)
    def __getitem__(self, i):
        x = Image.fromarray(self.X[i])
        return (self.transform(x) if self.transform else x), self.targets[i]

torchvision.datasets.CIFAR10 = LocalCIFAR
nb = json.load(open('ViT_from_scratch_CIFAR10.ipynb'))
src = []
for c in nb['cells']:
    if c['cell_type'] != 'code': continue
    s = ''.join(c['source'])
    s = s.replace('!pip -q install timm', '')
    s = re.sub(r'EPOCHS = \d+', 'EPOCHS = 1', s)
    s = re.sub(r'ABL_EPOCHS = \d+', 'ABL_EPOCHS = 1', s)
    s = s.replace('epochs=60', 'epochs=1').replace('range(3)', 'range(1)').replace('[:5000]', '[:256]')
    s = s.replace('plt.show()', 'plt.close("all")')
    src.append(s)
g = {'__name__': '__main__'}
import matplotlib; matplotlib.use('Agg')
for i, s in enumerate(src):
    print(f'--- cell {i}', flush=True)
    exec(compile(s, f'cell{i}', 'exec'), g)
print('ALL CELLS OK')
