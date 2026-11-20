"""Live demo for the presentation. Each part runs in seconds on a laptop CPU.

    python demo.py shapes     # follow one image through the ViT, printing every tensor shape
    python demo.py params     # parameter count per component, formula vs PyTorch
    python demo.py check      # our attention vs PyTorch's built-in attention
    python demo.py predict    # trained ViT classifies 8 random test images, then the full test set
    python demo.py all        # everything above, in order
"""
import math
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from vit import ViT, MultiHeadSelfAttention, count_params

torch.set_num_threads(4)
CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)


def header(t):
    print('\n' + '=' * 70 + f'\n{t}\n' + '=' * 70)


def load_model():
    m = ViT(dim=128, depth=6, heads=4)
    m.load_state_dict(torch.load('results/vit_main.pt', map_location='cpu'))
    return m.eval()


def load_test():
    d = np.load('cifar_test.npz')
    x = torch.from_numpy(d['X']).permute(0, 3, 1, 2).float() / 255
    return (x - MEAN) / STD, torch.from_numpy(d['y'])


@torch.no_grad()
def shapes():
    header('1. One image through the Vision Transformer')
    m = load_model()
    x = torch.randn(1, 3, 32, 32)
    print(f'{"input image":40s} {tuple(x.shape)}   (batch, channels, height, width)')
    z = m.patch.proj(x)
    print(f'{"Conv2d(k=4, stride=4)":40s} {tuple(z.shape)}   128 features on an 8x8 grid of patches')
    z = z.flatten(2).transpose(1, 2)
    print(f'{"flatten + transpose":40s} {tuple(z.shape)}   64 patch tokens x 128 features')
    z = torch.cat([m.cls.expand(1, -1, -1), z], 1)
    print(f'{"prepend CLS token":40s} {tuple(z.shape)}   65 tokens')
    z = z + m.pos
    print(f'{"add position embeddings":40s} {tuple(z.shape)}   same shape, now position-aware')
    a = m.blocks[0].attn
    qkv = a.qkv(m.blocks[0].ln1(z)).reshape(1, 65, 3, a.h, a.dk).permute(2, 0, 3, 1, 4)
    print(f'{"Q, K, V per head":40s} {tuple(qkv[0].shape)}   (batch, heads, tokens, d_k)')
    print(f'{"attention matrix softmax(QK^T/sqrt(32))":40s} (1, 4, 65, 65)   every token vs every token, per head')
    for blk in m.blocks:
        z = blk(z)
    print(f'{"after 6 encoder blocks":40s} {tuple(z.shape)}   shape never changes inside the encoder')
    cls = m.ln(z)[:, 0]
    print(f'{"final LayerNorm, take token 0 (CLS)":40s} {tuple(cls.shape)}')
    print(f'{"linear head -> class scores":40s} {tuple(m.head(cls).shape)}   10 logits')


def params():
    header('2. Where the 811,146 parameters live')
    m = load_model()
    D, r = 128, 2
    rows = [
        ('Patch embedding', '48*128 + 128', 48 * D + D, count_params(m.patch)),
        ('CLS token + positions', '128 + 65*128', D + 65 * D, m.cls.numel() + m.pos.numel()),
        ('One encoder block', '4D^2+4D + 2rD^2+(r+1)D + 4D', (4 * D * D + 4 * D) + (2 * r * D * D + (r + 1) * D) + 4 * D, count_params(m.blocks[0])),
        ('6 encoder blocks', '6 x 132,480', 6 * 132480, sum(count_params(b) for b in m.blocks)),
        ('Final LN + head', '256 + 128*10 + 10', 2 * D + D * 10 + 10, count_params(m.ln) + count_params(m.head)),
    ]
    print(f'{"component":24s} {"formula":30s} {"by hand":>10s} {"PyTorch":>10s}')
    for n, f, h, p in rows:
        print(f'{n:24s} {f:30s} {h:>10,} {p:>10,}  {"OK" if h == p else "MISMATCH"}')
    print(f'{"TOTAL":24s} {"":30s} {"":>10s} {count_params(m):>10,}')


@torch.no_grad()
def check():
    header('3. Is our hand-written attention correct?')
    torch.manual_seed(0)
    att = MultiHeadSelfAttention(64, heads=4)
    x = torch.randn(2, 10, 64)
    ours = att(x)
    q, k, v = att.qkv(x).reshape(2, 10, 3, 4, 16).permute(2, 0, 3, 1, 4)
    ref = att.out(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(2, 10, 64))
    print('same input into our MultiHeadSelfAttention and into PyTorch scaled_dot_product_attention')
    print(f'max absolute difference: {(ours - ref).abs().max().item():.2e}   (float32 rounding is ~1e-7)')
    print(f'initial loss of an untrained model should be ln(10) = {math.log(10):.3f}:', end=' ')
    m = ViT(dim=128, depth=6, heads=4)
    print(f'{F.cross_entropy(m(torch.randn(256, 3, 32, 32)), torch.randint(0, 10, (256,))).item():.3f}')


@torch.no_grad()
def predict():
    header('4. The trained ViT on CIFAR-10 test images')
    m = load_model()
    X, y = load_test()
    idx = torch.randperm(len(y))[:8]
    probs = m(X[idx]).softmax(1)
    for i, p in zip(idx.tolist(), probs):
        c = p.argmax().item()
        mark = 'correct' if c == y[i].item() else f'WRONG (true: {CLASSES[y[i]]})'
        print(f'test image {i:5d}: predicted {CLASSES[c]:10s} {p[c].item() * 100:5.1f}% confident   {mark}')
    t = time.time()
    pred = torch.cat([m(X[k:k + 500]).argmax(1) for k in range(0, len(y), 500)])
    acc = (pred == y).float().mean().item()
    print(f'\nall 10,000 test images: {acc * 100:.2f}% accuracy   ({time.time() - t:.1f} s on CPU)')


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'all'
    steps = {'shapes': [shapes], 'params': [params], 'check': [check], 'predict': [predict],
             'all': [shapes, params, check, predict]}
    if what not in steps:
        sys.exit(__doc__)
    for f in steps[what]:
        f()
