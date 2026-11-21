"""Classify your own photos with our ViT (from scratch) and the fine-tuned pretrained ViT.

    python predict.py examples/cat.jpg
    python predict.py examples/*.jpg                 # several images (macOS/Linux shell)
    python predict.py my_photo.png --model scratch   # only our from-scratch ViT
    python predict.py examples/cat.jpg --no-show     # save the figure, don't open a window

Prints the top-3 guesses of each model and saves predictions.png: the photo, the 32x32 image
the models actually see, where our ViT looked (attention rollout), and both models' confidence
for all 10 classes.

The models only know the 10 CIFAR-10 classes: airplane, automobile, bird, cat, deer, dog,
frog, horse, ship, truck. Any other photo is forced into one of these 10.
"""
import argparse
import glob
import sys
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from vit import ViT

CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)


def load_image(path):
    """Centre-crop to a square and shrink to 32x32, exactly like a CIFAR-10 image."""
    im = Image.open(path).convert('RGB')
    s = min(im.size)
    left, top = (im.width - s) // 2, (im.height - s) // 2
    im = im.crop((left, top, left + s, top + s))
    small = im.resize((32, 32), Image.BICUBIC)
    x = torch.from_numpy(np.array(small)).permute(2, 0, 1).float().unsqueeze(0) / 255  # (1, 3, 32, 32) in [0, 1]
    return im, small, x


def load_scratch():
    m = ViT(dim=128, depth=6, heads=4)
    m.load_state_dict(torch.load('results/vit_main.pt', map_location='cpu'))
    return m.eval()


def load_pretrained():
    import timm
    m = timm.create_model('vit_tiny_patch16_224.augreg_in21k_ft_in1k', pretrained=False, num_classes=10)
    m.load_state_dict(torch.load('results/finetuned_vit_tiny.pt', map_location='cpu'))
    return m.eval()


@torch.no_grad()
def predict_scratch(m, x):
    logits = m((x - MEAN) / STD, keep_attn=True)
    # attention rollout: average heads, add identity for the residual, chain the 6 layers
    R = torch.eye(65)
    for blk in m.blocks:
        A = blk.attn.last_attn[0].mean(0)
        A = 0.5 * A + 0.5 * torch.eye(65)
        A = A / A.sum(-1, keepdim=True)
        R = A @ R
    rollout = R[0, 1:].reshape(8, 8)
    return logits.softmax(1)[0], (rollout - rollout.min()) / (rollout.max() - rollout.min())


@torch.no_grad()
def predict_pretrained(m, x):
    # same preparation as in finetune.py: the 32x32 image is upscaled to 224x224, scaled to [-1, 1]
    x224 = F.interpolate(x, size=224, mode='bilinear', align_corners=False)
    return m((x224 - 0.5) / 0.5).softmax(1)[0]


def top3(p):
    v, i = p.topk(3)
    return ',  '.join(f'{CLASSES[k]} {100 * s:.1f}%' for s, k in zip(v.tolist(), i.tolist()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('images', nargs='+')
    ap.add_argument('--model', choices=['both', 'scratch', 'pretrained'], default='both')
    ap.add_argument('--out', default='predictions.png')
    ap.add_argument('--no-show', action='store_true', help="save the figure but don't open a window")
    a = ap.parse_args()
    torch.set_num_threads(4)

    scratch = load_scratch() if a.model in ('both', 'scratch') else None
    pre = None
    if a.model in ('both', 'pretrained'):
        try:
            pre = load_pretrained()
        except (FileNotFoundError, ImportError) as e:
            print(f'(pretrained model unavailable: {e}; run finetune.py first, or use --model scratch)')

    # Windows PowerShell does not expand wildcards like examples/*.jpg, so expand them here
    paths = [q for p in a.images for q in (sorted(glob.glob(p)) or [p])]
    rows = []
    for path in paths:
        try:
            im, small, x = load_image(path)
        except (FileNotFoundError, OSError) as e:
            print(f'{path}: cannot open ({e})')
            continue
        print(f'\n{path}')
        ps = ro = pp = None
        if scratch is not None:
            ps, ro = predict_scratch(scratch, x)
            print(f'  our ViT (from scratch):  {top3(ps)}')
        if pre is not None:
            pp = predict_pretrained(pre, x)
            print(f'  pretrained ViT-Tiny:     {top3(pp)}')
        rows.append((path, im, small, ps, ro, pp))
    if not rows:
        sys.exit('no images could be read')

    import matplotlib
    if a.no_show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    n = len(rows)
    fig, axs = plt.subplots(n, 4, figsize=(16, 3.6 * n), squeeze=False,
                            gridspec_kw={'width_ratios': [1, 1, 1, 1.6]})
    for r, (path, im, small, ps, ro, pp) in enumerate(rows):
        ax = axs[r]
        ax[0].imshow(im); ax[0].set_title(path.replace('\\', '/').split('/')[-1], fontsize=11)
        ax[1].imshow(small, interpolation='nearest'); ax[1].set_title('what the model sees: 32x32', fontsize=11)
        if ro is not None:
            ax[2].imshow(small, interpolation='nearest')
            ax[2].imshow(ro, cmap='inferno', alpha=0.55, extent=(-0.5, 31.5, 31.5, -0.5), interpolation='nearest')
            ax[2].set_title('where our ViT looked (rollout)', fontsize=11)
        for k in range(3):
            ax[k].axis('off')
        y = np.arange(10)
        h = 0.38 if (ps is not None and pp is not None) else 0.7
        if ps is not None:
            ax[3].barh(y - (h / 2 if pp is not None else 0), 100 * ps.numpy(), height=h, color='#D9661A', label='our ViT (from scratch)')
        if pp is not None:
            ax[3].barh(y + (h / 2 if ps is not None else 0), 100 * pp.numpy(), height=h, color='#7A5CC7', label='pretrained ViT-Tiny')
        ax[3].set_yticks(y); ax[3].set_yticklabels(CLASSES); ax[3].invert_yaxis()
        ax[3].set_xlim(0, 100); ax[3].set_xlabel('confidence (%)')
        ax[3].legend(loc='lower right', fontsize=9, frameon=False)
        for s in ('top', 'right'):
            ax[3].spines[s].set_visible(False)
    plt.tight_layout()
    plt.savefig(a.out, dpi=130)
    print(f'\nsaved {a.out}')
    if not a.no_show:
        plt.show()


if __name__ == '__main__':
    main()
