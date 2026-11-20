"""Train ViT / CNN on CIFAR-10.  Usage: python train.py --name vit --epochs 20 [--model vit|cnn] ..."""
import argparse, json, math, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from vit import ViT, SmallResNet, count_params

p = argparse.ArgumentParser()
p.add_argument('--name', required=True)
p.add_argument('--model', default='vit')
p.add_argument('--epochs', type=int, default=20)
p.add_argument('--bs', type=int, default=128)
p.add_argument('--lr', type=float, default=1e-3)
p.add_argument('--wd', type=float, default=0.05)
p.add_argument('--warmup', type=int, default=2)
p.add_argument('--dim', type=int, default=128)
p.add_argument('--depth', type=int, default=6)
p.add_argument('--heads', type=int, default=4)
p.add_argument('--mlp_ratio', type=float, default=2.0)
p.add_argument('--patch', type=int, default=4)
p.add_argument('--no_pos', action='store_true')
p.add_argument('--no_aug', action='store_true')
p.add_argument('--train_frac', type=float, default=1.0)
p.add_argument('--threads', type=int, default=4)
p.add_argument('--seed', type=int, default=0)
p.add_argument('--cnn_w', type=int, default=48)
a = p.parse_args()
torch.manual_seed(a.seed); np.random.seed(a.seed)
torch.set_num_threads(a.threads)

MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)

def load(split):
    d = np.load(f'cifar_{split}.npz')
    x = torch.from_numpy(d['X']).permute(0, 3, 1, 2).float().div(255)
    return (x - MEAN) / STD, torch.from_numpy(d['y'])

Xtr, ytr = load('train'); Xte, yte = load('test')
if a.train_frac < 1:
    idx = torch.randperm(len(ytr))[: int(len(ytr) * a.train_frac)]
    Xtr, ytr = Xtr[idx], ytr[idx]

def augment(x):
    # random crop (pad 4) + horizontal flip, on a batch tensor
    B = x.shape[0]
    xp = F.pad(x, (4, 4, 4, 4), mode='reflect')
    i, j = np.random.randint(0, 9, 2)
    x = xp[:, :, i:i + 32, j:j + 32]
    flip = torch.rand(B) < 0.5
    x[flip] = x[flip].flip(3)
    return x

if a.model == 'vit':
    model = ViT(patch_size=a.patch, dim=a.dim, depth=a.depth, heads=a.heads,
                mlp_ratio=a.mlp_ratio, use_pos=not a.no_pos)
else:
    model = SmallResNet(w=(a.cnn_w, 2 * a.cnn_w, 4 * a.cnn_w))
nparams = count_params(model)
print(f'{a.name}: {nparams/1e6:.3f}M params', flush=True)

decay = [p_ for n, p_ in model.named_parameters() if p_.ndim >= 2 and 'pos' not in n and 'cls' not in n]
no_decay = [p_ for n, p_ in model.named_parameters() if not (p_.ndim >= 2 and 'pos' not in n and 'cls' not in n)]
opt = torch.optim.AdamW([{'params': decay, 'weight_decay': a.wd}, {'params': no_decay, 'weight_decay': 0}], lr=a.lr)
steps_per_epoch = math.ceil(len(ytr) / a.bs)
total = a.epochs * steps_per_epoch
warm = a.warmup * steps_per_epoch
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
crit = nn.CrossEntropyLoss(label_smoothing=0.1)

@torch.no_grad()
def evaluate():
    model.eval(); correct = 0; preds = []
    for k in range(0, len(yte), 500):
        out = model(Xte[k:k + 500]).argmax(1); preds.append(out)
        correct += (out == yte[k:k + 500]).sum().item()
    return correct / len(yte), torch.cat(preds)

hist = []
t0 = time.time()
for ep in range(a.epochs):
    model.train(); perm = torch.randperm(len(ytr)); tl = 0; tc = 0
    for k in range(0, len(ytr), a.bs):
        b = perm[k:k + a.bs]; x, y = Xtr[b], ytr[b]
        if not a.no_aug:
            x = augment(x)
        out = model(x); loss = crit(out, y)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        tl += loss.item() * len(b); tc += (out.argmax(1) == y).sum().item()
    acc, preds = evaluate()
    hist.append({'epoch': ep + 1, 'train_loss': tl / len(ytr), 'train_acc': tc / len(ytr), 'test_acc': acc,
                 'time_s': time.time() - t0})
    print(json.dumps(hist[-1]), flush=True)

json.dump({'name': a.name, 'args': vars(a), 'params': nparams, 'history': hist,
           'final_test_acc': hist[-1]['test_acc'], 'preds': preds.tolist()},
          open(f'results/{a.name}.json', 'w'))
torch.save(model.state_dict(), f'results/{a.name}.pt')
