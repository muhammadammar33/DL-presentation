"""Fine-tune a pretrained ViT-Tiny/16 (ImageNet-21k -> 1k) on CIFAR-10 at 224px (CPU budget version)."""
import argparse, json, time, numpy as np, torch, torch.nn.functional as F, timm

p = argparse.ArgumentParser()
p.add_argument('--train_n', type=int, default=10000)
p.add_argument('--test_n', type=int, default=2000)
p.add_argument('--epochs', type=int, default=1)
p.add_argument('--bs', type=int, default=32)
p.add_argument('--lr', type=float, default=5e-5)
p.add_argument('--threads', type=int, default=4)
a = p.parse_args()
torch.manual_seed(0); torch.set_num_threads(a.threads)

def load(split, n):
    d = np.load(f'cifar_{split}.npz'); idx = np.random.RandomState(0).permutation(len(d['y']))[:n]
    x = torch.from_numpy(d['X'][idx]).permute(0, 3, 1, 2).float() / 255
    return (x - 0.5) / 0.5, torch.from_numpy(d['y'][idx])

Xtr, ytr = load('train', a.train_n); Xte, yte = load('test', a.test_n)
up = lambda x: F.interpolate(x, size=224, mode='bilinear', align_corners=False)
m = timm.create_model('vit_tiny_patch16_224.augreg_in21k_ft_in1k', pretrained=True, num_classes=10)
opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=0.05)
steps = a.epochs * (a.train_n // a.bs)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.1)

@torch.no_grad()
def evaluate():
    m.eval(); c = 0
    for k in range(0, len(yte), 100):
        c += (m(up(Xte[k:k + 100])).argmax(1) == yte[k:k + 100]).sum().item()
    return c / len(yte)

t0 = time.time(); step = 0
for ep in range(a.epochs):
    m.train(); perm = torch.randperm(len(ytr))
    for k in range(0, a.train_n - a.bs + 1, a.bs):
        b = perm[k:k + a.bs]; x = Xtr[b]
        flip = torch.rand(len(b)) < 0.5; x[flip] = x[flip].flip(3)
        loss = F.cross_entropy(m(up(x)), ytr[b])
        opt.zero_grad(); loss.backward(); opt.step(); sched.step(); step += 1
        if step % 20 == 0:
            print(f'step {step}/{steps} loss {loss.item():.3f} {time.time() - t0:.0f}s', flush=True)
acc = evaluate()
print('test acc', acc)
json.dump({'test_acc': acc, 'train_images': a.train_n, 'test_images': a.test_n, 'epochs': a.epochs, 'lr': a.lr,
           'params': sum(q.numel() for q in m.parameters()), 'time_s': time.time() - t0}, open('results/finetune.json', 'w'))
