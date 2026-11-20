import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s))
code = lambda s: C.append(nbf.v4.new_code_cell(s))

vit_src = open('vit.py').read()

md("""# Vision Transformer from Scratch: CIFAR-10
**Group presentation: Transformers for Vision (Implementation track)**

This notebook goes with our slides. It:
1. Builds a **Vision Transformer (ViT)** completely from scratch in PyTorch: patch embedding, multi-head self-attention, pre-LN encoder blocks, CLS token and positional embeddings.
2. Trains it on **CIFAR-10** and compares it to a **ResNet-style CNN** of similar size.
3. Runs **ablations**: positional embeddings, patch size, data augmentation and dataset size (the *inductive bias* story).
4. **Fine-tunes a pretrained ViT** (ImageNet-21k) to show the effect of pre-training.
5. Looks **inside** the model: attention maps, attention rollout, learned positional embeddings, patch-embedding filters and attention distance per head.

**Runtime:** `Runtime → Change runtime type → T4 GPU`. Section 3 takes about 15 to 25 min on a T4 and section 6 about 10 min. Lower `EPOCHS` if you are short on time.""")

code("""!pip -q install timm
import math, time, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import torchvision, torchvision.transforms as T
import matplotlib.pyplot as plt
plt.rcParams['image.interpolation'] = 'nearest'   # show 32x32 images pixel-sharp, not blurred
plt.rcParams['figure.dpi'] = 150
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(0)
print(device, torch.__version__)""")

md("""## 1. Data: CIFAR-10
50,000 training and 10,000 test images, each 32×32×3, in 10 classes. Training uses the standard augmentation for CIFAR: a random crop after 4-pixel padding, plus a horizontal flip.""")
code("""MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
CLASSES = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
train_tf = T.Compose([T.RandomCrop(32, padding=4, padding_mode='reflect'), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(MEAN, STD)])
test_tf  = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])
train_ds = torchvision.datasets.CIFAR10('data', train=True,  download=True, transform=train_tf)
test_ds  = torchvision.datasets.CIFAR10('data', train=False, download=True, transform=test_tf)
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True,  num_workers=2, pin_memory=True, drop_last=True)
test_dl  = torch.utils.data.DataLoader(test_ds,  batch_size=512, shuffle=False, num_workers=2, pin_memory=True)
print(len(train_ds), len(test_ds))""")

md("""## 2. The model, written from scratch
Here is how each piece maps onto the paper (Dosovitskiy et al., *An Image is Worth 16×16 Words*, ICLR 2021):

| Equation | Code |
|---|---|
| $z_0 = [x_{cls};\\, x_p^1E;\\, \\dots;\\, x_p^NE] + E_{pos}$ | `PatchEmbedding`, `self.cls`, `self.pos` |
| $z'_\\ell = \\text{MSA}(\\text{LN}(z_{\\ell-1})) + z_{\\ell-1}$ | `EncoderBlock` (first line) |
| $z_\\ell = \\text{MLP}(\\text{LN}(z'_\\ell)) + z'_\\ell$ | `EncoderBlock` (second line) |
| $y = \\text{LN}(z_L^0)$ → linear head | `self.ln`, `self.head` |
| $\\text{Attention}(Q,K,V)=\\text{softmax}(QK^\\top/\\sqrt{d_k})V$ | `MultiHeadSelfAttention` |

We deliberately **don't** use `nn.MultiheadAttention` or `timm` for this model, so you can see every operation.""")
code(vit_src)

md("""### 2.1 Shape walkthrough: following one image through the network""")
code("""m = ViT(patch_size=4, dim=192, depth=6, heads=3)
x = torch.randn(1, 3, 32, 32)
p = m.patch(x);                                print('patch embedding      ', tuple(p.shape), ' (N = (32/4)^2 = 64 patches, D = 192)')
z = torch.cat([m.cls.expand(1,-1,-1), p], 1);  print('+ CLS token          ', tuple(z.shape))
z = z + m.pos;                                 print('+ positional emb.    ', tuple(z.shape))
a = m.blocks[0].attn
qkv = a.qkv(m.blocks[0].ln1(z)).reshape(1, 65, 3, a.h, a.dk).permute(2,0,3,1,4)
print('Q, K, V (per head)   ', tuple(qkv[0].shape), ' (B, heads, T, d_k)')
print('attention matrix     ', (1, a.h, 65, 65))
for blk in m.blocks: z = blk(z)
print('after 6 blocks       ', tuple(z.shape))
print('logits               ', tuple(m.head(m.ln(z)[:,0]).shape))
print(f'parameters: {count_params(m):,}')""")

md("""### 2.2 Sanity check: our attention matches PyTorch's fused kernel""")
code("""att = MultiHeadSelfAttention(64, 4); x = torch.randn(2, 10, 64)
q, k, v = att.qkv(x).reshape(2, 10, 3, 4, 16).permute(2, 0, 3, 1, 4)
ref = att.out(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(2, 10, 64))
print('max abs difference:', (att(x) - ref).abs().max().item())""")

md("""### 2.3 Parameter count by hand
For one encoder block with width $D$ and MLP ratio $r$:
- attention: $W_{QKV}$ is $D\\times 3D$, plus $3D$ bias; $W_O$ is $D\\times D$, plus $D$ bias. Total $4D^2+4D$
- MLP: $D\\times rD + rD + rD\\times D + D$. Total $2rD^2 + (r+1)D$
- 2 LayerNorms: $4D$

The cell below checks this formula against PyTorch.""")
code("""def block_params(D, r):
    return (4*D*D + 4*D) + (2*r*D*D + (r+1)*D) + 4*D
D, r = 192, 2
print('formula :', block_params(D, r))
print('pytorch :', count_params(EncoderBlock(D, 3, r)))""")

md("""## 3. Training loop
These are the **standard recipe for training a ViT**. Unlike a CNN, a ViT is sensitive to all of them:
- **AdamW** with weight decay 0.05, but *no decay* on biases, LayerNorm, the CLS token and positional embeddings
- **Linear warm-up, then cosine decay.** Without warm-up, the early attention logits blow up and training diverges.
- **Label smoothing** of 0.1, **gradient clipping** at 1.0, and **stochastic depth** (DropPath)""")
code("""def make_opt(model, lr, wd, epochs, warmup):
    decay    = [p for n, p in model.named_parameters() if p.ndim >= 2 and 'pos' not in n and 'cls' not in n]
    no_decay = [p for n, p in model.named_parameters() if not (p.ndim >= 2 and 'pos' not in n and 'cls' not in n)]
    opt = torch.optim.AdamW([{'params': decay, 'weight_decay': wd}, {'params': no_decay, 'weight_decay': 0}], lr=lr)
    total, warm = epochs * len(train_dl), warmup * len(train_dl)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s+1)/warm if s < warm else 0.5*(1+math.cos(math.pi*(s-warm)/max(1,total-warm))))
    return opt, sched

@torch.no_grad()
def evaluate(model, dl=test_dl):
    model.eval(); correct = n = 0; preds = []
    for x, y in dl:
        x, y = x.to(device), y.to(device)
        with torch.autocast(device, dtype=torch.float16, enabled=device=='cuda'):
            out = model(x)
        preds.append(out.argmax(1).cpu()); correct += (out.argmax(1) == y).sum().item(); n += len(y)
    return correct / n, torch.cat(preds)

def train(model, epochs=50, lr=1e-3, wd=0.05, warmup=5, dl=None, name='model'):
    dl = dl or train_dl
    model.to(device); opt, sched = make_opt(model, lr, wd, epochs, warmup)
    scaler = torch.cuda.amp.GradScaler(enabled=device=='cuda')
    crit = nn.CrossEntropyLoss(label_smoothing=0.1); hist = []
    for ep in range(epochs):
        model.train(); t0 = time.time(); tl = 0
        for x, y in dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device, dtype=torch.float16, enabled=device=='cuda'):
                loss = crit(model(x), y)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step(); tl += loss.item()
        acc, _ = evaluate(model)
        hist.append(dict(epoch=ep+1, loss=tl/len(dl), test_acc=acc))
        print(f'[{name}] epoch {ep+1:3d}  loss {tl/len(dl):.3f}  test acc {acc*100:.2f}%  ({time.time()-t0:.0f}s)')
    return hist""")

md("""## 4. Train the ViT from scratch, then the CNN baseline
Our CPU run in the slides used `dim=128, depth=6, heads=4` for 25 epochs. On a T4 you can afford the bigger config below. Expect roughly **80 to 85%** with 50 epochs, and about 88% or more with 100 epochs plus stronger augmentation (RandAugment or CutMix).""")
code("""EPOCHS = 50
vit = ViT(patch_size=4, dim=192, depth=9, heads=3, mlp_ratio=2, drop_path=0.1)
print(f'ViT params: {count_params(vit)/1e6:.2f}M')
vit_hist = train(vit, epochs=EPOCHS, lr=1e-3, name='ViT')""")
code("""cnn = SmallResNet(w=(64, 128, 256))
print(f'CNN params: {count_params(cnn)/1e6:.2f}M')
cnn_hist = train(cnn, epochs=EPOCHS, lr=2e-3, wd=5e-4, name='CNN')""")
code("""plt.figure(figsize=(7,4))
plt.plot([h['test_acc']*100 for h in vit_hist], label='ViT (from scratch)')
plt.plot([h['test_acc']*100 for h in cnn_hist], label='ResNet-style CNN')
plt.xlabel('epoch'); plt.ylabel('test accuracy (%)'); plt.legend(); plt.grid(alpha=.3); plt.show()""")

md("""## 5. Ablations
Each change below removes one design choice. Train each variant for the same (shorter) number of epochs and compare it with the full model.""")
code("""ABL_EPOCHS = 15
results = {}
results['full']        = train(ViT(dim=192, depth=6, heads=3), epochs=ABL_EPOCHS, name='full')[-1]['test_acc']
results['no pos-emb']  = train(ViT(dim=192, depth=6, heads=3, use_pos=False), epochs=ABL_EPOCHS, name='no-pos')[-1]['test_acc']
results['patch 8x8']   = train(ViT(dim=192, depth=6, heads=3, patch_size=8), epochs=ABL_EPOCHS, name='p8')[-1]['test_acc']
results['patch 2x2']   = train(ViT(dim=192, depth=6, heads=3, patch_size=2), epochs=ABL_EPOCHS, name='p2')[-1]['test_acc']  # 256 tokens: slower
print(results)""")
md("""### 5.1 Data hunger: ViT vs CNN with only 10% of the data
A CNN has locality and translation equivariance **built in**. A ViT has to **learn** them from data, so with little data it falls behind. This is the paper's main finding, at small scale.""")
code("""sub = torch.utils.data.Subset(train_ds, torch.randperm(len(train_ds))[:5000].tolist())
sub_dl = torch.utils.data.DataLoader(sub, batch_size=128, shuffle=True, num_workers=2, drop_last=True)
vit_small = train(ViT(dim=192, depth=6, heads=3), epochs=60, dl=sub_dl, name='ViT-10%')[-1]['test_acc']
cnn_small = train(SmallResNet(), epochs=60, lr=2e-3, wd=5e-4, dl=sub_dl, name='CNN-10%')[-1]['test_acc']
print(f'10% data -> ViT {vit_small*100:.1f}%  vs  CNN {cnn_small*100:.1f}%')""")

md("""## 6. Pre-training fixes data hunger: fine-tune a pretrained ViT
`vit_small_patch16_224` was pre-trained on ImageNet-21k (14M images) and fine-tuned on ImageNet-1k. We upsample CIFAR to 224×224, which gives 14×14 = 196 patches of 16×16, and fine-tune the whole network for 3 epochs.""")
code("""import timm
ft_tf_train = T.Compose([T.Resize(224), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize((0.5,)*3, (0.5,)*3)])
ft_tf_test  = T.Compose([T.Resize(224), T.ToTensor(), T.Normalize((0.5,)*3, (0.5,)*3)])
ft_train = torchvision.datasets.CIFAR10('data', train=True,  transform=ft_tf_train)
ft_test  = torchvision.datasets.CIFAR10('data', train=False, transform=ft_tf_test)
ft_train_dl = torch.utils.data.DataLoader(ft_train, batch_size=64, shuffle=True, num_workers=2, drop_last=True)
ft_test_dl  = torch.utils.data.DataLoader(ft_test,  batch_size=256, num_workers=2)

pre = timm.create_model('vit_small_patch16_224.augreg_in21k_ft_in1k', pretrained=True, num_classes=10).to(device)
opt = torch.optim.AdamW(pre.parameters(), lr=3e-5, weight_decay=0.05)
scaler = torch.cuda.amp.GradScaler(enabled=device=='cuda')
for ep in range(3):
    pre.train()
    for i, (x, y) in enumerate(ft_train_dl):
        x, y = x.to(device), y.to(device)
        with torch.autocast(device, dtype=torch.float16, enabled=device=='cuda'):
            loss = F.cross_entropy(pre(x), y)
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
    acc, _ = evaluate(pre, ft_test_dl); print(f'fine-tune epoch {ep+1}: {acc*100:.2f}%')""")

md("""## 7. Looking inside the model
### 7.1 What does the CLS token attend to? (last-layer attention and attention rollout)
**Attention rollout** (Abnar & Zuidema, 2020) multiplies the attention matrices of all layers. It adds the identity to each one to account for the residual connection, giving $\\tilde A = \\prod_\\ell \\tfrac12(A_\\ell + I)$, and averages over heads. Row 0 of $\\tilde A$ tells us how much information flows from each patch into the CLS token.""")
code("""@torch.no_grad()
def get_attn(model, x):
    model.eval(); model(x.to(device), keep_attn=True)
    return [b.attn.last_attn.cpu() for b in model.blocks]       # list of (B, h, T, T)

def rollout(attns):
    R = torch.eye(attns[0].shape[-1]).unsqueeze(0)
    for A in attns:
        A = A.mean(1); A = 0.5 * A + 0.5 * torch.eye(A.shape[-1]); A = A / A.sum(-1, keepdim=True)
        R = A @ R
    return R[:, 0, 1:]                                           # CLS -> patches

def denorm(x): return (x * torch.tensor(STD).view(3,1,1) + torch.tensor(MEAN).view(3,1,1)).clamp(0,1).permute(1,2,0)

xs, ys = next(iter(test_dl)); xs, ys = xs[:8], ys[:8]
att = get_attn(vit, xs); ro = rollout(att); g = int(ro.shape[1] ** .5)
fig, ax = plt.subplots(2, 8, figsize=(16, 4.4))
for i in range(8):
    ax[0, i].imshow(denorm(xs[i]), interpolation='nearest'); ax[0, i].set_title(CLASSES[ys[i]]); ax[0, i].axis('off')
    ax[1, i].imshow(denorm(xs[i]), interpolation='nearest'); ax[1, i].imshow(ro[i].reshape(g, g), cmap='inferno', alpha=.6, extent=(0,32,32,0), interpolation='nearest'); ax[1, i].axis('off')
plt.suptitle('Attention rollout (CLS → patches)'); plt.show()""")
md("""### 7.2 Learned positional embeddings
The model is never told where the patches are. Even so, the cosine similarity between position embeddings comes out showing the **2-D grid structure** (paper, Fig. 7).""")
code("""P = vit.pos[0, 1:].detach().cpu(); P = F.normalize(P, dim=-1); S = P @ P.T; g = int(len(P) ** .5)
fig, ax = plt.subplots(g, g, figsize=(8, 8))
for i in range(g*g):
    ax[i//g, i%g].imshow(S[i].reshape(g, g), cmap='viridis', vmin=-1, vmax=1); ax[i//g, i%g].axis('off')
plt.suptitle('cos-sim of each position embedding with all others'); plt.show()""")
md("""### 7.3 Mean attention distance per head
For each head we compute the average pixel distance between a query patch and the patches it attends to. In the paper (Fig. 7 right), large pretrained ViTs have a **mix of local and global heads** in early layers and almost all global heads in deep layers. In our small CIFAR model the trend is the same but much weaker: locality is hard to learn from 50k images.""")
code("""def attention_distance(attns, patch=4):
    T = attns[0].shape[-1] - 1; g = int(T ** .5)
    coords = torch.stack(torch.meshgrid(torch.arange(g), torch.arange(g), indexing='ij'), -1).reshape(-1, 2).float() * patch
    D = torch.cdist(coords, coords)                                    # (T, T) pixel distances
    return torch.stack([(A[:, :, 1:, 1:] * D).sum(-1).mean((0, 2)) for A in attns])   # (layers, heads)

xs, _ = next(iter(test_dl)); ad = attention_distance(get_attn(vit, xs[:256]))
plt.figure(figsize=(6, 4))
for h in range(ad.shape[1]): plt.scatter(range(1, ad.shape[0]+1), ad[:, h], label=f'head {h}')
plt.xlabel('layer'); plt.ylabel('mean attention distance (px)'); plt.legend(); plt.grid(alpha=.3); plt.show()""")
md("""### 7.4 Confusion matrix""")
code("""acc, preds = evaluate(vit); y = torch.tensor(test_ds.targets)
cm = torch.zeros(10, 10, dtype=torch.long)
for t, p_ in zip(y, preds): cm[t, p_] += 1
plt.figure(figsize=(6, 5)); plt.imshow(cm, cmap='Blues'); plt.colorbar()
plt.xticks(range(10), CLASSES, rotation=45); plt.yticks(range(10), CLASSES)
for i in range(10):
    for j in range(10): plt.text(j, i, cm[i, j].item(), ha='center', va='center', fontsize=7, color='white' if cm[i,j] > cm.max()/2 else 'black')
plt.title(f'ViT test accuracy {acc*100:.2f}%'); plt.xlabel('predicted'); plt.ylabel('true'); plt.tight_layout(); plt.show()""")
md("""## 8. Takeaways
1. A ViT is a **standard transformer encoder** applied to a sequence of patch embeddings. The only vision-specific parts are the patchify step and the 2-D layout of the position embeddings.
2. **Weak inductive bias**: trained from scratch on small data, it trails a CNN. It needs strong augmentation and regularisation, or pre-training.
3. **Pre-training at scale** reverses this. A pretrained ViT fine-tunes to about 97 to 98% on CIFAR-10 in a few epochs.
4. The model **learns** locality (local heads in early layers, grid-structured position embeddings) and long-range attention in a data-driven way.""")

nb['cells'] = C
nb['metadata'] = {'accelerator': 'GPU', 'colab': {'provenance': []}, 'kernelspec': {'name': 'python3', 'display_name': 'Python 3'}}
nbf.write(nb, 'ViT_from_scratch_CIFAR10.ipynb')
print('written', len(C), 'cells')
