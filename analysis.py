"""Produce figures + summary numbers for the slides from trained checkpoints."""
import json, numpy as np, torch, torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from vit import ViT, SmallResNet

torch.set_num_threads(4)
OUT = 'figs/'
CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)
INK = '#1B2333'; MUTED = '#5A6478'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'text.color': INK, 'axes.labelcolor': INK,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.edgecolor': '#C9CED8'})

d = np.load('cifar_test.npz'); Xraw, y = d['X'], torch.from_numpy(d['y'])
X = (torch.from_numpy(Xraw).permute(0, 3, 1, 2).float() / 255 - MEAN) / STD

vit = ViT(dim=128, depth=6, heads=4); vit.load_state_dict(torch.load('results/vit_main.pt')); vit.eval()
summary = {}

# ---------- 1. patchify illustration ----------
img = Xraw[7]  # pick an image
fig, ax = plt.subplots(figsize=(4, 4))
ax.imshow(img, interpolation='nearest')
for k in range(0, 33, 4):
    ax.axhline(k - .5, color='white', lw=1.2); ax.axvline(k - .5, color='white', lw=1.2)
ax.axis('off'); plt.savefig(OUT + 'patch_grid.png', dpi=260, bbox_inches='tight', pad_inches=0.02); plt.close()
fig, axs = plt.subplots(1, 16, figsize=(12, 1.0))
for i in range(16):
    r, c = divmod(i, 8)
    axs[i].imshow(img[r * 4:(r + 1) * 4, c * 4:(c + 1) * 4], interpolation='nearest'); axs[i].axis('off')
plt.subplots_adjust(wspace=0.15); plt.savefig(OUT + 'patch_seq.png', dpi=260, bbox_inches='tight', pad_inches=0.02); plt.close()
summary['patch_img_class'] = CLASSES[int(y[7])]

# ---------- 2. predictions / per-class ----------
res = {k: json.load(open(f'results/{k}.json')) for k in ['vit_main', 'cnn_main']}
for k, r in res.items():
    p = torch.tensor(r['preds'])
    summary[k + '_per_class'] = [((p == y) & (y == c)).sum().item() / 1000 for c in range(10)]
p = torch.tensor(res['vit_main']['preds'])
cm = torch.zeros(10, 10, dtype=torch.long)
for t, q in zip(y, p): cm[t, q] += 1
blues = LinearSegmentedColormap.from_list('b', ['#F3F8FB', '#9CCBE0', '#0089B5', '#004A66'])
fig, ax = plt.subplots(figsize=(6.2, 5.4))
ax.imshow(cm, cmap=blues)
ax.set_xticks(range(10)); ax.set_xticklabels(CLASSES, rotation=45, ha='right'); ax.set_yticks(range(10)); ax.set_yticklabels(CLASSES)
for i in range(10):
    for j in range(10):
        ax.text(j, i, cm[i, j].item(), ha='center', va='center', fontsize=8, color='white' if cm[i, j] > 450 else INK)
ax.set_xlabel('predicted'); ax.set_ylabel('true')
for s in ax.spines.values(): s.set_visible(False)
plt.tight_layout(); plt.savefig(OUT + 'confusion.png', dpi=260); plt.close()
off = cm.clone(); off.fill_diagonal_(0); top = torch.topk(off.flatten(), 3)
summary['top_confusions'] = [(CLASSES[i // 10], CLASSES[i % 10], v.item()) for v, i in zip(top.values, top.indices)]

# ---------- 3. attention: rollout + per-head ----------
@torch.no_grad()
def attns(x):
    vit(x, keep_attn=True); return [b.attn.last_attn for b in vit.blocks]

def rollout(A):
    R = torch.eye(A[0].shape[-1]).unsqueeze(0)
    for a in A:
        a = a.mean(1); a = 0.5 * a + 0.5 * torch.eye(a.shape[-1]); a = a / a.sum(-1, keepdim=True); R = a @ R
    return R[:, 0, 1:]

correct = (p == y).nonzero().flatten()
pick = []
for c in [0, 1, 2, 3, 4, 7, 8, 9]:
    cand = [i.item() for i in correct if y[i] == c]
    pick.append(cand[3])
A = attns(X[pick]); R = rollout(A)
fig, axs = plt.subplots(2, 8, figsize=(16, 4.3))
for k, i in enumerate(pick):
    axs[0, k].imshow(Xraw[i], interpolation='nearest'); axs[0, k].set_title(CLASSES[int(y[i])], fontsize=13); axs[0, k].axis('off')
    m = R[k].reshape(8, 8); m = (m - m.min()) / (m.max() - m.min())
    axs[1, k].imshow(Xraw[i], interpolation='nearest')
    axs[1, k].imshow(m, cmap='inferno', alpha=.55, extent=(-.5, 31.5, 31.5, -.5), interpolation='nearest'); axs[1, k].axis('off')
plt.tight_layout(); plt.savefig(OUT + 'rollout.png', dpi=260); plt.close()

# per-head CLS attention in first and last layer for one image
k0 = 6
fig, axs = plt.subplots(2, 5, figsize=(10, 4.4))
for row, L in enumerate([0, 5]):
    axs[row, 0].imshow(Xraw[pick[k0]], interpolation='nearest'); axs[row, 0].axis('off')
    axs[row, 0].set_title(f'layer {L + 1}', fontsize=12)
    for h in range(4):
        m = A[L][k0, h, 0, 1:].reshape(8, 8)
        axs[row, h + 1].imshow(m, cmap='inferno', interpolation='nearest'); axs[row, h + 1].axis('off')
        axs[row, h + 1].set_title(f'head {h + 1}', fontsize=12)
plt.tight_layout(); plt.savefig(OUT + 'heads.png', dpi=260); plt.close()

# ---------- 4. attention distance ----------
with torch.no_grad():
    A = attns(X[:500])
g = 8
coords = torch.stack(torch.meshgrid(torch.arange(g), torch.arange(g), indexing='ij'), -1).reshape(-1, 2).float() * 4
D = torch.cdist(coords, coords)
ad = torch.stack([(a[:, :, 1:, 1:] / a[:, :, 1:, 1:].sum(-1, keepdim=True) * D).sum(-1).mean((0, 2)) for a in A])
summary['attn_distance'] = ad.tolist()  # layers x heads, in pixels
# uniform attention baseline
summary['attn_distance_uniform'] = D.mean().item()

# ---------- 5. position embedding similarity ----------
P = F.normalize(vit.pos[0, 1:].detach(), dim=-1); S = P @ P.T
fig, axs = plt.subplots(8, 8, figsize=(6, 6))
for i in range(64):
    axs[i // 8, i % 8].imshow(S[i].reshape(8, 8), cmap='viridis', vmin=-1, vmax=1); axs[i // 8, i % 8].axis('off')
plt.subplots_adjust(wspace=0.08, hspace=0.08); plt.savefig(OUT + 'posemb.png', dpi=260, bbox_inches='tight', pad_inches=0.02); plt.close()
# how 'spatial' is it: corr between cos-sim and negative distance
summary['posemb_corr_with_distance'] = float(np.corrcoef(S[~torch.eye(64, dtype=bool)].numpy(), D[~torch.eye(64, dtype=bool)].numpy())[0, 1])

# ---------- 6. patch-embedding filters: PCA ----------
W = vit.patch.proj.weight.detach().reshape(128, -1)  # (D, 3*4*4)
W = W - W.mean(0)
U, Sv, Vt = torch.linalg.svd(W, full_matrices=False)
fig, axs = plt.subplots(2, 8, figsize=(8, 2.2))
for i in range(16):
    f = Vt[i].reshape(3, 4, 4).permute(1, 2, 0); f = (f - f.min()) / (f.max() - f.min())
    axs[i // 8, i % 8].imshow(f, interpolation='nearest'); axs[i // 8, i % 8].axis('off')
plt.subplots_adjust(wspace=0.1, hspace=0.1); plt.savefig(OUT + 'filters.png', dpi=260, bbox_inches='tight', pad_inches=0.02); plt.close()

# ---------- 7. misclassified ----------
wrong = (p != y).nonzero().flatten()[:8]
fig, axs = plt.subplots(1, 8, figsize=(16, 2.6))
for k, i in enumerate(wrong):
    axs[k].imshow(Xraw[i], interpolation='nearest'); axs[k].axis('off')
    axs[k].set_title(f'true: {CLASSES[int(y[i])]}\npred: {CLASSES[int(p[i])]}', fontsize=12)
plt.tight_layout(); plt.savefig(OUT + 'wrong.png', dpi=260); plt.close()

json.dump(summary, open('results/summary.json', 'w'), indent=1)
print(json.dumps(summary, indent=1))
