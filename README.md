# Vision Transformers from Scratch on CIFAR-10

Deep Learning class presentation: **Implementation Explanation, Vision Transformers**.
Group: Ammar, Dilawar, Khizar, Farhan.
We built a Vision Transformer (ViT) by hand in PyTorch, trained it on CIFAR-10, compared it with a CNN,
ran ablations, fine-tuned a pretrained ViT, and visualised what the model learned.

| Model | Test accuracy |
|---|---|
| Our ViT from scratch (0.81M params, 25 epochs) | **79.0%** |
| ResNet-style CNN (0.69M params, same recipe) | **90.3%** |
| ViT with only 10% of the data | 56.2% (CNN: 72.7%) |
| Pretrained ViT-Tiny/16, fine-tuned 1 epoch on 10k images | **94.5%** |

The story: a ViT is a plain transformer on image patches. With little data it loses to a CNN
because it must learn the structure a CNN has built in, and pre-training fixes that.

---

## What is in this repository

| File / folder | What it is |
|---|---|
| `ViT_Presentation.pptx` | The 32-slide deck, with speaker notes on every slide |
| `ViT_from_scratch_CIFAR10.ipynb` | Colab/Jupyter notebook: the full project end to end |
| `vit.py` | The ViT (patch embedding, attention, encoder blocks) and the CNN baseline, written by hand |
| `train.py` | Trains one model from scratch (used for every run in the slides) |
| `finetune.py` | Fine-tunes a pretrained ViT-Tiny/16 from `timm` |
| `analysis.py` | Makes all figures in `figs/` from the trained model |
| `demo.py` | Live demo for the presentation: shapes, parameter count, attention check, live predictions |
| `predict.py` | Classify **your own photo** with both models; shows the guesses, the 32x32 view and where the model looked |
| `examples/` | 10 example photos (one per class) from Wikimedia Commons; credits in `examples/CREDITS.md` |
| `prepare_data.py` | Downloads CIFAR-10 and writes `cifar_train.npz` / `cifar_test.npz` |
| `results/` | Each run's numbers (`*.json`) and trained weights (`*.pt`), including the fine-tuned ViT-Tiny (`finetuned_vit_tiny.pt`) |
| `figs/` | The figures used in the slides |
| `log_a.txt`, `log_b.txt`, `log_ft.txt` | Console logs of the real training runs |
| `run_a.sh`, `run_b.sh` | The exact commands used for all training runs |
| `make_notebook.py` | Regenerates the notebook from `vit.py` |
| `smoke_test_notebook.py` | Runs every notebook cell quickly on CPU, to check nothing is broken |
| `build_deck.js` | Regenerates the PowerPoint from `results/` and `figs/` (optional) |

The CIFAR-10 data is **not** in the repo: GitHub refuses files over 100 MB. `prepare_data.py` recreates it in one command.

---

## Just want to look at everything? (no setup)

1. **Slides:** download `ViT_Presentation.pptx` and open it in PowerPoint, Keynote or Google Slides.
   Speaker notes: in PowerPoint, choose **View → Notes**.
2. **Notebook:** click `ViT_from_scratch_CIFAR10.ipynb` on GitHub to read it. To run it on a free GPU,
   open [Google Colab](https://colab.research.google.com), choose **File → Open notebook → GitHub**, paste this repo's URL,
   then **Runtime → Change runtime type → T4 GPU** and **Runtime → Run all**. It takes about 45 to 60 minutes.
3. **Figures:** open any image in `figs/`.
4. **Numbers:** open any file in `results/*.json` (`final_test_acc` and the per-epoch `history`).

---

## Full setup on your laptop

### Step 0: install the tools (once)

| Tool | Why | Get it |
|---|---|---|
| Python 3.10, 3.11 or 3.12 | runs everything | [python.org](https://www.python.org/downloads/) (on Windows, tick **Add Python to PATH**) |
| Git | to clone the repo | [git-scm.com](https://git-scm.com/downloads) |
| VS Code (recommended) | editor with Python and Jupyter support | [code.visualstudio.com](https://code.visualstudio.com/), then install the **Python** and **Jupyter** extensions |
| Node.js 18+ (optional) | only to rebuild the slides | [nodejs.org](https://nodejs.org/) |

### Step 1: clone the repo

```bash
git clone https://github.com/muhammadammar33/DL-presentation.git
cd DL-presentation
```

### Step 2: create a virtual environment

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell refuses to run the script, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then try again.

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

You should now see `(.venv)` at the start of your terminal prompt. Activate it again every time you open a new terminal.

### Step 3: install PyTorch, then the rest

Install PyTorch first, picking the command for your machine from [pytorch.org/get-started](https://pytorch.org/get-started/locally/):

- **No NVIDIA GPU (most laptops), or a Mac:**
  ```bash
  pip install torch torchvision
  ```
  On Windows and Linux, the CPU-only build is smaller: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`
- **NVIDIA GPU:** use the CUDA command from the PyTorch website.

Then install everything else:

```bash
pip install -r requirements.txt
```

Check it worked:

```bash
python -c "import torch, timm; print('torch', torch.__version__, '| GPU:', torch.cuda.is_available())"
```

### Step 4: download the dataset (about 170 MB, once)

```bash
python prepare_data.py
```

This downloads CIFAR-10 from the Hugging Face mirror and writes `cifar_train.npz` (50,000 images) and `cifar_test.npz` (10,000 images).
The official University of Toronto server (`python prepare_data.py --source torchvision`) is often extremely slow, so it is not the default.

### Step 5: check the model code (a few seconds)

```bash
python -c "import torch, torch.nn.functional as F; from vit import *; m = ViT(dim=128, depth=6, heads=4); print('params', count_params(m)); a = MultiHeadSelfAttention(64, 4); x = torch.randn(2, 10, 64); q, k, v = a.qkv(x).reshape(2, 10, 3, 4, 16).permute(2, 0, 3, 1, 4); ref = a.out(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(2, 10, 64)); print('max diff vs PyTorch attention', (a(x) - ref).abs().max().item())"
```

Expected: `params 811146` and a max difference around `1e-7`.

---

## Using everything locally

### Live demo used in the presentation (about 20 seconds)

```bash
python demo.py all
```

Or one part at a time: `python demo.py shapes`, `params`, `check` or `predict`.
`predict` loads the trained ViT, classifies 8 random test images and scores the full test set (79.02%).

### Classify your own photo (live demo, a few seconds)

```bash
python predict.py examples/cat.jpg
```

Use any photo (`python predict.py C:\Users\you\Pictures\my_cat.jpg`) or several at once (`python predict.py examples/*.jpg`).
It prints each model's top-3 guesses and opens a figure (also saved as `predictions.png`).
The models only know the 10 CIFAR-10 classes, so a photo of anything else is forced into one of them.
On the 10 example photos, our from-scratch ViT gets 7 right and the pretrained ViT gets all 10.

### A. Regenerate the figures from the trained model (about 1 minute)

The trained weights are already in `results/`, so no training is needed:

```bash
python analysis.py
```

This rewrites everything in `figs/` and prints the numbers used in Part 4 of the slides
(per-class accuracy, top confusions, attention distances, position-embedding correlation).

### B. Run the notebook

```bash
jupyter notebook ViT_from_scratch_CIFAR10.ipynb
```

Or open it in VS Code and pick the `.venv` interpreter as the kernel.
**On a laptop CPU, lower the training settings first**, or the notebook will take many hours:
in section 4 set `EPOCHS = 5`, and in section 5 set `ABL_EPOCHS = 3`.
The fine-tuning cell (section 6) is also slow on CPU. For full runs, use Colab with a GPU (see above).

### C. Re-train the models yourself (optional, slow on CPU)

Each command writes `results/<name>.json` and `results/<name>.pt`.
These timings are from a 4-core CPU; a laptop will be similar, and a GPU much faster.

```bash
python train.py --name vit_main --epochs 25                                   # ~67 min, 79.0%
python train.py --name cnn_main --model cnn --epochs 25                       # ~90 min, 90.3%
python train.py --name vit_nopos --epochs 10 --no_pos                         # ~28 min
python train.py --name vit_p8 --epochs 10 --patch 8                           # ~7 min
python train.py --name vit_noaug --epochs 10 --no_aug                         # ~26 min
python train.py --name vit_main10 --epochs 10                                 # ~25 min
python train.py --name vit_10pct --epochs 30 --train_frac 0.1 --warmup 3      # ~13 min
python train.py --name cnn_10pct --model cnn --epochs 30 --train_frac 0.1 --warmup 3   # ~18 min
python finetune.py --train_n 10000 --test_n 2000 --epochs 1                   # ~6 min, 94.5%
```

`run_a.sh` and `run_b.sh` contain the same commands as two queues (macOS/Linux: `bash run_a.sh`).
To try something quickly, use fewer epochs: `python train.py --name quick --epochs 2`.
Re-running overwrites the result files in `results/`. Results can differ slightly from ours, because of
random seeds and hardware.

### D. Test the notebook without a GPU (about 5 minutes)

```bash
python smoke_test_notebook.py
```

This runs every notebook cell with tiny settings and prints `ALL CELLS OK` at the end.
It needs the dataset from step 4 and downloads a pretrained model from Hugging Face.

### E. Rebuild the slides (optional)

The finished deck is already in the repo. To regenerate it after changing results or figures:

```bash
npm install pptxgenjs
node build_deck.js
```

Without the separate theme helper this repo does not include, the rebuilt deck uses PowerPoint's
default colours instead of the custom theme. To change only the text, edit `ViT_Presentation.pptx` directly in PowerPoint.

### F. Regenerate the notebook after editing `vit.py` (optional)

```bash
python make_notebook.py
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `python` not found (Windows) | Reinstall Python with **Add Python to PATH** ticked, or use `py` instead of `python` |
| `No module named torch` | The virtual environment is not active: run the activate command from step 2 again |
| `FileNotFoundError: cifar_train.npz` | Run `python prepare_data.py` in the repo folder |
| `No module named pyarrow` | `pip install pyarrow`, then run `python prepare_data.py` again |
| `RuntimeError: File not found or corrupted` | The slow official server timed out: use `python prepare_data.py` (Hugging Face mirror is the default), and delete `data/cifar-10-python.tar.gz` |
| Training is very slow | Normal on CPU: use fewer `--epochs`, or use Colab with a GPU |
| Out of memory | Close other programs; `train.py` keeps the whole dataset in memory (about 0.7 GB) |
| Jupyter can't find the packages | Select the `.venv` interpreter as the notebook kernel (VS Code: top-right kernel picker) |

---

## References

- Dosovitskiy et al. (2021). [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929). ICLR.
- Vaswani et al. (2017). [Attention Is All You Need](https://arxiv.org/abs/1706.03762). NeurIPS.
- Abnar & Zuidema (2020). [Quantifying Attention Flow in Transformers](https://arxiv.org/abs/2005.00928). ACL.
- Krizhevsky (2009). [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html).
