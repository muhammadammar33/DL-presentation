const pptxgen = require('pptxgenjs');
const fs = require('fs');
const path = require('path');
// Optional helper that writes the deck's colour theme into the .pptx. It is not part of this repo;
// without it the deck still builds, but PowerPoint shows its default Office colours.
// The finished deck (ViT_Presentation.pptx) is already in the repo, so rebuilding is optional.
let applyTheme = null;
try {
  if (process.env.SKILL_DIR) ({ applyTheme } = require(path.join(process.env.SKILL_DIR, 'scripts/apply_theme.js')));
} catch (e) { applyTheme = null; }

const R = (n) => JSON.parse(fs.readFileSync(`results/${n}.json`));
const S = JSON.parse(fs.readFileSync('results/summary.json'));
const FT = fs.existsSync('results/finetune.json') ? R('finetune') : null;
const pct = (v, d = 1) => (v * 100).toFixed(d) + '%';
const vm = R('vit_main'), cm = R('cnn_main');
const abl = { full: R('vit_main10'), nopos: R('vit_nopos'), p8: R('vit_p8'), noaug: R('vit_noaug') };
const v10 = R('vit_10pct'), c10 = R('cnn_10pct');
const bestOf = (r) => Math.max(...r.history.map((h) => h.test_acc));

const THEME = {
  name: 'ViT Deck',
  headFontFace: 'Cambria',
  bodyFontFace: 'Calibri',
  colors: {
    dk1: '1B2333', lt1: 'FFFFFF', dk2: '101828', lt2: 'F2F4F8',
    accent1: 'D9661A', accent2: '0089B5', accent3: '7A5CC7', accent4: '5A6478',
    accent5: 'F6E3D3', accent6: 'D8ECF4', hlink: '0089B5', folHlink: '7A5CC7',
  },
};
const HEX = THEME.colors;

const pres = new pptxgen();
pres.layout = 'LAYOUT_WIDE'; // 13.33 x 7.5
pres.title = 'Vision Transformers: from scratch on CIFAR-10';
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;

// ---------------- click-by-click animation tagging ----------------
// step(n) marks every element added after it as appearing on click n. Elements added while
// ANIM = 0 (titles, footer, anything before the first step) are visible when the slide opens.
// add_animations.py turns the a<n>_ object names into PowerPoint entrance animations.
let ANIM = 0, OBJ = 0;
const ALL_SLIDES = [];
function step(n) { ANIM = n; }
const _addSlide = pres.addSlide.bind(pres);
pres.addSlide = (opts) => {
  const s = _addSlide(opts);
  ANIM = 0; s._maxStep = 0; s._notes = '';
  ALL_SLIDES.push(s);
  const tag = (o) => {
    if (!o || o.placeholder || ANIM === 0) return o;
    o.objectName = `a${ANIM}_${++OBJ}`; s._maxStep = Math.max(s._maxStep, ANIM); return o;
  };
  const wrap = (name, optIndex) => {
    const orig = s[name].bind(s);
    s[name] = (...args) => { if (args.length <= optIndex) args[optIndex] = {}; tag(args[optIndex]); return orig(...args); };
  };
  wrap('addText', 1); wrap('addShape', 1); wrap('addImage', 0); wrap('addChart', 2); wrap('addTable', 1);
  s._origNotes = s.addNotes.bind(s);
  s.addNotes = (t) => { s._notes = t; };
  return s;
};
function finishNotes() {
  for (const s of ALL_SLIDES) {
    const n = s._maxStep;
    const hint = n ? `[${n} click${n > 1 ? 's' : ''} to reveal this slide, then 1 more to move on]\n` : '';
    if (hint || s._notes) s._origNotes(hint + (s._notes || ''));
  }
}
const ORANGE = C.accent1, BLUE = C.accent2, PURPLE = C.accent3, MUTED = C.accent4;
const W = 13.333;

// ---------------- layouts ----------------
pres.defineSlideMaster({
  title: 'TITLE_DARK', background: { color: C.text2 },
  objects: [
    { placeholder: { options: { name: 'title', type: 'title', x: 0.8, y: 2.0, w: 11.7, h: 1.9, fontSize: 44, bold: true, color: C.background1, valign: 'bottom', fontFace: THEME.headFontFace, align: 'left' }, text: '' } },
    { placeholder: { options: { name: 'body', type: 'body', x: 0.8, y: 4.1, w: 11.7, h: 1.6, fontSize: 20, color: 'C9CED8', valign: 'top' }, text: '' } },
  ],
});
pres.defineSlideMaster({
  title: 'SECTION_DARK', background: { color: C.text2 },
  objects: [
    { placeholder: { options: { name: 'title', type: 'title', x: 0.8, y: 2.9, w: 11.7, h: 1.2, fontSize: 40, bold: true, color: C.background1, valign: 'top', fontFace: THEME.headFontFace, align: 'left' }, text: '' } },
    { placeholder: { options: { name: 'body', type: 'body', x: 0.8, y: 4.2, w: 11.7, h: 1.2, fontSize: 18, color: 'C9CED8', valign: 'top' }, text: '' } },
  ],
  slideNumber: { x: 12.2, y: 6.95, w: 0.8, h: 0.3, fontSize: 10, color: '8D96A8', align: 'right' },
});
pres.defineSlideMaster({
  title: 'CONTENT', background: { color: C.background1 },
  objects: [
    { placeholder: { options: { name: 'title', type: 'title', x: 0.6, y: 0.35, w: 12.1, h: 0.85, fontSize: 32, bold: true, color: C.text1, valign: 'middle', fontFace: THEME.headFontFace, margin: 0, align: 'left' }, text: '' } },
    { text: { text: 'Vision Transformers · from scratch on CIFAR-10', options: { x: 0.6, y: 7.0, w: 8, h: 0.3, fontSize: 10, color: MUTED, margin: 0 } } },
  ],
  slideNumber: { x: 12.2, y: 7.0, w: 0.5, h: 0.3, fontSize: 10, color: MUTED, align: 'right' },
});

// ---------------- helpers ----------------
// the group, with student IDs, in presenting order (title and closing slides)
const GROUP = [['Muhammad Ammar Cheema', '2604920'], ['Dilawar Shah', '2604915'], ['Syed Khizar Hussain Kazmi', '2604927'], ['Muhammad Farhan Khan', '2604916']];
function groupNames(s) {
  GROUP.forEach((g, i) => {
    const x = 0.8 + i * 2.95;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 5.75, w: 2.8, h: 0.95, rectRadius: 0.08, fill: { color: '1E2A3E' }, line: { type: 'none' } });
    s.addText([{ text: g[0], options: { bold: true, color: 'FFFFFF', fontSize: 13, breakLine: true } }, { text: g[1], options: { color: 'AEB7C6', fontSize: 12 } }],
      { x: x + 0.15, y: 5.8, w: 2.55, h: 0.85, valign: 'middle', isTextBox: true, margin: 0 });
  });
}

let section = '';
function content(title, notes) {
  const s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: section });
  s.addText(title, { placeholder: 'title' });
  // footer progress indicator: which of the 4 parts we are in
  const m = /^Part (\d): (.*)$/.exec(section);
  if (m) {
    const idx = Number(m[1]);
    s.addText(m[2], { x: 8.6, y: 7.0, w: 2.25, h: 0.3, fontSize: 10, color: MUTED, align: 'right', isTextBox: true, margin: 0 });
    for (let i = 0; i < 4; i++) {
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 11.0 + i * 0.26, y: 7.07, w: 0.17, h: 0.17, rectRadius: 0.03,
        fill: { color: i + 1 === idx ? HEX.accent1 : 'D5D9E0' }, line: { type: 'none' } });
    }
  }
  if (notes) s.addNotes(notes);
  return s;
}
function sectionSlide(num, title, sub, presenter, notes) {
  section = title;
  pres.addSection({ title });
  const s = pres.addSlide({ masterName: 'SECTION_DARK', sectionTitle: title });
  // motif: row of patch tiles, the active one highlighted
  for (let i = 0; i < 4; i++) {
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.8 + i * 0.75, y: 1.7, w: 0.6, h: 0.6, rectRadius: 0.08,
      fill: { color: i === num - 1 ? ORANGE : '2A3448' }, line: { type: 'none' }, objectName: `tile${i}` });
    s.addText(String(i + 1), { x: 0.8 + i * 0.75, y: 1.7, w: 0.6, h: 0.6, align: 'center', valign: 'middle', fontSize: 18, bold: true,
      color: C.background1, isTextBox: true, margin: 0 });
  }
  s.addText(title, { placeholder: 'title' });
  s.addText([{ text: sub, options: { breakLine: true } }, { text: presenter, options: { color: ORANGE, bold: true } }], { placeholder: 'body' });
  if (notes) s.addNotes(notes);
  return s;
}
// A straight connector from (x1, y1) to (x2, y2). PowerPoint needs a positive width and height,
// with flipH / flipV for lines that run left or up; a negative height draws the wrong line.
function connect(s, x1, y1, x2, y2, line) {
  s.addShape(pres.shapes.LINE, { x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1), h: Math.abs(y2 - y1),
    flipH: x2 < x1, flipV: y2 < y1, line });
}
function card(s, x, y, w, h, head, body, opts = {}) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.1, fill: { color: opts.fill || C.background2 }, line: { type: 'none' }, objectName: 'card' });
  const runs = [];
  if (head) runs.push({ text: head, options: { bold: true, fontSize: opts.headSize || 18, color: opts.headColor || C.text1, breakLine: true, paraSpaceAfter: 6 } });
  (Array.isArray(body) ? body : [body]).forEach((b, i, arr) => {
    const isBullet = opts.bullets !== false && Array.isArray(body);
    runs.push({ text: b, options: { fontSize: opts.size || 16, color: C.text1, bullet: isBullet ? { indent: 14 } : false, breakLine: i < arr.length - 1, paraSpaceAfter: 5 } });
  });
  s.addText(runs, { x: x + 0.2, y: y + 0.15, w: w - 0.4, h: h - 0.3, valign: 'top', isTextBox: true, margin: 0 });
}
function tile(s, x, y, size, color, label, labelColor) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: size, h: size, rectRadius: 0.06, fill: { color }, line: { type: 'none' } });
  if (label !== undefined) s.addText(String(label), { x, y, w: size, h: size, align: 'center', valign: 'middle', fontSize: 14, bold: true, color: labelColor || C.background1, isTextBox: true, margin: 0 });
}
function code(s, x, y, w, h, src, size = 12) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: C.text2 }, line: { type: 'none' }, objectName: 'code' });
  const lines = src.split('\n');
  s.addText(lines.map((l, i) => ({ text: l || ' ', options: { breakLine: i < lines.length - 1, color: l.trim().startsWith('#') ? '8FA3BF' : 'E8ECF3' } })),
    { x: x + 0.2, y: y + 0.12, w: w - 0.4, h: h - 0.24, fontFace: 'Courier New', fontSize: size, valign: 'top', isTextBox: true, margin: 0 });
}
function eq(s, x, y, w, h, text, size = 22) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.1, fill: { color: C.accent5 }, line: { type: 'none' }, objectName: 'equation' });
  s.addText(text, { x, y, w, h, align: 'center', valign: 'middle', fontFace: 'Cambria', italic: true, fontSize: size, color: C.text1, isTextBox: true, margin: 0.1 });
}
function stat(s, x, y, w, big, label, color) {
  s.addText(big, { x, y, w, h: 0.95, fontSize: 54, bold: true, color, fontFace: 'Cambria', isTextBox: true, margin: 0 });
  s.addText(label, { x, y: y + 0.95, w, h: 0.6, fontSize: 14, color: MUTED, isTextBox: true, margin: 0, valign: 'top' });
}
function txt(s, x, y, w, h, text, o = {}) {
  s.addText(text, Object.assign({ x, y, w, h, fontSize: 14, color: C.text1, isTextBox: true, margin: 0, valign: 'top' }, o));
}
const chartBase = (extra) => Object.assign({
  catAxisLabelColor: '5A6478', valAxisLabelColor: '5A6478', catAxisLabelFontSize: 12, valAxisLabelFontSize: 12,
  catAxisLabelFontFace: '+mn-lt', valAxisLabelFontFace: '+mn-lt', dataLabelFontFace: '+mn-lt', legendFontFace: '+mn-lt', titleFontFace: '+mn-lt',
  valGridLine: { color: 'E3E6EC', size: 0.75 }, catGridLine: { style: 'none' }, catAxisLineShow: true, valAxisLineShow: false,
  dataLabelColor: '1B2333', dataLabelFontSize: 11, legendFontSize: 12, legendColor: '1B2333', titleColor: '1B2333', titleFontSize: 14,
}, extra);

// ======================================================================
// TITLE
// ======================================================================
pres.addSection({ title: 'Opening' }); section = 'Opening';
{
  const s = pres.addSlide({ masterName: 'TITLE_DARK', sectionTitle: section });
  // motif: an 8x2 strip of patch tiles from the real image
  s.addImage({ path: 'figs/patch_seq.png', x: 0.8, y: 0.9, w: 8.4, h: 0.62, transparency: 0 });
  s.addText('Vision Transformers: An Image is a Sequence of Patches', { placeholder: 'title' });
  s.addText([
    { text: 'Building a ViT from scratch in PyTorch and training it on CIFAR-10', options: { breakLine: true } },
    { text: 'Implementation Explanation · Deep Learning', options: { color: ORANGE, fontSize: 16 } },
  ], { placeholder: 'body' });
  groupNames(s);
  s.addNotes(`[Ammar] Assalam o alaikum. Our group took the implementation track for vision transformers. We did not just call a library: we wrote every part of a Vision Transformer ourselves, trained it on CIFAR-10, compared it with a CNN, ran ablations and looked inside the trained model. The strip at the top is a real CIFAR-10 image (a ${S.patch_img_class}) cut into 4x4-pixel patches. That is literally what our model sees: a sequence of patches, the same way a language model sees a sequence of words.`);
}
{
  const s = content('Roadmap: four parts, four presenters', 'Quick roadmap. Each member presents one part, about 7 minutes each, with questions at the end.');
  const parts = [
    ['1', 'Architecture', 'Patches → tokens → attention → encoder. Every equation and tensor shape.', 'Muhammad Ammar Cheema'],
    ['2', 'Implementation', 'Our from-scratch PyTorch code, how we verified it, and the training recipe.', 'Dilawar Shah'],
    ['3', 'Experiments', 'ViT vs CNN, ablations, data hunger, pre-training.', 'Syed Khizar Hussain Kazmi'],
    ['4', 'Inside the model', 'Attention maps, attention distance, learned positions, failures, what came after ViT.', 'Muhammad Farhan Khan'],
  ];
  parts.forEach((p, i) => {
    step(i + 1);
    const x = 0.6 + i * 3.08;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.7, w: 2.85, h: 4.6, rectRadius: 0.12, fill: { color: C.background2 }, line: { type: 'none' } });
    tile(s, x + 0.25, 1.95, 0.7, i === 0 ? ORANGE : i === 1 ? BLUE : i === 2 ? PURPLE : C.text2, p[0]);
    txt(s, x + 0.25, 2.9, 2.4, 0.5, p[1], { fontSize: 20, bold: true, fontFace: 'Cambria' });
    txt(s, x + 0.25, 3.45, 2.4, 2.2, p[2], { fontSize: 17 });
    txt(s, x + 0.25, 5.55, 2.45, 0.6, p[3], { fontSize: 13, color: MUTED, bold: true, valign: 'bottom' });
  });
}

// ======================================================================
// PART 1: ARCHITECTURE
// ======================================================================
sectionSlide(1, 'Part 1: The Architecture', 'From pixels to tokens to attention', 'Muhammad Ammar Cheema',
  '[Ammar] I will explain the architecture exactly as we implemented it, with real tensor shapes from our model.');
{
  const s = content('Why put a transformer on images at all?',
    `A CNN builds in two strong assumptions (inductive biases). Locality: a 3x3 kernel only looks at neighbours. Translation equivariance: the same kernel slides everywhere. These priors are why CNNs learn well from small data. But a pixel at layer L can only see a window that grows by about 2 pixels per layer, so long-range interactions arrive late.
A transformer makes almost no assumptions: from layer 1, every token can attend to every other token. Transformers already dominated NLP and scale very well with data and compute. Dosovitskiy et al. (ICLR 2021) asked: can we use a standard transformer, with as little vision-specific design as possible, directly on images? The answer: yes, but only with a lot of data. With JFT-300M pre-training, ViT-H/14 reached 88.55% top-1 on ImageNet. Trained on ImageNet-1k alone, it is worse than a ResNet. Keep this trade-off in mind: we reproduce it in Part 3.`);
  step(1);
  card(s, 0.6, 1.55, 5.9, 3.9, 'CNN: priors built in', [
    'Locality: 3×3 kernels see only neighbours',
    'Translation equivariance: shared weights slide everywhere',
    'Receptive field grows slowly, layer by layer',
    'Data-efficient, but the priors are fixed by design',
  ], { headColor: BLUE, size: 18, headSize: 21 });
  step(2);
  card(s, 6.8, 1.55, 5.9, 3.9, 'Transformer: priors learned', [
    'Global from layer 1: every token attends to every token',
    'Weights depend on the input (attention is dynamic)',
    'Almost no image-specific assumptions',
    'Scales with data and compute, but is data-hungry',
  ], { headColor: ORANGE, size: 18, headSize: 21 });
  step(3);
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y: 5.7, w: 12.1, h: 1.0, rectRadius: 0.1, fill: { color: C.accent5 }, line: { type: 'none' } });
  txt(s, 0.85, 5.8, 11.6, 0.8, [
    { text: 'Dosovitskiy et al., ICLR 2021: ', options: { bold: true } },
    { text: 'a plain transformer encoder with no convolutions in the backbone reaches 88.55% top-1 on ImageNet, but only after pre-training on JFT-300M (300M images).' },
  ], { fontSize: 15, valign: 'middle' });
}
{
  const s = content('Step 1: turn the image into a sequence of patches',
    `This is the only real "vision" idea in ViT. Cut an H x W image into non-overlapping P x P patches. The number of tokens is N = HW / P^2. For ImageNet, 224/16 gives 14x14 = 196 tokens. For our CIFAR-10 model, a 32x32 image with 4x4 patches gives 8x8 = 64 tokens.
Each patch is flattened to a vector of P^2 * C numbers (4*4*3 = 48 for us) and multiplied by a learned matrix E of size 48 x D. That projection is the "word embedding" of vision.
Why not use pixels as tokens? Attention costs O(N^2): 1024 pixels would mean about a million attention scores per head per layer, and 224x224 ImageNet images would mean 2.5 billion. Patches are the trick that makes attention affordable.
Patch size is a trade-off: smaller patches give more tokens, finer detail and quadratically more compute. We measure this in the ablation.`);
  step(1);
  s.addImage({ path: 'figs/patch_grid.png', x: 0.6, y: 1.55, w: 3.6, h: 3.6 });
  txt(s, 0.6, 5.25, 3.6, 0.5, `32×32 image → 8×8 grid of 4×4 patches`, { fontSize: 12, color: MUTED, align: 'center' });
  step(2);
  s.addImage({ path: 'figs/patch_seq.png', x: 4.6, y: 1.65, w: 8.1, h: 0.6 });
  txt(s, 4.6, 2.3, 8.1, 0.3, 'flattened in raster order: the first 16 of 64 tokens', { fontSize: 12, color: MUTED });
  step(3);
  eq(s, 4.6, 2.85, 8.1, 0.85, 'N = HW / P²  →  (32 × 32) / 4² = 64 tokens', 24);
  step(4);
  const rows = [
    [{ text: '', options: { bold: true } }, { text: 'ImageNet ViT-B/16', options: { bold: true } }, { text: 'Our CIFAR ViT', options: { bold: true } }],
    ['Image', '224 × 224 × 3', '32 × 32 × 3'],
    ['Patch P', '16 × 16', '4 × 4'],
    ['Tokens N', '14 × 14 = 196', '8 × 8 = 64'],
    ['Patch vector P²C', '768', '48'],
    ['Width D', '768', '128'],
  ];
  s.addTable(rows, { x: 4.6, y: 3.95, w: 8.1, colW: [2.5, 2.8, 2.8], fontSize: 14, color: '1B2333', fontFace: 'Calibri',
    border: { type: 'solid', pt: 0.75, color: 'E3E6EC' }, fill: { color: 'FFFFFF' }, rowH: 0.42 });
}
{
  const s = content('Step 2: patch embedding is a strided convolution',
    `In code, "flatten each patch and multiply by E" is exactly a Conv2d with kernel size equal to stride equal to P. The windows don't overlap, so each output position is one patch dotted with D filters. It is the same maths, just computed efficiently.
Follow the shapes on the right: batch x 3 x 32 x 32 becomes batch x 128 x 8 x 8, then we flatten the spatial grid and transpose to get batch x 64 tokens x 128 features. From this point on, the model has no idea it is looking at an image. It is just a set of 64 vectors.
A common question: is this "cheating" because it uses a convolution? No. It is one linear layer applied per patch, with no overlap and no hierarchy. This layer is the only place where the 2-D structure enters.`);
  step(1);
  code(s, 0.6, 1.55, 6.4, 3.6, `class PatchEmbedding(nn.Module):
    def __init__(self, img=32, P=4, C=3, D=128):
        super().__init__()
        self.num_patches = (img // P) ** 2
        # kernel = stride = P == linear map/patch
        self.proj = nn.Conv2d(C, D, P, stride=P)

    def forward(self, x):       # (B, 3, 32, 32)
        x = self.proj(x)        # (B, 128, 8, 8)
        x = x.flatten(2)        # (B, 128, 64)
        return x.transpose(1, 2)  # (B, 64, 128)`, 13);
  const steps = [['(B, 3, 32, 32)', 'image'], ['(B, 128, 8, 8)', 'Conv2d k=4, s=4'], ['(B, 128, 64)', 'flatten(2)'], ['(B, 64, 128)', 'transpose: tokens × features']];
  steps.forEach((st, i) => {
    step(i + 2);
    const y = 1.55 + i * 0.95;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 7.4, y, w: 2.6, h: 0.7, rectRadius: 0.08, fill: { color: i === 3 ? ORANGE : C.background2 }, line: { type: 'none' } });
    txt(s, 7.4, y, 2.6, 0.7, st[0], { fontFace: 'Courier New', fontSize: 14, bold: true, align: 'center', valign: 'middle', color: i === 3 ? C.background1 : C.text1 });
    txt(s, 10.2, y, 2.5, 0.7, st[1], { fontSize: 14, color: MUTED, valign: 'middle' });
  });
  step(6);
  card(s, 0.6, 5.5, 12.1, 0.9, null, [
    'Parameters: 48 × 128 weights + 128 biases = 6,272. That is the entire "vision-specific" front end of the network.',
  ], { bullets: false, size: 17 });
}
{
  const s = content('Step 3: CLS token and learned position embeddings',
    `Two additions before the encoder.
1) The CLS token: a learnable vector we prepend to the sequence, borrowed from BERT. It doesn't belong to any patch. Through attention it gathers information from all patches, and its final state is the only thing the classifier reads. We now have 65 tokens.
2) Position embeddings: attention is permutation-invariant. If you shuffle the patches, the output for each token is just shuffled the same way. Without positions, the model can't tell top-left from bottom-right. So we add a learned vector for each of the 65 positions. These are learned 1-D embeddings, and they are never told about the 2-D grid. In Part 4 we show that they learn the grid structure by themselves.
The equation is equation 1 of the paper. Parameters: CLS has 128, positions have 65 x 128 = 8,320.`);
  step(1);
  eq(s, 0.6, 1.5, 12.1, 0.9, 'z₀ = [ x_cls ;  x¹E ;  x²E ;  … ;  x⁶⁴E ]  +  E_pos        E ∈ ℝ^(48×128),  E_pos ∈ ℝ^(65×128)', 22);
  // token row diagram
  const y0 = 2.85;
  step(2);
  tile(s, 0.6, y0, 0.75, PURPLE, 'CLS');
  for (let i = 0; i < 9; i++) tile(s, 1.5 + i * 0.9, y0, 0.75, i === 8 ? 'C9CED8' : ORANGE, i === 8 ? '…' : `x${i + 1}`, i === 8 ? C.text1 : undefined);
  txt(s, 1.5 + 9 * 0.9, y0, 3.0, 0.75, '65 tokens × 128', { fontSize: 16, bold: true, valign: 'middle' });
  step(3);
  txt(s, 0.6, y0 + 0.85, 0.75, 0.4, '+', { fontSize: 20, bold: true, align: 'center' });
  for (let i = 0; i < 9; i++) txt(s, 1.5 + i * 0.9, y0 + 0.85, 0.75, 0.4, '+', { fontSize: 20, bold: true, align: 'center' });
  tile(s, 0.6, y0 + 1.3, 0.75, '5A6478', 'p0');
  for (let i = 0; i < 9; i++) tile(s, 1.5 + i * 0.9, y0 + 1.3, 0.75, i === 8 ? 'C9CED8' : '5A6478', i === 8 ? '…' : `p${i + 1}`, i === 8 ? C.text1 : undefined);
  txt(s, 1.5 + 9 * 0.9, y0 + 1.3, 3.0, 0.75, 'learned positions', { fontSize: 16, color: MUTED, valign: 'middle' });
  step(4);
  card(s, 0.6, 5.15, 5.9, 1.55, 'CLS token', 'A learnable "summary" slot, as in BERT. It attends to all patches, and its final state feeds the classifier head.', { bullets: false });
  card(s, 6.8, 5.15, 5.9, 1.55, 'Why positions are needed', 'Self-attention is permutation-invariant: shuffle the patches and the outputs only shuffle the same way. Positions break that symmetry.', { bullets: false });
}
{
  const s = content('Step 4: scaled dot-product self-attention',
    `The core operation. Each token x_i produces three vectors through learned matrices: a query q_i (what am I looking for), a key k_i (what do I contain) and a value v_i (what I pass on if selected).
Score between token i and j = q_i · k_j. Stacking all tokens gives Q K-transpose, a 65 x 65 matrix for us, one per head. Softmax over each row turns the scores into weights that sum to 1. The output for token i is a weighted average of all value vectors.
Why divide by sqrt(d_k)? If q and k have unit-variance entries, their dot product has variance d_k. Large logits push the softmax into saturation, where gradients vanish. Dividing by sqrt(d_k) brings the variance back to 1.
Cost: computing QK^T and multiplying by V is O(T^2 d). That is why patch size matters so much: halving P makes T four times bigger and attention sixteen times more expensive.
Everything on this slide is in our forward() code: 6 lines of PyTorch.`);
  step(1);
  eq(s, 0.6, 1.5, 7.2, 1.0, 'Attention(Q, K, V) = softmax( QKᵀ / √dₖ ) V', 26);
  step(2);
  txt(s, 0.6, 2.7, 7.2, 2.5, [
    { text: 'Q = XW_Q,  K = XW_K,  V = XW_V', options: { bold: true, breakLine: true, paraSpaceAfter: 8 } },
    { text: 'QKᵀ is a 65 × 65 matrix: score of every token pair', options: { bullet: true, breakLine: true, paraSpaceAfter: 5 } },
    { text: 'Each row is softmaxed into weights that sum to 1', options: { bullet: true, breakLine: true, paraSpaceAfter: 5 } },
    { text: 'Output token i is a weighted mix of all value vectors', options: { bullet: true, breakLine: true, paraSpaceAfter: 5 } },
    { text: '√dₖ: q·k has variance dₖ; scaling keeps softmax out of saturation', options: { bullet: true, breakLine: true, paraSpaceAfter: 5 } },
    { text: 'Cost O(T²·d): quadratic in the number of tokens', options: { bullet: true } },
  ], { fontSize: 15 });
  step(3);
  code(s, 8.1, 1.5, 4.6, 3.75, `# x: (B, T=65, D=128), h=4, dk=32
qkv = self.qkv(x)          # (B,65,384)
qkv = qkv.reshape(B,T,3,h,dk)
qkv = qkv.permute(2,0,3,1,4)
q, k, v = qkv               # (B,4,65,32)

s = q @ k.transpose(-2,-1)  # (B,4,65,65)
s = s / math.sqrt(dk)
a = s.softmax(dim=-1)
y = a @ v                   # (B,4,65,32)`, 12);
  step(4);
  card(s, 0.6, 5.5, 12.1, 1.2, null, [
    'Dimensions in our model: T = 65 tokens, D = 128, h = 4 heads, dₖ = D/h = 32. Each layer has 4 × 65 × 65 = 16,900 attention weights per image.',
  ], { bullets: false, size: 15 });
}
{
  const s = content('Step 5: multi-head attention',
    `One attention head computes one weighted average, so it can only focus on one kind of relationship at a time. Multi-head attention runs h heads in parallel. Each head works in its own smaller dk = D/h subspace, with its own Q, K and V projections. The outputs are concatenated and mixed by an output matrix W_O.
The total cost is the same as one big head, because the dimension is split, not multiplied.
Implementation detail: we don't create 3h separate linear layers. One Linear(D, 3D) computes Q, K and V for all heads in a single matrix multiply, and then a reshape splits them. That is the qkv line in our code.
Parameter count for attention: W_QKV is 128x384 plus a 384 bias; W_O is 128x128 plus a 128 bias. Total 4D^2 + 4D = 66,048 per layer.
In Part 4 we will see that different heads really do specialise: some local, some global.`);
  step(1);
  eq(s, 0.6, 1.5, 12.1, 0.9, 'MSA(X) = Concat( head₁, …, head_h ) W_O ,     headᵢ = Attention( XW_Qⁱ, XW_Kⁱ, XW_Vⁱ )', 22);
  // diagram: X -> 4 heads -> concat -> W_O
  const y = 2.8;
  step(2);
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y: y + 0.9, w: 1.7, h: 1.2, rectRadius: 0.08, fill: { color: C.text2 }, line: { type: 'none' } });
  txt(s, 0.6, y + 0.9, 1.7, 1.2, 'X\n65 × 128', { color: C.background1, bold: true, align: 'center', valign: 'middle', fontSize: 15 });
  for (let i = 0; i < 4; i++) {
    const hy = y + i * 0.75;
    connect(s, 2.3, y + 1.5, 3.1, hy + 0.3, { color: 'C9CED8', width: 1.5 });
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 3.1, y: hy, w: 2.9, h: 0.6, rectRadius: 0.08, fill: { color: [ORANGE, BLUE, PURPLE, '5A6478'][i] }, line: { type: 'none' } });
    txt(s, 3.1, hy, 2.9, 0.6, `head ${i + 1}: 65 × 32`, { color: C.background1, bold: true, align: 'center', valign: 'middle', fontSize: 14 });
    connect(s, 6.0, hy + 0.3, 6.7, y + 1.5, { color: 'C9CED8', width: 1.5, endArrowType: 'triangle' });
  }
  step(3);
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 6.7, y: y + 0.9, w: 2.0, h: 1.2, rectRadius: 0.08, fill: { color: C.background2 }, line: { type: 'none' } });
  txt(s, 6.7, y + 0.9, 2.0, 1.2, 'concat\n65 × 128', { bold: true, align: 'center', valign: 'middle', fontSize: 15 });
  s.addShape(pres.shapes.LINE, { x: 8.7, y: y + 1.5, w: 0.5, h: 0, line: { color: 'C9CED8', width: 1.5, endArrowType: 'triangle' } });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 9.2, y: y + 0.9, w: 1.4, h: 1.2, rectRadius: 0.08, fill: { color: C.background2 }, line: { type: 'none' } });
  txt(s, 9.2, y + 0.9, 1.4, 1.2, '× W_O', { bold: true, align: 'center', valign: 'middle', fontSize: 15 });
  s.addShape(pres.shapes.LINE, { x: 10.6, y: y + 1.5, w: 0.5, h: 0, line: { color: 'C9CED8', width: 1.5, endArrowType: 'triangle' } });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 11.1, y: y + 0.9, w: 1.6, h: 1.2, rectRadius: 0.08, fill: { color: C.text2 }, line: { type: 'none' } });
  txt(s, 11.1, y + 0.9, 1.6, 1.2, 'out\n65 × 128', { color: C.background1, bold: true, align: 'center', valign: 'middle', fontSize: 15 });
  step(4);
  card(s, 0.6, 5.85, 5.9, 0.9, null, 'Each head attends in its own 32-dim subspace, so heads can specialise.', { bullets: false });
  card(s, 6.8, 5.85, 5.9, 0.9, null, 'Same cost as one wide head. Parameters: 4D² + 4D = 66,048 per layer.', { bullets: false });
}
{
  const s = content('Step 6: the encoder block, stacked L = 6 times',
    `One encoder block has two sub-layers, each wrapped in a residual connection: multi-head attention mixes information across tokens, and the MLP processes each token independently.
ViT uses Pre-LN: LayerNorm is applied before each sub-layer, not after as in the original 2017 transformer. With Pre-LN, the residual path is a clean identity from input to output, so gradients flow through 6, 12 or 32 layers without warm-up tricks blowing up. Post-LN transformers are notoriously hard to train deep.
The MLP expands the width (ratio 2 for our small model, 4 in the paper), applies GELU, and projects back. Attention is the "communication" step, the MLP is the "computation" step.
We also use stochastic depth (DropPath): during training, each residual branch is skipped at random for some samples, with a probability rising to 0.1 in the last layer. This is a regulariser that matters for small datasets.
Parameters per block: attention 66,048 + MLP 65,920 + 2 LayerNorms 512 = 132,480. Times 6 = 794,880, which is 98% of our model.`);
  // block diagram (vertical)
  const bx = 0.9, bw = 3.2;
  const items = [['LayerNorm', C.background2, C.text1], ['Multi-Head Attention', ORANGE, C.background1], ['⊕ residual', 'FFFFFF', C.text1], ['LayerNorm', C.background2, C.text1], ['MLP: 128→256→128, GELU', BLUE, C.background1], ['⊕ residual', 'FFFFFF', C.text1]];
  step(1);
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: bx - 0.3, y: 1.45, w: bw + 0.6, h: 5.3, rectRadius: 0.12, fill: { color: 'FFFFFF' }, line: { color: 'C9CED8', width: 1, dashType: 'dash' } });
  txt(s, bx - 0.25, 1.5, 1.2, 0.35, '× 6', { fontSize: 15, bold: true, color: MUTED });
  items.forEach((it, i) => {
    const y = 6.05 - i * 0.82;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: bx, y, w: bw, h: 0.55, rectRadius: 0.08, fill: { color: it[1] }, line: it[1] === 'FFFFFF' ? { color: 'C9CED8', width: 1 } : { type: 'none' } });
    txt(s, bx, y, bw, 0.55, it[0], { align: 'center', valign: 'middle', bold: true, color: it[2], fontSize: 14 });
    if (i < items.length - 1) connect(s, bx + bw / 2, y, bx + bw / 2, y - 0.27, { color: '8D96A8', width: 1.25, endArrowType: 'triangle' });
  });
  step(2);
  eq(s, 4.7, 1.5, 8.0, 1.2, 'z′ₗ = zₗ₋₁ + MSA( LN(zₗ₋₁) )\nzₗ = z′ₗ + MLP( LN(z′ₗ) )', 22);
  code(s, 4.7, 2.95, 8.0, 1.6, `def forward(self, x):
    # communicate across tokens
    x = x + self.dp(self.attn(self.ln1(x)))
    # compute per token
    x = x + self.dp(self.mlp(self.ln2(x)))
    return x`, 13);
  step(3);
  card(s, 4.7, 4.8, 3.85, 1.95, 'Pre-LN, not Post-LN', 'The residual path stays an identity, so gradients reach early layers and deep stacks train stably.', { bullets: false, size: 16 });
  card(s, 8.85, 4.8, 3.85, 1.95, 'Stochastic depth', 'Randomly skips whole residual branches during training (p rises linearly to 0.1). It is a strong regulariser on small data.', { bullets: false, size: 16 });
}
{
  const s = content('The full model, and where its parameters live',
    `[LIVE DEMO, Ammar: in the terminal run  python demo.py shapes  and read the shapes aloud, then  python demo.py params  to show the table below computed live.]
Putting it together: patch embedding, then add CLS and positions, then 6 encoder blocks, then a final LayerNorm, then read the CLS token, then a linear classifier over 10 classes.
The parameter table was computed by hand with the formulas you have just seen, and it matches PyTorch's count exactly: 811,146. Almost everything is in the encoder blocks. The image-specific parts (patch projection and positions) are under 2% of the model.
For scale: the paper's ViT-Base has D=768, 12 layers and 12 heads, which is 86 million parameters, about 100 times bigger. Our model is deliberately small enough to train on a CPU or a free Colab GPU.
Compute per image: about 2 x 65 x 8D^2 = 17 MFLOPs for the linear layers per block, plus 4 x 65^2 x 128 = 2.2 MFLOPs for the attention matrices. At T=65, the quadratic attention part is only about 11% of the work. It dominates only at long sequence lengths.`);
  // pipeline diagram
  const steps = [['Image', '3×32×32', C.background2, C.text1], ['Patch embed', '64×128', ORANGE, C.background1], ['+CLS +pos', '65×128', PURPLE, C.background1], ['Encoder ×6', '65×128', C.text2, C.background1], ['LN → CLS', '128', '5A6478', C.background1], ['Linear head', '10 logits', BLUE, C.background1]];
  step(1);
  steps.forEach((st, i) => {
    const x = 0.6 + i * 2.05;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.55, w: 1.75, h: 1.05, rectRadius: 0.08, fill: { color: st[2] }, line: { type: 'none' } });
    txt(s, x, 1.6, 1.75, 0.5, st[0], { align: 'center', valign: 'middle', bold: true, color: st[3], fontSize: 14 });
    txt(s, x, 2.05, 1.75, 0.45, st[1], { align: 'center', valign: 'middle', color: st[3], fontSize: 13, fontFace: 'Courier New' });
    if (i < steps.length - 1) s.addShape(pres.shapes.LINE, { x: x + 1.77, y: 2.07, w: 0.26, h: 0, line: { color: '8D96A8', width: 1.25, endArrowType: 'triangle' } });
  });
  const H = { bold: true, color: 'FFFFFF', fill: { color: '101828' } };
  const rows = [
    [{ text: 'Component', options: H }, { text: 'Formula', options: H }, { text: 'Parameters', options: Object.assign({ align: 'right' }, H) }],
    ['Patch embedding', '48·128 + 128', { text: '6,272', options: { align: 'right' } }],
    ['CLS token + positions', '128 + 65·128', { text: '8,448', options: { align: 'right' } }],
    ['Attention (per block)', '4D² + 4D', { text: '66,048', options: { align: 'right' } }],
    ['MLP (per block, r = 2)', '2rD² + (r+1)D', { text: '65,920', options: { align: 'right' } }],
    ['LayerNorms (per block)', '4D', { text: '512', options: { align: 'right' } }],
    ['6 encoder blocks', '6 × 132,480', { text: '794,880', options: { align: 'right' } }],
    ['Final LN + head', '256 + 128·10 + 10', { text: '1,546', options: { align: 'right' } }],
    [{ text: 'Total (matches PyTorch)', options: { bold: true } }, '', { text: '811,146', options: { bold: true, align: 'right', color: 'D9661A' } }],
  ];
  step(2);
  s.addTable(rows, { x: 0.6, y: 2.95, w: 7.6, colW: [3.0, 2.8, 1.8], fontSize: 13, color: '1B2333', border: { type: 'solid', pt: 0.75, color: 'E3E6EC' }, rowH: 0.4 });
  step(3);
  stat(s, 8.7, 3.0, 4.0, '0.81M', 'parameters in our ViT (D=128, 6 layers, 4 heads)', ORANGE);
  stat(s, 8.7, 4.75, 4.0, '86M', 'ViT-Base/16 in the paper (D=768, 12 layers, 12 heads)', MUTED);
}

// ======================================================================
// PART 2: IMPLEMENTATION
// ======================================================================
sectionSlide(2, 'Part 2: Implementation', 'Writing it from scratch, verifying it, training it', 'Dilawar Shah',
  '[Dilawar] Now how we built it. Our rule was no nn.MultiheadAttention and no timm for our model. Every operation is written out, so we can explain and inspect every number.');
{
  const s = content('Our code: about 120 lines of plain PyTorch',
    `[LIVE DEMO, Dilawar: switch to VS Code, open vit.py, show MultiHeadSelfAttention.forward (the qkv reshape/permute, scores, softmax, attn @ v), then EncoderBlock.forward (two lines = two equations).]
This is the top-level forward pass of our ViT, and it reads like the paper. Patch-embed, prepend CLS, add positions, run the blocks, apply a final LayerNorm, take token 0 and classify.
The modules are: PatchEmbedding (Conv2d k=s=P), MultiHeadSelfAttention (a fused qkv linear, a scaled softmax and an output projection; it optionally stores the attention matrix so we can visualise it later), MLP (Linear-GELU-Linear), DropPath, EncoderBlock (Pre-LN with residuals) and ViT.
Initialisation follows the reference implementation: truncated normal with std 0.02 for linear layers, positions and CLS, zeros for biases, and ones for LayerNorm scales. A bad init noticeably slows ViT training.
We also wrote a ResNet-style CNN baseline of similar size (0.69M parameters) in the same file, so the comparison is fair: the same data pipeline, augmentation, optimiser family and epochs.
Everything is in vit.py and in the Colab notebook we are sharing.`);
  step(1);
  code(s, 0.6, 1.5, 7.3, 5.25, `class ViT(nn.Module):
    def forward(self, x, keep_attn=False):
        x = self.patch(x)                 # (B, 64, 128)
        cls = self.cls.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)    # (B, 65, 128)
        x = x + self.pos                  # positions
        for blk in self.blocks:           # 6 blocks
            x = blk(x, keep_attn)
        x = self.ln(x)
        return self.head(x[:, 0])         # CLS -> logits

# modules written by hand:
#   PatchEmbedding         Conv2d(k=P, s=P)
#   MultiHeadSelfAttention fused qkv, softmax(QK^T/sqrt(dk))V
#   MLP                    Linear -> GELU -> Linear
#   DropPath               stochastic depth
#   EncoderBlock           Pre-LN + residuals
#   SmallResNet            CNN baseline, 0.69M params`, 12);
  step(2);
  card(s, 8.2, 1.5, 4.5, 2.45, 'Design choices (paper defaults)', [
    'Pre-LN blocks, GELU MLP',
    'CLS-token readout, linear head',
    'trunc-normal(0.02) initialisation',
  ], { size: 16 });
  card(s, 8.2, 4.2, 4.5, 2.55, 'Our configuration', [
    'P = 4 → 64 patches (+CLS)',
    'D = 128, L = 6, h = 4, MLP 256',
    'DropPath 0.1, label smoothing 0.1',
  ], { size: 16 });
}
{
  const s = content('How we know the implementation is correct',
    `[LIVE DEMO, Dilawar: run  python demo.py check  in the terminal: it prints the difference vs PyTorch (~1e-7) and the initial loss (~2.31).]
Writing a transformer from scratch makes it easy to introduce silent bugs, such as transposing the wrong axes in the head reshape. Training still "works" but gets worse accuracy. So we tested three things before training.
1) Attention correctness: we fed the same Q, K and V into PyTorch's fused scaled_dot_product_attention and into our version. The maximum absolute difference was 6.7e-8, which is floating-point round-off. So our reshape, permute, scaling and softmax are right.
2) Parameter count: our hand formula 4D^2+4D + 2rD^2+(r+1)D + 4D per block, plus the embedding and head, gives 811,146, which is exactly what PyTorch reports.
3) Shapes: we traced one image through the network and checked every intermediate shape against the paper's equations.
We also checked that the training loss starts near ln(10) = 2.30, which is the expected value for a 10-class model with random outputs.`);
  step(1);
  stat(s, 0.6, 1.6, 3.8, '6.7e-8', 'max |our MSA − torch SDPA| on random inputs: identical up to float round-off', ORANGE);
  step(2);
  stat(s, 4.75, 1.6, 3.8, '811,146', 'parameters: hand formula = PyTorch count, exactly', BLUE);
  step(3);
  stat(s, 8.9, 1.6, 3.8, '≈ 2.30', 'initial loss = ln(10), as expected for a random 10-way classifier', PURPLE);
  step(4);
  code(s, 0.6, 3.65, 12.1, 3.1, `# 1) our attention vs PyTorch's fused kernel
att = MultiHeadSelfAttention(64, heads=4); x = torch.randn(2, 10, 64)
q, k, v = att.qkv(x).reshape(2, 10, 3, 4, 16).permute(2, 0, 3, 1, 4)
ref = att.out(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(2, 10, 64))
print((att(x) - ref).abs().max())        # 6.7e-08

# 2) parameter formula vs PyTorch
block = (4*D*D + 4*D) + (2*r*D*D + (r+1)*D) + 4*D   # 132,480 for D=128, r=2`, 14);
}
{
  const s = content('The training recipe, and why ViT needs each part',
    `ViTs are much less forgiving to train than CNNs, because there are no priors to fall back on. Each item here fixes a specific problem.
Warm-up then cosine decay: at initialisation the attention logits are random. A large learning rate in the first steps can make some softmax rows saturate (one-hot), and those heads stop learning. Warm-up ramps the learning rate linearly over the first 2 epochs.
AdamW with decoupled weight decay 0.05: the standard optimiser for transformers. We exclude biases, LayerNorm parameters, CLS and positions from weight decay, because shrinking them towards zero has no regularising meaning.
Label smoothing 0.1 prevents over-confident logits. Gradient clipping at norm 1.0 protects against rare loss spikes.
Augmentation: random crop with 4-pixel reflection padding, plus horizontal flips. This is the minimum. Without it, a ViT memorises CIFAR quickly, as our ablation shows.
All runs: batch 128, peak learning rate 1e-3, the same recipe for the CNN apart from architecture-specific choices. We trained on a 4-core CPU; the Colab notebook runs the same thing on a free T4 GPU.`);
  const items = [
    ['AdamW, wd 0.05', 'Decoupled weight decay. No decay on bias, LayerNorm, CLS or positions.'],
    ['Warm-up → cosine', '2 epochs of linear warm-up stop attention logits saturating early.'],
    ['Label smoothing 0.1', 'Targets 0.91 / 0.01 to avoid over-confident logits.'],
    ['Grad-clip 1.0', 'Guards against occasional loss spikes.'],
    ['DropPath 0.1', 'Randomly drops residual branches: depth-wise regularisation.'],
    ['Crop + flip aug.', 'Pad 4 + random 32×32 crop, horizontal flip.'],
  ];
  items.forEach((it, i) => {
    step(Math.floor(i / 2) + 1);
    const col = i % 2, row = Math.floor(i / 2);
    const x = 0.6 + col * 6.15, y = 1.55 + row * 1.25;
    tile(s, x, y + 0.12, 0.5, [ORANGE, BLUE, PURPLE][row], i + 1);
    txt(s, x + 0.7, y, 5.2, 0.4, it[0], { fontSize: 17, bold: true });
    txt(s, x + 0.7, y + 0.42, 5.2, 0.7, it[1], { fontSize: 14, color: C.text1 });
  });
  // LR schedule chart
  const ep = 25, warm = 2, pts = [], lab = [];
  for (let e = 0; e <= ep; e += 0.5) {
    lab.push(Number.isInteger(e) && e % 5 === 0 ? String(e) : '');
    pts.push(e < warm ? 1e-3 * e / warm : 1e-3 * 0.5 * (1 + Math.cos(Math.PI * (e - warm) / (ep - warm))));
  }
  step(4);
  s.addChart(pres.charts.LINE, [{ name: 'learning rate', labels: lab, values: pts.map((v) => +(v * 1000).toFixed(4)) }], chartBase({
    x: 0.6, y: 5.3, w: 12.1, h: 1.6, chartColors: [HEX.accent1], lineSize: 2, lineDataSymbol: 'none', showLegend: false,
    showTitle: true, title: 'Learning rate (×10⁻³) vs epoch: warm-up then cosine', valAxisMaxVal: 1, valAxisMinVal: 0, valAxisMajorUnit: 0.5,
    catAxisLabelFrequency: 1, catAxisTickLblSkip: 1, titleFontSize: 12,
  }));
}
{
  const s = content('Experimental setup',
    `Everything is controlled. The same CIFAR-10 split (50k train, 10k test), the same augmentation, batch size, epochs and cosine schedule for both the ViT and the CNN.
The CNN is a small ResNet: a 3x3 stem, three residual stages at widths 48, 96 and 192 with stride-2 downsampling, global average pooling and a linear layer. That is 0.69M parameters, slightly fewer than our ViT's 0.81M, so the CNN has no size advantage.
Main runs: 25 epochs each. The ablations use 10 epochs each, compared against a 10-epoch run of the full model, so each comparison is like-for-like. The data-hunger experiment uses a random 10% of the training set (5,000 images) for 30 epochs.
Honest limitations: these are single runs with one seed on CPU, so differences under about 1% should not be over-interpreted. The paper-scale models train for hundreds of epochs; ours train for 25.`);
  const H = { bold: true, color: 'FFFFFF', fill: { color: '101828' } };
  const rows = [
    [{ text: 'Run', options: H }, { text: 'Model', options: H }, { text: 'Params', options: H }, { text: 'Data', options: H }, { text: 'Epochs', options: H }, { text: 'Purpose', options: H }],
    ['vit_main', 'ViT P=4, D=128, L=6, h=4', `${(vm.params / 1e6).toFixed(2)}M`, '50k', '25', 'main result'],
    ['cnn_main', 'ResNet-style (48-96-192)', `${(cm.params / 1e6).toFixed(2)}M`, '50k', '25', 'CNN baseline'],
    ['vit_main10', 'same ViT', `${(vm.params / 1e6).toFixed(2)}M`, '50k', '10', 'ablation reference'],
    ['vit_nopos', 'no position embedding', `${(abl.nopos.params / 1e6).toFixed(2)}M`, '50k', '10', 'role of positions'],
    ['vit_p8', 'patch 8 → 16 tokens', `${(abl.p8.params / 1e6).toFixed(2)}M`, '50k', '10', 'patch size'],
    ['vit_noaug', 'no augmentation', `${(vm.params / 1e6).toFixed(2)}M`, '50k', '10', 'regularisation'],
    ['vit / cnn_10pct', 'both models', '—', '5k (10%)', '30', 'data hunger'],
  ];
  step(1);
  s.addTable(rows, { x: 0.6, y: 1.55, w: 12.1, colW: [1.9, 3.3, 1.2, 1.3, 1.1, 3.3], fontSize: 14, color: '1B2333', border: { type: 'solid', pt: 0.75, color: 'E3E6EC' }, rowH: 0.48 });
  step(2);
  card(s, 0.6, 5.65, 12.1, 1.05, null, 'Same split, augmentation, batch size (128), optimiser family and cosine schedule for every run. Single seed on a 4-core CPU, so treat differences under about 1 point as noise.', { bullets: false, size: 15 });
}

// ======================================================================
// PART 3: RESULTS
// ======================================================================
sectionSlide(3, 'Part 3: Experiments', 'ViT vs CNN, ablations, data hunger, pre-training', 'Syed Khizar Hussain Kazmi',
  '[Khizar] Now the results. Every number on these slides comes from our own runs.');
{
  const vAcc = vm.final_test_acc, cAcc = cm.final_test_acc;
  const s = content('Head-to-head: ViT vs CNN at equal budget',
    `[LIVE DEMO, Khizar: run  python demo.py predict  : the trained model classifies 8 random test images, then scores all 10,000 in about 10 seconds and prints 79.02%, the same number as on this slide.]
After 25 epochs on the full CIFAR-10 training set, our from-scratch ViT reaches ${pct(vAcc)} test accuracy. The ResNet-style CNN, with slightly fewer parameters, reaches ${pct(cAcc)}.
That gap is the inductive-bias story in one number. The CNN gets locality and translation equivariance for free. The ViT has to learn them from 50,000 small images. This is exactly the paper's finding at small scale: on mid-sized datasets without pre-training, ViTs trail CNNs of similar size.
Note the training accuracy too. The ViT reaches ${pct(vm.history.at(-1).train_acc)} train accuracy (with augmentation and label smoothing) against a test accuracy of ${pct(vAcc)}. The CNN reaches ${pct(cm.history.at(-1).train_acc)} against ${pct(cAcc)}.
Training cost on our CPU: about ${Math.round(vm.history.at(-1).time_s / 60)} minutes for the ViT and ${Math.round(cm.history.at(-1).time_s / 60)} minutes for the CNN. On CPU, the ViT is actually cheaper per epoch, because its large matrix multiplies are very efficient compared with many small convolutions.`);
  step(1);
  stat(s, 0.6, 1.7, 3.8, pct(vAcc), `ViT from scratch: ${(vm.params / 1e6).toFixed(2)}M params, 25 epochs`, ORANGE);
  step(2);
  stat(s, 0.6, 3.55, 3.8, pct(cAcc), `ResNet-style CNN: ${(cm.params / 1e6).toFixed(2)}M params, 25 epochs`, BLUE);
  txt(s, 0.6, 5.4, 3.8, 1.3, [
    { text: `Gap: ${((cAcc - vAcc) * 100).toFixed(1)} points`, options: { bold: true, breakLine: true } },
    { text: 'Same data, augmentation, schedule and epochs. The CNN even has fewer parameters.' },
  ], { fontSize: 15 });
  const labels = vm.history.map((h) => String(h.epoch));
  step(3);
  s.addChart(pres.charts.LINE, [
    { name: 'ViT (ours)', labels, values: vm.history.map((h) => +(h.test_acc * 100).toFixed(2)) },
    { name: 'CNN', labels, values: cm.history.map((h) => +(h.test_acc * 100).toFixed(2)) },
  ], chartBase({
    x: 4.7, y: 1.45, w: 8.0, h: 5.35, chartColors: [HEX.accent1, HEX.accent2], lineSize: 2, lineDataSymbol: 'circle', lineDataSymbolSize: 5,
    showLegend: true, legendPos: 't', showTitle: true, title: 'Test accuracy (%) per epoch', valAxisMinVal: 30, valAxisMaxVal: 100, valAxisMajorUnit: 10,
    showCatAxisTitle: true, catAxisTitle: 'epoch', catAxisTitleColor: '5A6478', catAxisTitleFontSize: 12,
  }));
}
{
  const s = content('Where each model wins and fails',
    `Per-class test accuracy for both models. The ordering of classes is the same for both: vehicles (automobile, ship, truck) are easy, while animals that share texture and pose (cat, dog, bird) are hard. Both models struggle with the same classes.
The ViT's most common confusions are: ${S.top_confusions.map((c) => `${c[0]} predicted as ${c[1]} (${c[2]} times)`).join('; ')}. Cat versus dog is the classic CIFAR-10 failure: at 32x32 pixels, much of the distinguishing evidence is a few pixels on the face.
The ViT's gap to the CNN is largest on the fine-grained animal classes, where local texture cues matter most and the CNN's locality prior helps.`);
  step(1);
  s.addChart(pres.charts.BAR, [
    { name: 'ViT (ours)', labels: ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'], values: S.vit_main_per_class.map((v) => +(v * 100).toFixed(1)) },
    { name: 'CNN', labels: ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'], values: S.cnn_main_per_class.map((v) => +(v * 100).toFixed(1)) },
  ], chartBase({
    x: 0.6, y: 1.45, w: 7.3, h: 5.35, barDir: 'col', barGapWidthPct: 60, barOverlapPct: -10, chartColors: [HEX.accent1, HEX.accent2],
    showLegend: true, legendPos: 't', showTitle: true, title: 'Per-class test accuracy (%)', valAxisMinVal: 40, valAxisMaxVal: 100, valAxisMajorUnit: 10,
  }));
  step(2);
  s.addImage({ path: 'figs/confusion.png', x: 8.15, y: 1.45, w: 4.6, h: 4.0 });
  txt(s, 8.15, 5.5, 4.6, 1.3, [
    { text: 'ViT confusion matrix. ', options: { bold: true } },
    { text: `Top error: ${S.top_confusions[0][0]} → ${S.top_confusions[0][1]} (${S.top_confusions[0][2]}×). At 32×32 the evidence is a few pixels.` },
  ], { fontSize: 13 });
}
{
  const full = abl.full.final_test_acc;
  const items = [['Full ViT', full], ['No position emb.', abl.nopos.final_test_acc], ['Patch 8 (16 tokens)', abl.p8.final_test_acc], ['No augmentation', abl.noaug.final_test_acc]];
  const noaugTrain = abl.noaug.history.at(-1).train_acc;
  const s = content('Ablations: removing one ingredient at a time',
    `Each bar is a 10-epoch run that changes exactly one thing from the full model, which reaches ${pct(full)} at 10 epochs.
No positional embedding: ${pct(abl.nopos.final_test_acc)}. Without positions, the model sees a bag of patches. It still does surprisingly well, because many classes can be recognised from the patch contents alone (colour, texture), but it loses ${((full - abl.nopos.final_test_acc) * 100).toFixed(1)} points. Spatial arrangement matters.
Patch size 8: ${pct(abl.p8.final_test_acc)}. Only 16 tokens, each covering a quarter of the image's width. That is about 4x fewer tokens and much cheaper (about ${Math.round(abl.p8.history.at(-1).time_s / 60)} vs ${Math.round(abl.full.history.at(-1).time_s / 60)} minutes), but each patch is too coarse for 32x32 images. This is why we use P=4 on CIFAR, while ImageNet models use P=16 on 224x224 images: both give patches about 1/8 to 1/14 of the image side.
No augmentation: ${pct(abl.noaug.final_test_acc)} test accuracy, but ${pct(noaugTrain)} train accuracy. With augmentation, train and test accuracy stay about equal (${pct(abl.full.history.at(-1).train_acc)} train vs ${pct(full)} test). Without it, a ${((noaugTrain - abl.noaug.final_test_acc) * 100).toFixed(0)}-point generalisation gap opens after only 10 epochs, and it keeps growing with longer training. The test-accuracy drop is smallest at 10 epochs; the overfitting signal is the important part. Augmentation is not optional for ViTs on small data.`);
  step(1);
  s.addChart(pres.charts.BAR, [{ name: 'test acc', labels: items.map((i) => i[0]), values: items.map((i) => +(i[1] * 100).toFixed(1)) }], chartBase({
    x: 0.6, y: 1.45, w: 7.2, h: 5.35, barDir: 'bar', chartColors: [HEX.accent1, HEX.accent4, HEX.accent4, HEX.accent4], invertedColors: [HEX.accent1],
    showValue: true, dataLabelPosition: 'outEnd', dataLabelFormatCode: '0.0"%"', showLegend: false, showTitle: true, title: 'Test accuracy after 10 epochs (%)',
    valAxisMinVal: 0, valAxisMaxVal: 100, valAxisMajorUnit: 20, catAxisOrientation: 'maxMin', barGapWidthPct: 50, catAxisLabelFontSize: 13,
  }));
  step(2);
  card(s, 8.1, 1.45, 4.6, 1.65, `Positions: −${((full - abl.nopos.final_test_acc) * 100).toFixed(1)} pts`, 'Without them the model sees a bag of patches. Content alone gets far, but layout matters.', { bullets: false, size: 13, headSize: 16 });
  step(3);
  card(s, 8.1, 3.3, 4.6, 1.65, `Patch 8: −${((full - abl.p8.final_test_acc) * 100).toFixed(1)} pts`, '4× fewer tokens is much faster, but each 8×8 patch is a quarter of the image: too coarse.', { bullets: false, size: 13, headSize: 16 });
  step(4);
  card(s, 8.1, 5.15, 4.6, 1.65, `No aug: −${((full - abl.noaug.final_test_acc) * 100).toFixed(1)} pts`, `Train ${pct(noaugTrain)} vs test ${pct(abl.noaug.final_test_acc)}: overfitting starts already (it is about equal with augmentation).`, { bullets: false, size: 13, headSize: 16 });
}
{
  const vf = vm.final_test_acc, cf = cm.final_test_acc, vs = v10.final_test_acc, cs = c10.final_test_acc;
  const s = content('Data hunger: shrink the dataset and the gap widens',
    `This is the central claim of the ViT paper, reproduced at our scale. We trained both models on a random 10% of CIFAR-10 (5,000 images) for 30 epochs.
With 100% of the data the CNN leads by ${((cf - vf) * 100).toFixed(1)} points. With 10% of the data the CNN leads by ${((cs - vs) * 100).toFixed(1)} points. The ViT drops from ${pct(vf)} to ${pct(vs)} and the CNN from ${pct(cf)} to ${pct(cs)}.
Why? A CNN's architecture already "knows" that nearby pixels are related and that a cat is a cat anywhere in the frame. The ViT must discover both facts from examples, and with 5,000 images there are not enough examples.
The paper shows the other end of this curve. On ImageNet-1k (1.3M images), ViT-B is worse than a ResNet. On ImageNet-21k (14M) they are similar. On JFT-300M (300M images) ViT wins. Weak priors are a handicap with little data, and an advantage with a lot of data, because the model is not limited by assumptions that are only approximately true.`);
  step(1);
  s.addChart(pres.charts.BAR, [
    { name: 'ViT (ours)', labels: ['10% data (5k)', '100% data (50k)'], values: [+(vs * 100).toFixed(1), +(vf * 100).toFixed(1)] },
    { name: 'CNN', labels: ['10% data (5k)', '100% data (50k)'], values: [+(cs * 100).toFixed(1), +(cf * 100).toFixed(1)] },
  ], chartBase({
    x: 0.6, y: 1.45, w: 6.6, h: 5.35, barDir: 'col', chartColors: [HEX.accent1, HEX.accent2], barGapWidthPct: 70, barOverlapPct: -10,
    showValue: true, dataLabelPosition: 'outEnd', dataLabelFormatCode: '0.0"%"', showLegend: true, legendPos: 't', showTitle: true, title: 'Test accuracy (%)',
    valAxisMinVal: 0, valAxisMaxVal: 100, valAxisMajorUnit: 20,
  }));
  step(2);
  stat(s, 7.6, 1.6, 5.1, `${((cs - vs) * 100).toFixed(1)} pts`, 'CNN lead with 10% of the data', BLUE);
  stat(s, 7.6, 3.25, 5.1, `${((cf - vf) * 100).toFixed(1)} pts`, 'CNN lead with 100% of the data', ORANGE);
  step(3);
  card(s, 7.6, 4.95, 5.1, 1.85, 'Paper, at much larger scale', 'ImageNet-1k: ViT < ResNet. ImageNet-21k: about equal. JFT-300M: ViT > ResNet. Priors help with little data and limit you with a lot.', { bullets: false, size: 13, headSize: 15 });
}
{
  const s = content('The fix: pre-train big, then fine-tune',
    FT ? `[LIVE DEMO, Khizar: run  python predict.py examples/cat.jpg  then  python predict.py examples/horse.jpg  : both models guess the cat; on the horse our ViT fails and the pretrained ViT is right. All 10 example photos: our ViT 7/10, pretrained 10/10.]\nIf a ViT needs lots of data, give it lots of data once and reuse it. We took a ViT-Tiny/16 from timm (5.5M parameters), pre-trained on ImageNet-21k and fine-tuned on ImageNet-1k. We upsampled CIFAR images to 224x224 (196 patches of 16x16), replaced the head with a 10-class layer, and fine-tuned the whole network.
Result: ${pct(FT.test_acc)} test accuracy after fine-tuning on only ${FT.train_images.toLocaleString()} training images for ${FT.epochs} epoch(s) (evaluated on ${FT.test_images.toLocaleString()} test images, CPU budget). Our from-scratch ViT reached ${pct(vm.final_test_acc)} with all 50,000 images and 25 epochs.
Note on fairness: this model is about 7x larger and sees images at 224 resolution, so it is not an architecture comparison. It is a demonstration that pre-training removes the data-hunger problem. With the full training set on a Colab GPU (section 6 of the notebook), ViT-Small/16 typically reaches about 97 to 98%.
This is how ViTs are used in practice: almost nobody trains one from scratch on a small dataset.` : 'Fine-tuning results pending.');
  if (FT) {
    step(1);
    s.addChart(pres.charts.BAR, [{ name: 'acc', labels: ['ViT from scratch (50k imgs, 25 ep)', 'CNN from scratch (50k imgs, 25 ep)', `Pretrained ViT-Ti/16, fine-tuned (${(FT.train_images / 1000).toFixed(0)}k imgs, ${FT.epochs} ep)`],
      values: [+(vm.final_test_acc * 100).toFixed(1), +(cm.final_test_acc * 100).toFixed(1), +(FT.test_acc * 100).toFixed(1)] }], chartBase({
      x: 0.6, y: 1.45, w: 7.4, h: 5.35, barDir: 'bar', chartColors: [HEX.accent1, HEX.accent2, HEX.accent3], showValue: true, dataLabelPosition: 'outEnd', dataLabelFormatCode: '0.0"%"',
      showLegend: false, showTitle: true, title: 'CIFAR-10 test accuracy (%)', valAxisMinVal: 0, valAxisMaxVal: 100, valAxisMajorUnit: 20, catAxisOrientation: 'maxMin', barGapWidthPct: 45, catAxisLabelFontSize: 12,
    }));
    step(2);
    stat(s, 8.4, 1.6, 4.3, pct(FT.test_acc), `pretrained ViT after fine-tuning on ${(FT.train_images / 1000).toFixed(0)}k images`, PURPLE);
    step(3);
    card(s, 8.4, 3.55, 4.3, 3.25, 'Recipe', [
      'timm vit_tiny_patch16_224 (IN-21k → IN-1k)',
      'Resize 32 → 224: 196 tokens of 16×16',
      'New 10-class head; fine-tune all layers',
      `AdamW, lr 5e-5, ${FT.epochs} epoch(s) on CPU`,
    ], { size: 13, headSize: 15 });
  }
}

// ======================================================================
// PART 4: INSIDE THE MODEL
// ======================================================================
sectionSlide(4, 'Part 4: Inside the Model', 'What did our ViT actually learn?', 'Muhammad Farhan Khan',
  '[Farhan] Accuracy tells us how well the model works. Now we open it up and look at what it learned.');
{
  const s = content('Where does the CLS token look? Attention rollout',
    `[CODE, Farhan: switch to VS Code, open analysis.py and show the rollout() function: average heads, add identity for the residual, renormalise, multiply through the layers, take the CLS row. Optionally run  python analysis.py  beforehand and open figs/rollout.png.]
Note for the audience: CIFAR-10 images are only 32x32 pixels, so we show them enlarged pixel-for-pixel. The blocky look is the real resolution, and each 4x4 block of pixels in the heat-map row is one token.
Top row: correctly classified test images. Bottom row: attention rollout from our trained ViT, brighter meaning more influence on the CLS token.
How rollout works (Abnar and Zuidema, 2020): looking at only the last layer is misleading, because information has already been mixed by earlier layers. Rollout multiplies the attention matrices of all 6 layers. At each layer we average the heads and add the identity matrix, to account for the residual connection that carries each token's own information forward: R = product over layers of 0.5*(A + I), row-normalised. Row 0 of R shows how much each patch contributes to the CLS token.
What we see: for many images the attention concentrates on the object. The deer's body, the cat, the truck's cab and the horse and rider stand out, while plain background such as the sky behind the airplane is mostly ignored. Nobody gave the model bounding boxes or segmentation masks; it learned where to look from class labels alone.
But be honest about it: the maps are not clean. For the automobile, attention spreads over the background as well. A small model trained for 25 epochs on 50k images also uses context. Large pretrained ViTs, and especially self-supervised ones like DINO, give much sharper object maps.
Caveat: rollout is a heuristic visualisation, not a causal explanation, and on an 8x8 token grid the maps are coarse.`);
  step(1);
  s.addImage({ path: 'figs/rollout.png', x: 0.6, y: 1.5, w: 12.1, h: 12.1 * 4.3 / 16 });
  step(2);
  eq(s, 0.6, 5.0, 5.4, 0.85, 'R = ∏ₗ ½ ( Āₗ + I ) ,   saliency = R[CLS, patches]', 18);
  txt(s, 6.3, 4.95, 6.4, 1.8, [
    { text: 'Often on the object, sometimes on context', options: { bold: true, breakLine: true, paraSpaceAfter: 4 } },
    { text: 'Deer, cat and truck light up; the sky is ignored; the car map is diffuse. Learned from labels alone, with no boxes or masks.' },
  ], { fontSize: 14 });
}
{
  const ad = S.attn_distance; // layers x heads
  const lm = ad.map((l) => l.reduce((a, b) => a + b, 0) / l.length);
  const s = content('Heads differ, but locality is only weakly learned',
    `Left: for one ship image, the CLS token's attention in each of the 4 heads, in the first layer (top) and the last layer (bottom). The heads clearly do different things. In layer 1, head 3 looks only at the sky at the top of the image, while heads 1 and 4 look at the hull. In layer 6, head 3 focuses on the centre of the ship.
Right: mean attention distance. For every query patch we compute the average pixel distance to the patches it attends to, weighted by attention, and average over 500 test images. Each dot is one head in one layer. For reference, uniform attention over the 8x8 grid would give ${S.attn_distance_uniform.toFixed(1)} px.
Result: the trend matches Figure 7 of the ViT paper. Early layers are the most local (the layer-averaged distance is ${lm[1].toFixed(1)} px in layer 2) and the distance grows with depth to ${lm[5].toFixed(1)} px in layer 6, close to uniform.
The honest part: the effect is weak. Even the most local head (${Math.min(...ad.flat()).toFixed(1)} px) is far from convolution-like. In the paper, ImageNet-21k and JFT-trained ViTs have early heads attending only a few pixels away. Our small-data model has not had enough examples to learn a strong locality prior. This is the data-hunger result again, seen from inside the network, and it explains the CNN's advantage.`);
  step(1);
  s.addImage({ path: 'figs/heads.png', x: 0.6, y: 1.5, w: 6.3, h: 6.3 * 4.4 / 10 });
  txt(s, 0.6, 4.4, 6.3, 0.7, 'CLS attention per head in layer 1 (top) and layer 6 (bottom), for one test image', { fontSize: 12, color: MUTED });
  const series = [0, 1, 2, 3].map((h) => ({ name: `head ${h + 1}`, labels: ['L1', 'L2', 'L3', 'L4', 'L5', 'L6'], values: ad.map((l) => +l[h].toFixed(2)) }));
  step(2);
  s.addChart(pres.charts.LINE, series, chartBase({
    x: 7.2, y: 1.45, w: 5.5, h: 5.35, chartColors: [HEX.accent1, HEX.accent2, HEX.accent3, HEX.accent4], lineSize: 0, lineDataSymbol: 'circle', lineDataSymbolSize: 10,
    showLegend: true, legendPos: 't', showTitle: true, title: `Mean attention distance (px); uniform = ${S.attn_distance_uniform.toFixed(1)}`, valAxisMinVal: 10, valAxisMaxVal: 18, valAxisMajorUnit: 2,
    showCatAxisTitle: true, catAxisTitle: 'layer', catAxisTitleColor: '5A6478', catAxisTitleFontSize: 12,
  }));
  step(3);
  card(s, 0.6, 5.15, 6.3, 1.65, null, [
    `Distance grows with depth: ${lm[1].toFixed(1)} px (layer 2) → ${lm[5].toFixed(1)} px (layer 6)`,
    `Uniform baseline ${S.attn_distance_uniform.toFixed(1)} px: locality only weakly learned on 50k images`,
  ], { size: 16 });
}
{
  const s = content('The model discovers 2-D geometry by itself',
    `Left: each small square corresponds to one of the 64 patch positions. It shows the cosine similarity between that position's learned embedding and every other position's embedding (yellow = similar).
Remember: the position embeddings are a plain list of 65 vectors, with no 2-D structure given. Yet look at the tiles: the bright band in each tile sits on that patch's own row. The model has rediscovered the rows of the 8x8 grid from data. Quantitatively, the correlation between embedding similarity and the true distance between patches is r = ${S.posemb_corr_with_distance.toFixed(2)}: the further apart two patches are, the less similar their embeddings.
A second, more surprising pattern: every tile is mirror-symmetric left to right, and column 1 looks like column 8. Why? Our horizontal-flip augmentation shows every image both ways round, so "left" and "right" carry no information about the class. The model learned that, and spent its position capacity on the vertical axis, which is informative (sky at the top, ground at the bottom). This is a fingerprint of the training data in the weights.
This also explains why the paper found 2-D-aware position embeddings gave no real gain over simple 1-D learned ones: the model learns whatever spatial structure is useful.`);
  step(1);
  s.addImage({ path: 'figs/posemb.png', x: 0.6, y: 1.45, w: 5.35, h: 5.35 });
  step(2);
  card(s, 6.4, 1.45, 6.3, 1.65, 'Rows are learned', `In each tile the bright band sits on that patch's own row. Similarity falls with distance: r = ${S.posemb_corr_with_distance.toFixed(2)}.`, { bullets: false, size: 16 });
  step(3);
  card(s, 6.4, 3.3, 6.3, 1.65, 'Left-right mirror symmetry', 'Every tile is symmetric, and column 1 ≈ column 8. Horizontal-flip augmentation makes left and right uninformative, and the model learned that.', { bullets: false, size: 16 });
  step(4);
  card(s, 6.4, 5.15, 6.3, 1.65, 'Nothing about 2-D was given', 'The embeddings are a plain list of 65 vectors. The spatial structure comes only from data, which is why 2-D-aware embeddings add little.', { bullets: false, size: 16 });
}
{
  const s = content('Failure cases: what the model gets wrong',
    `These are the first misclassified test images, not cherry-picked. Many errors are understandable even for a person at 32x32: the typical mistakes are between animals of similar shape and colour, or vehicles seen from unusual angles.
Lessons. First, colour shortcuts: both airplanes here are green, or sit on green, and both were predicted as frog. The model has learned that "a lot of green" means frog. Second, unusual viewpoints and close-ups: an ostrich face filling the frame becomes a cat, and a reindeer lying down becomes a bird. Third, similar shapes: deer and horse are confused in both directions, and so are cat and dog.
How to do better: more epochs (ViT keeps improving well past 25 epochs), stronger augmentation (RandAugment, Mixup and CutMix are standard in DeiT), and above all pre-training, as we showed on the fine-tuning slide.`);
  step(1);
  s.addImage({ path: 'figs/wrong.png', x: 0.6, y: 1.5, w: 12.1, h: 12.1 * 2.6 / 16 });
  step(2);
  card(s, 0.6, 3.9, 3.85, 2.3, 'Colour shortcuts', 'Both green airplanes were predicted as frog. The model leans on colour and texture, not only shape.', { bullets: false, size: 16 });
  step(3);
  card(s, 4.72, 3.9, 3.85, 2.3, 'Odd views, similar shapes', 'A close-up ostrich face became a cat. Deer and horse, and cat and dog, are confused both ways.', { bullets: false, size: 16 });
  step(4);
  card(s, 8.85, 3.9, 3.85, 2.3, 'How to improve', 'Train longer; add RandAugment, Mixup or CutMix; and above all start from pretrained weights.', { bullets: false, size: 16 });
}
{
  const s = content('After ViT: the ideas that followed',
    `ViT opened a family of models. Very briefly:
DeiT (Touvron et al., 2021) showed that ViTs can be trained on ImageNet-1k alone, with no extra data, using strong augmentation, regularisation and a distillation token that learns from a CNN teacher.
Swin Transformer (Liu et al., 2021) brings back some CNN priors. It uses attention within local windows, shifts the windows between layers so information crosses window boundaries, and merges patches into a hierarchy. Cost becomes linear in image size, which made transformers practical for detection and segmentation.
MAE (He et al., 2022) does self-supervised pre-training without labels: hide 75% of the patches and train the model to reconstruct the missing pixels. It is BERT's masked-word idea for images.
DINO (Caron et al., 2021) uses self-distillation without labels. Its CLS attention maps segment objects almost perfectly, a sharper version of what we saw in our rollout.
CLIP (Radford et al., 2021) trains a ViT image encoder with a text transformer on 400M image-caption pairs, enabling zero-shot classification. It is also the vision encoder used in many multimodal LLMs.`);
  const rows = [
    ['DeiT', '2021', 'Strong aug + distillation token', 'ViT trained on ImageNet-1k alone, no extra data'],
    ['Swin', '2021', 'Shifted local windows + hierarchy', 'Linear cost; backbone for detection and segmentation'],
    ['MAE', '2022', 'Mask 75% of patches, reconstruct pixels', 'Self-supervised pre-training without labels'],
    ['DINO', '2021', 'Self-distillation, no labels', 'Attention maps segment objects'],
    ['CLIP', '2021', 'ViT + text encoder, contrastive on 400M pairs', 'Zero-shot classification; vision encoder of many VLMs'],
  ];
  rows.forEach((r, i) => {
    step(i + 1);
    const y = 1.5 + i * 1.04;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y, w: 12.1, h: 0.88, rectRadius: 0.08, fill: { color: i % 2 ? 'FFFFFF' : C.background2 }, line: i % 2 ? { color: 'E3E6EC', width: 0.75 } : { type: 'none' } });
    tile(s, 0.8, y + 0.14, 0.6, [ORANGE, BLUE, PURPLE, '5A6478', C.text2][i], '');
    txt(s, 1.6, y, 1.5, 0.88, r[0], { fontSize: 20, bold: true, fontFace: 'Cambria', valign: 'middle' });
    txt(s, 3.1, y, 0.8, 0.88, r[1], { fontSize: 13, color: MUTED, valign: 'middle' });
    txt(s, 3.9, y, 4.3, 0.88, r[2], { fontSize: 14, valign: 'middle', bold: true });
    txt(s, 8.4, y, 4.1, 0.88, r[3], { fontSize: 14, valign: 'middle' });
  });
  txt(s, 0.6, 6.75 - 0.05, 12.1, 0.3, ' ', { fontSize: 10 });
}

// ======================================================================
// CLOSING
// ======================================================================
pres.addSection({ title: 'Closing' }); section = 'Closing';
{
  const s = content('Takeaways',
    `Four things to remember.
1. A ViT is just a standard transformer encoder applied to patch tokens. The only vision-specific parts are the patch projection (6,272 parameters in our model) and the position embeddings.
2. Weak inductive bias cuts both ways. From scratch on CIFAR-10, our ViT trails a same-size CNN, and the gap grows when data shrinks. This is the paper's main finding, reproduced at small scale.
3. Pre-training solves it. A pretrained ViT fine-tunes to far higher accuracy with a fraction of the data and epochs. That is how ViTs are used in practice.
4. A ViT learns structure that a CNN hard-codes, but only as much as the data supports. Its position embeddings learned the rows of the grid and even the left-right symmetry from flip augmentation, and its attention often lands on the object. Locality was only weakly learned on 50k images, which is the inside view of data hunger.
All our code is in one Colab notebook, and every number in this talk can be reproduced from it.`);
  const items = [
    ['Just a transformer', 'Patches are tokens. The only vision-specific parts are the patch projection and the positions.', ORANGE],
    ['Weak priors cut both ways', `Behind the CNN by ${((cm.final_test_acc - vm.final_test_acc) * 100).toFixed(1)} pts on 50k images and ${((c10.final_test_acc - v10.final_test_acc) * 100).toFixed(1)} pts on 5k.`, BLUE],
    ['Pre-training fixes it', FT ? `A fine-tuned pretrained ViT reaches ${pct(FT.test_acc)}.` : 'Fine-tuned pretrained ViT.', PURPLE],
    ['It learns structure from data', 'Rows in the position embeddings, flip symmetry, object-focused attention. Locality is only weakly learned at this scale.', C.text2],
  ];
  items.forEach((it, i) => {
    step(i + 1);
    const x = 0.6 + (i % 2) * 6.15, y = 1.55 + Math.floor(i / 2) * 2.6;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: 5.95, h: 2.35, rectRadius: 0.12, fill: { color: C.background2 }, line: { type: 'none' } });
    tile(s, x + 0.3, y + 0.3, 0.65, it[2], i + 1);
    txt(s, x + 1.2, y + 0.3, 4.5, 0.65, it[0], { fontSize: 20, bold: true, fontFace: 'Cambria', valign: 'middle' });
    txt(s, x + 1.2, y + 1.05, 4.5, 1.2, it[1], { fontSize: 17 });
  });
}
{
  const s = content('Backup: questions we expect',
    `Backup slide. Don't present it; use it during Q&A.
Q: Why is the CLS token needed? Could you average the tokens? A: Global average pooling also works. The paper reports similar accuracy with a tuned learning rate. CLS is kept for consistency with BERT. Our code supports both (use_cls=False).
Q: Why patches of 16 on ImageNet but 4 on CIFAR? A: To keep a sensible number of tokens and patch granularity relative to the image. 224/16 = 14 and 32/4 = 8 patches per side.
Q: Isn't the patch embedding a convolution? A: Mathematically it is a linear map per patch with no overlap, so there is no hierarchy or sliding-window prior beyond the patch itself.
Q: What is the complexity? A: O(N^2 D) for attention and O(N D^2) for the MLPs. At N=65 the MLP and projections dominate. At high resolution attention dominates, which is why Swin uses windows.
Q: Why does ViT beat CNNs at scale if it has weaker priors? A: Priors are assumptions. When data is abundant, a model that can learn the right structure, including long-range interactions, is no longer limited by approximate assumptions.
Q: Does the model really need positions? A: It needs them for spatial reasoning. Our ablation without positions lost several points.`);
  const qa = [
    ['CLS vs average pooling?', 'Both work. The paper reports similar accuracy; CLS mirrors BERT. Our code supports both.'],
    ['P = 16 vs P = 4?', 'Keeps tokens per side similar: 224/16 = 14 and 32/4 = 8.'],
    ['Is patch embedding a CNN?', 'It is one linear map per non-overlapping patch, with no sliding window or hierarchy.'],
    ['Complexity?', 'O(N²D) for attention and O(ND²) for the MLPs. Attention dominates only at high resolution.'],
    ['Why does ViT win at scale?', 'Priors are assumptions. With enough data, learned structure beats approximate assumptions.'],
    ['Are positions needed?', `Yes, for layout: −${((abl.full.final_test_acc - abl.nopos.final_test_acc) * 100).toFixed(1)} pts without them in our ablation.`],
  ];
  qa.forEach((q, i) => {
    const x = 0.6 + (i % 2) * 6.15, y = 1.5 + Math.floor(i / 2) * 1.75;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: 5.95, h: 1.55, rectRadius: 0.1, fill: { color: C.background2 }, line: { type: 'none' } });
    txt(s, x + 0.25, y + 0.15, 5.45, 0.4, q[0], { fontSize: 16, bold: true, color: ORANGE });
    txt(s, x + 0.25, y + 0.6, 5.45, 0.85, q[1], { fontSize: 14 });
  });
}
{
  const s = content('References',
    'Key references. The ViT paper is the essential one; the Attention Is All You Need paper is the origin of the transformer architecture itself.');
  const refs = [
    'Vaswani et al. (2017). Attention Is All You Need. NeurIPS.',
    'Dosovitskiy et al. (2021). An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale. ICLR.',
    'Touvron et al. (2021). Training data-efficient image transformers & distillation through attention (DeiT). ICML.',
    'Liu et al. (2021). Swin Transformer: Hierarchical Vision Transformer using Shifted Windows. ICCV.',
    'He et al. (2022). Masked Autoencoders Are Scalable Vision Learners (MAE). CVPR.',
    'Caron et al. (2021). Emerging Properties in Self-Supervised Vision Transformers (DINO). ICCV.',
    'Radford et al. (2021). Learning Transferable Visual Models From Natural Language Supervision (CLIP). ICML.',
    'Abnar & Zuidema (2020). Quantifying Attention Flow in Transformers (attention rollout). ACL.',
    'Krizhevsky (2009). Learning Multiple Layers of Features from Tiny Images (CIFAR-10).',
  ];
  txt(s, 0.6, 1.5, 12.1, 5.2, refs.map((r, i) => ({ text: r, options: { bullet: { type: 'number' }, breakLine: i < refs.length - 1, paraSpaceAfter: 8 } })), { fontSize: 15 });
}
{
  const s = pres.addSlide({ masterName: 'TITLE_DARK', sectionTitle: section });
  s.addImage({ path: 'figs/patch_seq.png', x: 0.8, y: 0.9, w: 8.4, h: 0.62 });
  s.addText('Thank you: questions?', { placeholder: 'title' });
  s.addText([{ text: 'Code and notebook: ViT_from_scratch_CIFAR10.ipynb (runs on a free Colab T4)', options: { breakLine: true } },
    { text: 'Implementation Explanation · Deep Learning', options: { color: ORANGE, fontSize: 16 } }], { placeholder: 'body' });
  groupNames(s);
  s.addNotes('Thank you. We are happy to take questions. The notebook reproduces every number in this talk.');
}

(async () => {
  finishNotes();
  await pres.writeFile({ fileName: 'ViT_Presentation.pptx' });
  if (applyTheme) await applyTheme('ViT_Presentation.pptx', THEME);
  else console.warn('apply_theme.js not found: built without the custom colour theme');
  console.log('done');
})();
