
# CDSSL: Self-Supervised Learning Using Nonlinear Dependence



**CDSSL** is a general SSL framework that unifies linear *correlation* objectives (e.g., Barlow Twins / VICReg style) with nonlinear *dependence* via **HSIC** in RKHS. It introduces **8 complementary loss terms** across sample/feature × auto/cross × linear/nonlinear axes, yielding richer, less-redundant representations and stronger transfer. :contentReference[oaicite:0]{index=0}

---

## Highlights
- **Unified view of SSL:** Shows how VICReg, Barlow Twins, SimCLR/InfoNCE, and SSL-HSIC fit as **special cases** of CDSSL.  
- **Eight losses (overview):**
  - Linear correlation: L<sub>acs</sub>, L<sub>ccs</sub>, L<sub>acf</sub>, L<sub>ccf</sub>
  - Nonlinear dependence (HSIC): L<sub>ads</sub>, L<sub>cds</sub>, L<sub>adf</sub>, L<sub>cdf</sub>
- **Why HSIC?** Captures **nonlinear** sample/feature dependencies; maximizes informative spread (self-HSIC) and cross-view alignment while reducing redundancy.  
- **Results (ResNet-18 where applicable):**
  - Beats VICReg / Barlow Twins / SSL-HSIC / SimCLR on **MNIST, CIFAR-10, CIFAR-100, STL-10** linear eval; strong **nonlinear eval** (MLP) and **domain adaptation**; **ImageNet-100 kNN** trails only DINO by 0.7%.  

---

## Method (1-Minute Read)
CDSSL computes embeddings `z, z'` of two augmented views per image using an **encoder** and **expander**. It then applies:

- **Linear correlation terms** to (i) decorrelate non-matching samples/features and (ii) align corresponding ones.  
- **HSIC dependence terms** to (i) maximize self-HSIC (spread/variance) for samples/features and (ii) maximize cross-HSIC for corresponding pairs while reducing dependence under feature shuffles (anti-redundancy).

All eight terms combine with non-negative weights `λ_*` in the overall loss.

---

## Quick Start
> Minimal template (pseudo-config)

```python
# build encoder/expander -> z, z'
# normalize z, z' to unit length for HSIC terms (empirically helpful)

loss = (λ_acs*L_acs(z,z') + λ_ccs*L_ccs(z,z')
      + λ_acf*L_acf(z,z') + λ_ccf*L_ccf(z,z')
      + λ_ads*L_ads(z)    + λ_cds*L_cds(z,z')
      + λ_adf*L_adf(z)    + λ_cdf*L_cdf(z,z', shuffle_features=True))
loss.backward()



# CDSSL: Self-Supervised Learning Using Nonlinear Dependence

CDSSL is an SSL framework that unifies linear *correlation* objectives (Barlow Twins / VICReg–style) with nonlinear *dependence* via **HSIC** in RKHS. It instantiates **8 complementary losses** across {sample, feature} × {auto, cross} × {linear, nonlinear}, yielding richer and less-redundant representations.

> Source code for the paper *Self-Supervised Learning Using Nonlinear Dependence*. :contentReference[oaicite:0]{index=0}

---

## What’s in this repo

- **Training & configs**
  - `main.py` — main training script for standard datasets. :contentReference[oaicite:1]{index=1}
  - `main_imagenet.py` — ImageNet-100 (or similar) entrypoint. :contentReference[oaicite:2]{index=2}
  - `config.yaml`, `Temp_Config.yaml` — training hyperparams & settings. :contentReference[oaicite:3]{index=3}
  - `environment.yaml` — Conda env spec. :contentReference[oaicite:4]{index=4}
- **Baselines / components**
  - `Custom_vic/` — custom VICReg components. :contentReference[oaicite:5]{index=5}
  - `statistical_HSIC/` — HSIC/RBF-kernel utilities. :contentReference[oaicite:6]{index=6}
  - `utils/` — common helpers (datasets, augmentations, meters, etc.). :contentReference[oaicite:7]{index=7}
- **Evaluation & analysis**
  - `DownStream.py` — linear probe / kNN style downstream eval. :contentReference[oaicite:8]{index=8}
  - `Downstream_clustering.py` — unsupervised clustering eval. :contentReference[oaicite:9]{index=9}
  - `HEatmap.py` — heatmaps (e.g., covariance/HSIC). :contentReference[oaicite:10]{index=10}
- **Other**
  - `Optimize.py`, `SIMClr.py`, `SimCLR2.py`, `test_distributed.py`, `.gitignore`. :contentReference[oaicite:11]{index=11}

> If any of the above scripts expose CLI flags via `argparse`, use them; otherwise set values in `config.yaml`.

---

## Install

Using the provided Conda spec:

```bash
conda env create -f environment.yaml
conda activate cdssl
