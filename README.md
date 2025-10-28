
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
```
---


## Install

Using the provided Conda spec:

```bash
conda env create -f environment.yaml
conda activate cdssl
```

