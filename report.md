## Task
Train a UNet model from scratch to generate a **colored polygon image** based on two inputs:
- A **grayscale image** of a polygon (e.g., triangle, square, octagon).
- A **textual color label** (e.g., `"red"`, `"cyan"`, `"yellow"`).

---
## Flow 
<img width="961" height="360" alt="image" src="https://github.com/user-attachments/assets/f475bf23-47e7-4301-a9c2-16df32e5b0a5" />


## Dataset Observations
1. Only **8 distinct colors** are used across the dataset.
2. Each color is **consistently mapped** with no internal variation. The color map is extracted **directly from the dataset** (rather than relying on conventional names).
3. **Output images do not contain black borders**, simplifying the reconstruction process.

## Approach

### Framing the Problem
- Chose a **grayscale-to-RGB** approach over **L-to-AB**.
  - L-to-AB retains black borders in the L channel, requiring extra processing.
  - Grayscale-to-RGB allows simpler manipulation.
- I briefly experimented with L-AB, but postponed further exploration due to time constraints.

## Model Architecture
- A standard **UNet** was used.
- **Color conditioning** was implemented to inject the target color label.
- No architectural changes were made due to the small dataset size.

## Training Strategy

- Initially used **MSELoss**, but observed **slow convergence**.
- Switched to **Binary Cross-Entropy (BCE) Loss**, treating the task as a **per-pixel classification problem**. BCE converged faster and gave **better visual results**.
- A consistent issue was a **grayish halo/dullness** in outputs. Most target colors have **at least one channel with high values** (normalized = 1).  To resolve this, used a **weighted BCE loss** that penalizes predictions in the range **(0.96, 1)**.This encourages the model to predict **exact color values** (i.e., 1.0) rather than near-whites.
- For fair comparison all models were trained for **500 epochs** using the same hyperparameters: `Adam` optimizer, learning rate `1e-4`, `cosine annealing` scheduler, and `early stopping` with `patience = 20`.


## Postprocessing

- Applied **thresholding** to push all pixel values > `0.9` to `1.0`.
- This removed unwanted gray areas and enhanced image brightness.
- Thresholding was valid since **none of the target colors** fall in the `(0.9, 1.0)` range.

---

## Evaluation

- Did **not** rely on metrics like **PSNR** or **MSE**, due to their poor visual interpretability.
- Instead, used **qualitative visual comparison** between outputs and ground truth.
- A potential future improvement is to use a **CNN-based embedding extractor** to compare **feature similarity** between predicted and ground truth images.

---
## References

- **UNet PyTorch Implementation**: [github.com/kgkgzrtk/cUNet-Pytorch](https://github.com/kgkgzrtk/cUNet-Pytorch)
- **Papers:** - [Let there be color](https://cs231n.stanford.edu/reports/2022/pdfs/109.pdf), [Colorful Image Colorization (ECCV '16)](https://arxiv.org/abs/1603.08511)


