## Task
Train a UNet model from scratch to generate a **colored polygon image** based on two inputs:
- A **grayscale image** of a polygon (e.g., triangle, square, octagon).
- A **textual color label** (e.g., `"red"`, `"cyan"`, `"yellow"`).

---
## Flow 
<img width="963" height="386" alt="image" src="https://github.com/user-attachments/assets/ff925d74-10f2-4bbf-bb6c-2061e8d66654" />



## Dataset Observations
1. Only **8 distinct colors** are used across the dataset.
2. Each color is **consistently mapped** with no internal variation. The color map is extracted **directly from the dataset** (rather than relying on conventional names).
3. **Output images do not contain black borders**, simplifying the reconstruction process.

## Approach

### Framing the Problem
- Chose a **grayscale-to-RGB** approach over **L-to-AB**.
  - L-to-AB retains black borders in the L channel, requiring extra processing (to remove black border, one will have to flip balck pixels with white during post processing)
  - Grayscale-to-RGB allows simpler manipulation.
- I briefly experimented with L-AB, but postponed further exploration due to time constraints.

## Model Architecture
- A standard **UNet** was used.
- **Color conditioning** was implemented to inject the target color label.
- No architectural changes were made due to the small dataset size.

## Training Strategy

- Initially used **MSELoss**, but observed **slow convergence**.
- Switched to **Binary Cross-Entropy (BCE) Loss**, treating the task as a **per-pixel classification problem**. BCE converged faster and gave **better visual results**.
- A consistent issue was a **grayish halo/dullness** in outputs. Most target colors have **at least one channel with high values** (normalized = 1). So, encouraging model with near 1 preds should improve overall output quality.
- Used a **weighted BCE loss** (`=bce_per_pixel * white_mask + 1.2 * bce_per_pixel * normal_mask`) - to focus on foreground non white pixels. Similar formulation  giving higher weight to near white pixels reduced halo and dullness but color consistency suffered (class imbalance, model was biased to predicted higher 1's due to large no of white pixels). Will revisit this variant.
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


