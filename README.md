# FedStNN: Federated Learning on Stochastic Neural Networks

Official code for the paper **"Federated Learning on Stochastic Neural Networks"**
Jingqiao Tang, Ryan Bausback, Feng Bao, Richard Archibald
[arXiv:2506.08169](https://arxiv.org/abs/2506.08169) · *Journal of Machine Learning for Modeling and Computing*, **6**(4), 125–150 (2025) · [DOI:10.1615/JMachLearnModelComput.2025060047](https://doi.org/10.1615/JMachLearnModelComput.2025060047)

## Overview

Federated learning keeps client data on-device, but that same design makes it
vulnerable to latent noise in local datasets — measurement error, human error,
and instrument limitations that never get corrected centrally. **FedStNN**
addresses this by using a **Stochastic Neural Network (SNN)** as each client's
local model.

An SNN reformulates the network as a stochastic differential equation and solves
it via the Stochastic Maximum Principle. It contains two internal networks:

- a **drift network** that recovers the true underlying signal, and
- a **diffusion network** that quantifies the latent noise.

Because all SNN parameters are deterministic (unlike Bayesian Neural Networks,
which treat parameters as random variables), the drift and diffusion networks can
be aggregated separately with standard federated averaging. The result is a
federated method that not only predicts the underlying function but also
reproduces the noise band — even under non-IID client data.

## Repository structure

| File | Description |
| --- | --- |
| `FedSNN_1D_Function.ipynb` | 1D experiment: approximates `f(x) = sin(x)` perturbed by Gaussian noise, distributes data across clients, and runs FedStNN to recover both the function and the noise. |
| `FedSNN_1D_global.pth` | Trained global model checkpoint for the 1D experiment. |
| `FedSNN_2d_Function.ipynb` | 2D experiment: approximates a 2D piecewise function with Gaussian noise under a non-IID client split. |
| `FedSNN_2D_Function.pth` | Trained global model checkpoint for the 2D function experiment. |
| `FedSNN_2d_Image.ipynb` | 2D image experiment: reconstructs an image of letters, with each letter's pixels assigned to clients as local data. |
| `FedSNN_2d_Image_All_Letters.ipynb` | Full multi-letter image reconstruction via FedStNN. |
| `FedSNN_2d_Image_Single_Letter.ipynb`, `FedSNN_2d_Image_Single_Letter2.ipynb` | Single-letter variants of the image experiment. |
| `FedSNN_2d_Image_new_1.ipynb`, `FedSNN_2d_Image_All_Letters-Copy1.ipynb` | Additional / working versions of the image experiments. |

## Experiments

### 1. 1D function approximation
Approximates `f(x) = sin(x)` on `[0, 2π]` from noisy observations
`yᵢ = f(xᵢ) + N(0, 0.1²)`. Uses 100 clients (10% sampled per round), each with a
4-residual-block SNN (`8×16×8` per block). Both IID and non-IID client
distributions are tested. The global model recovers the sine curve *and* the noise
bandwidth.

### 2. 2D function approximation
Approximates a piecewise function on `[-1,1]²` under a non-IID split where each of
4 client groups focuses on one quadrant. SNN uses 4 residual blocks (`16×32×16`).
The global model captures the true surface while keeping predictions within a
two-standard-deviation band of the observation noise.

### 3. 2D image learning
Learns a black-and-white letter image (FSU logo) by mapping pixel position `(i, j)`
to color. The image is split so each of 3 client groups focuses on one letter
(F, S, U). No single client can reconstruct the full image, but FedStNN recovers
all three letters collaboratively without sharing raw pixels.

## Requirements

- Python 3.x
- PyTorch
- NumPy
- Matplotlib
- Jupyter

```bash
pip install torch numpy matplotlib jupyter
```

## Usage

Open any notebook in Jupyter and run all cells top to bottom.

The 1D and 2D function experiments are fully reproducible by regenerating the
synthetic data. Pretrained global checkpoints (`.pth`) are provided for the 1D and
2D function experiments.

## Data availability

The 1D and 2D function datasets are reproducible by defining the same functions.
The image used in Experiment 3 is the FSU logo, converted to black and white.

## Citation

```bibtex
@article{Tang_2025,
  author  = {Jingqiao Tang and Ryan Bausback and Feng Bao and Richard Archibald},
  title   = {Federated Learning on Stochastic Neural Networks},
  journal = {Journal of Machine Learning for Modeling and Computing},
  issn    = {2689-3967},
  year    = {2025},
  volume  = {6},
  number  = {4},
  pages   = {125--150},
  doi     = {10.1615/JMachLearnModelComput.2025060047},
  url     = {https://dl.begellhouse.com/journals/558048804a15188a,1c7e14936ceba523,7d79b6394969fcf3.html}
}
```
