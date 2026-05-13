# Self-Supervised Cascade Network for DAS-VSP Denoising

PyTorch implementation of the three-stage self-supervised cascade network proposed in:

> Jun, H., Kim, S., Lee, M., Cho, Y., Lee, C., Park, K.-G., and Yoon, B.
> *Self-Supervised Cascade Network for Denoising of Distributed Acoustic Sensing Vertical Seismic Profile Data.*
> Submitted to *Geophysical Journal International* (under review).

| Stage | Module | Architecture | Target noise |
|:----:|:--------|:--------------|:--------------|
| 1 | `model_r` | Autoencoder (`Autoencoder_r`) | Random noise |
| 2 | `model_l` | Residual DnCNN (`DnCNN`)      | Linear / coupling noise |
| 3 | `model_d` | Autoencoder (`Autoencoder_d`) | Correlated multi-channel noise (CMN) |

Stages 1 and 3 follow a Noise2Noise-style scheme; Stage 2 injects separated linear-coupling noise into a Stage-1 output and learns to recover it.

## Files

- `DAS_CascadeNet_3noise_torch.ipynb` — training (3-stage cascade)
- `Apply_CascadeNet_torch.ipynb` — inference on a full DAS-VSP gather

## Requirements

Python 3.8, PyTorch 2.1 + CUDA 11.8, torchvision, numpy, matplotlib, tqdm. CPU works (set `CUDA_VISIBLE_DEVICES=""`); GPU recommended (trained on RTX A5000).

## Data layout

All inputs are `float32` raw binaries. File lists are matched by `glob` and `sorted()`, so any zero-padded naming works as long as the three training directories stay aligned element-wise.

**Training** — three sibling directories of paired `128 × 128` patches:

```
./data/train/data1/   patch01_*.bin (input), patch02_*.bin (Noise2Noise target)
./data/train/data2/   noise_*.bin   (separated linear-coupling noise)
./data/train/data3/   patch01_*.bin (Stage-1 inputs without linear-coupling noise)
./data/test/{data1,data2,data3}/   # same structure for evaluation
```

**Inference** — one directory of full gathers shaped `n2 × n1` (default `3760 × 2000`):

```
./data/apply/   *.bin
```

Update `data*_dir`, `test*_dir`, `data_dir_apply`, and `n1, n2` in the second code cell of each notebook.

## Usage

Run the notebooks in order:

```bash
jupyter notebook DAS_CascadeNet_3noise_torch.ipynb   # train
jupyter notebook Apply_CascadeNet_torch.ipynb        # apply
```

`model_r` updates every epoch, `model_l` starts at epoch 5, and `model_d` starts at epoch 10. Checkpoints are written to `./models/model_first/` (`best_*.pth.tar` and `model_*_<epoch>.pth.tar`); inference outputs (`input`, `pred_r`, `pred_l`, `pred_d`, `label`) are written to `./apply_out/` as `float32` binaries with shape `(n2, n1)`.

## Reproducibility note

The DAS-VSP field datasets acquired at Southern East region of Korea test sites are subject to institutional data-sharing restrictions and cannot be released with this repository. The code, hyper-parameters, and architecture are provided in full so that the method can be reproduced on any DAS-VSP dataset organised as described above. Pre-trained checkpoints and relevent DAS-VSP data may be made available upon reasonable request to the corresponding author.

## Citation

```bibtex
@article{Jun2026DASDSCNet,
  author  = {Jun, Hyunggu and Kim, Sujeong and Lee, Myunghun and Cho, Yongchae and
             Lee, Changhyun and Park, Kwon-Gyu and Yoon, Byoungjoon},
  title   = {Self-Supervised Cascade Network for Denoising of Distributed Acoustic
             Sensing Vertical Seismic Profile Data},
  journal = {Geophysical Journal International},
  year    = {2026},
  note    = {Under review}
}
```

## Contact

- Hyunggu Jun (Kyungpook National University) — hgjun@knu.ac.kr
- Byoungjoon Yoon (KIGAM, corresponding author) — yoonstation@kigam.re.kr
