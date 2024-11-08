# SpectraDiff: Enhancing the Realism and Physical Consistency of Infrared Image Translation with Semantic Guidance
## Status
- [x] SpectraDiff Model Pipeline
- [x] Train/Test Process
- [x] Save/Load Training State
- [x] Metrics (PSNR, SSIM, FID, LPIPS, DISTS)
## Usage

### Environment
```bash
pip install -r requirements.txt
```

### Pre-trained Model & Configuration
| Dataset     | Safetensors & Configuration                            |
|-------------|--------------------------------------------------------|
| FLIR        | [Google Drive](https://github.com/anon045/SpectraDiff) |
| FMB         | [Google Drive](https://github.com/anon045/SpectraDiff) |
| MFNet       | [Google Drive](https://github.com/anon045/SpectraDiff) |
| RANUS       | [Google Drive](https://github.com/anon045/SpectraDiff) |
| IDD-AW      | [Google Drive](https://github.com/anon045/SpectraDiff) |

### Data Prepare
| Dataset     | URL        |
|-------------|------------|
| FLIR        | [URL](https://www.flir.in/oem/adas/adas-dataset-form/)                   |
| FMB         | [URL](https://arxiv.org/abs/2308.02097)                                  |
| MFNet       | [URL](https://www.mi.t.u-tokyo.ac.jp/static/projects/mil_multispectral/) |
| RANUS       | [URL](https://ieeexplore.ieee.org/document/8279453)                      |
| IDD-AW      | [URL](https://iddaw.github.io/)                                          |

### Set Configuration

### Training
```bash
python train.py
```
```bash
accelerate launch train.py
```

## Inference
```bash
python test.py
```
