# SpectraDiff: Enhancing the Realism and Physical Consistency of Infrared Image Translation with Semantic Guidance
## Status
- [x] SpectraDiff Model Pipeline
- [x] Train/Inference
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
For segmentation: https://github.com/IDEA-Research/Grounded-Segment-Anything

| Dataset     | URL        |
|-------------|------------|
| FLIR        | [URL](https://www.flir.in/oem/adas/adas-dataset-form/)                   |
| FMB         | [URL](https://arxiv.org/abs/2308.02097)                                  |
| MFNet       | [URL](https://www.mi.t.u-tokyo.ac.jp/static/projects/mil_multispectral/) |
| RANUS       | [URL](https://ieeexplore.ieee.org/document/8279453)                      |
| IDD-AW      | [URL](https://iddaw.github.io/)                                          |

### Set Configuration
The config_base.yaml file contains important settings for both training and testing.

Make sure to review and modify this file if you’re using custom data or want to alter the network structure.

Please refer to the provided configuration file and update the data root in the data section.

### Training
1. Navigate to the UNET/trainer folder
2. Run one of the following commands:
```bash
python train.py
```
or
```bash
accelerate launch train.py
```

### Inference
1. Navigate to the UNET/tester folder
2. Modify the configuration path and safetensors path in test.py
3. Run the following command:
```bash
python test.py
```
