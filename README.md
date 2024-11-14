# SpectraDiff: Enhancing the Realism and Physical Consistency of Infrared Image Translation with Semantic Guidance

<div align="center">
    <img width="170" alt="image" src="https://github.com/user-attachments/assets/d99f7cd8-305b-48e2-826e-18176717a0ef">
    <img width="170" alt="image" src="https://github.com/user-attachments/assets/1051a2d3-69e4-48a6-8c29-fcf877c15048">
    <img width="170" alt="image" src="https://github.com/user-attachments/assets/08301c77-6808-4cc0-9098-9cfebfd9a658">
    <img width="170" alt="image" src="https://github.com/user-attachments/assets/e1af5af3-6a89-4622-bd93-0eb28b05dc23">
</div>


<div align="center">
    <img width="170" alt="image" src="https://github.com/user-attachments/assets/7b6117f9-e741-441a-abc5-f43ea3655eee">
    <img width="170" alt="image" src="https://github.com/user-attachments/assets/3485269a-2f00-4cca-85eb-c55b71449666">
    <img width="170" alt="image" src="https://github.com/user-attachments/assets/5017de7f-129a-48e2-aa30-983e1f597f86">
    <img width="170" alt="image" src="https://github.com/user-attachments/assets/0125d551-31b0-4fa5-81f9-29f1a5804857">
</div>


## Status
- [x] SpectraDiff Model Pipeline
- [x] Sample Train/Inference
- [x] Metrics (PSNR, SSIM, FID, LPIPS, DISTS)

## Usage

### Environment
```bash
pip install -r requirements.txt
```

### Pre-trained Model & Configuration
| Dataset     | Safetensors & Configuration                            |
|-------------|--------------------------------------------------------|
| FLIR        | [Google Drive](https://drive.google.com/drive/folders/1rgzWU_7Cq0sCOnvjIQWmUsOLJGVCkBIf?usp=drive_link) |
| FMB         | [Google Drive](https://drive.google.com/drive/folders/1Lw1XkmtyNp40zsao_cyM5yu51vKQQbCo?usp=drive_link) |
| MFNet       | [Google Drive](https://drive.google.com/drive/folders/1oUF_zQB5awe43FJqygANn8qsxLJ7obpe?usp=drive_link) |
| RANUS       | [Google Drive](https://drive.google.com/drive/folders/1Jb8zFMZdXNsRLDsZF9D9E9AinuGwiW11?usp=drive_link) |
| IDD-AW      | [Google Drive](https://drive.google.com/drive/folders/18_CqicZZ8YbjNiEP21uRK5gsdzVTIAIP?usp=drive_link) |

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
