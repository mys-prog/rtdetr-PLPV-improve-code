# rtdetr-PLPV-improve-code

# PL-DETR: An Efficient Framework for Defect Detection in Solar Cell Photoluminescence Images

This repository provides the implementation of **PL-DETR**, an improved real-time object detection framework based on RT-DETR for defect detection in solar cell photoluminescence (PL) images.

PL-DETR aims to improve defect detection accuracy while maintaining computational efficiency. The proposed framework incorporates three key modules to enhance multi-scale feature representation and defect detection performance.

## 1. Overview

The proposed PL-DETR introduces the following modules:

- **ACFP (Adaptive Cross-scale Feature Pyramid):** Enhances cross-scale feature interaction and integrates spatial and semantic information to improve feature representation.
- **GHAB (Ghost-based Hierarchical Aggregation Block):** Reduces computational redundancy while preserving fine-grained defect representations.
- **MSLA (Multi-Scale Linear Attention):** Captures multi-scale local structures and long-range contextual dependencies.

These components are integrated into the RT-DETR architecture to improve its ability to detect various defects in solar cell PL images.

## 2. Project Structure

```text
PL-DETR/
│
├── config/
│   ├── PL-DETR.yaml
│   └── PVPLdata.yaml
│
├── modules/
│   ├── ACFP.py
│   ├── GHAB.py
│   └── MSLA.py
│
├── tools/
│   ├── train.py
│   └── val.py
│
├── README.md
├── requirements.txt
```

## 3. Installation

Clone this repository:

```bash
git clone https://github.com/mys-prog/rtdetr-PLPV-improve-code.git
cd PL-DETR
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Please ensure that the installed PyTorch and CUDA versions are compatible with your hardware and environment.

## 4. Dataset Preparation

The dataset used in this study consists of nine defect categories in solar cell PL images.

Update the dataset paths and class information in:

```text
config/mydata.yaml
```

### Note on Class Labels

**The defect category names used in the paper and the code are not exactly the same.** To avoid ambiguity, the correspondence between the class names in the manuscript and those used in the implementation is provided below.

Please refer to this table when interpreting the experimental results and reproducing the experiments.

| Class ID | Class Name in the Paper | Class Name in the Code |
|:---:|:---|:---|
| 0 | clouding-defect | mistdefect |
| 1 | black-spot | blackspot |
| 2 | clustered-spot | polymistdefect |
| 3 | dark-corner | darkcorner |
| 4 | low-lifetime-Ⅰ | rectlowcell |
| 5 | low-lifetime-Ⅱ | fulllowcell |
| 6 | ring-artifact | concentriccircle |
| 7 | cross-mark | crossmark |
| 8 | bright-spot | whitespot |

*Note: The class IDs follow the annotation order defined in the dataset configuration file.*

## 5. Model Configuration

The model architecture is defined in:

```text
config/PL-DETR.yaml
```

The dataset configuration is defined in:

```text
config/PVPLdata.yaml
```

Before training or evaluation, please check the dataset paths, number of classes, and other configuration parameters.

## 6. Training

To train PL-DETR, run:

```bash
python tools/train.py
```

Please refer to the training script for additional configuration options.

## 7. Evaluation

To evaluate the trained model, run:

```bash
python tools/val.py
```

The evaluation process reports the detection performance on the specified dataset.

## 8. Pre-trained Weights

The trained model weights are available in the **GitHub Releases** section.

Two pre-trained weight files (`best.pt`) are provided for downloading.

**Download:**

[GitHub Releases](https://github.com/your-username/PL-DETR/releases)

Please download the corresponding checkpoint from the Releases page and place it in the appropriate directory before evaluation or inference.


## 9. Acknowledgements

This work is based on the RT-DETR framework.

We sincerely acknowledge the authors of RT-DETR and the open-source community for their valuable contributions.


## 10. Contact

For questions or further information, please feel free to contact the corresponding author:Y30240774@mail.ecust.edu.cn.

