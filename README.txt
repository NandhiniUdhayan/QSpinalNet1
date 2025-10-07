# Q-SpinalNet: Hybrid Quantum–Spinal Neural Network for Breast Cancer Detection

This repository contains the **implementation code** for the paper:

> **Q-SpinalNet: A Hybrid Quantum–Spinal Neural Network for Accurate Breast Cancer Detection**  
> (Based on the integration of DQNN and SpinalNet architectures)

---

## 🧠 Model Overview

- **Triple Preprocessing:** CEAMF Denoising, Z-Score Normalization, and Context-Aware Contrast Enhancement  
- **Segmentation:** Swin ResUNet3+–based segmentation backbone  
- **Classification:** DQNN (Quantum-Inspired Deep Neural Network) + SpinalNet hybrid  
- **Dataset:** DDSM / CBIS-DDSM mammogram datasets  

---

## 🧰 Installation

```bash
git clone https://github.com/<your-username>/QSpinalNet_BreastCancer.git
cd QSpinalNet_BreastCancer
pip install -r requirements.txt
