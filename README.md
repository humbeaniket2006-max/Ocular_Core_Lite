# Ocular-Core (Lite): Edge-Optimized Synthetic Biological Texture Generation

![Status](https://img.shields.io/badge/Status-Research%20Prototype-blue)
![Hardware](https://img.shields.io/badge/Hardware-Intel%20i3%20CPU-orange)
![Optimization](https://img.shields.io/badge/Framework-OpenVINO-green)

## 1. Project Overview
**Ocular-Core (Lite)** is a research proof-of-concept designed to democratize high-fidelity biological data generation. While standard generative pipelines typically require high-end GPUs (NVIDIA CUDA or Apple Silicon MPS), this project demonstrates that complex synthetic biology workflows can be executed on constrained **Intel x86_64 CPUs** (like the MacBook Air Core i3) without sacrificing geometric accuracy.

This pipeline utilizes **Latent Consistency Models (LCM)** and **Intel OpenVINO™** quantization to generate macro-scale human iris textures and automatically extract 3D surface normals, all within an 8GB RAM envelope.

### Key Innovations
* **Edge-Native Inference:** Runs Stable Diffusion v1.5 on a dual-core CPU using FP16 quantization.
* **Thermal Management Protocol:** Implements algorithmic "cool-down" cycles to prevent thermal throttling on passive-cooled hardware.
* **3D Surface Recovery:** Automates the extraction of Normal Maps using lightweight Monocular Depth Estimation (`dpt-hybrid-midas`).

---

## 2. Visual Results

| Dilated Pupil (Stress Test) | Constricted Pupil (Stress Test) | 3D Normal Map |
| :---: | :---: | :---: |
| ![Dilated](experiments/stress_test_0.png) | ![Constricted](experiments/stress_test_1.png) | ![Normal Map](experiments/normal_map_0.png) |
| *Prompt: "Pupil fully dilated"* | *Prompt: "Pupil constricted"* | *Derived from Depth Map* |

---

## 3. Technical Architecture

| Component | Technology | Purpose |
| :--- | :--- | :--- |
| **Generative Engine** | Stable Diffusion v1.5 | Base latent diffusion model for texture synthesis. |
| **Inference Runtime** | Intel OpenVINO™ | Compiles neural network layers for x86_64 CPU execution. |
| **Depth Estimation** | Intel DPT Hybrid Midas | Lightweight model for extracting depth data. |
| **Orchestration** | Python 3.11 + Diffusers | Pipeline logic and image processing. |

### Hardware Constraints Tested
* **Device:** MacBook Air (2020)
* **Processor:** Intel Core i3 (1.1 GHz Dual-Core)
* **Memory:** 8 GB 3733 MHz LPDDR4X
* **Graphics:** Intel Iris Plus (Integrated) - *Not used, pure CPU inference.*

---

## 4. Installation & Setup

To replicate this environment, you must use specific library versions to avoid conflicts between NumPy 2.0 and PyTorch on macOS.

**Prerequisites:** Python 3.11

```bash
# 1. Clone the repository
git clone https://github.com/humbeaniket2006-max/Ocular_Core_Lite.git
cd Ocular_Core_Lite

# 2. Create a virtual environment (Recommended)
python3 -m venv venv_ocular_lite
source venv_ocular_lite/bin/activate  # On Windows use: venv_ocular_lite\Scripts\activate

# 3. Install Optimized Dependencies
# Note: These versions are pinned to prevent conflicts (Dependency Hell).
pip install --upgrade pip
pip install "torch==2.2.2" "optimum==1.18.0" "optimum-intel==1.16.0" "diffusers==0.27.2" "numpy<2.0" "huggingface_hub==0.23.0" "opencv-python-headless" "pillow" "scipy"

# 4. Install the library in editable mode
pip install -e .
