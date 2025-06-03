# 🔍 MotionSight

<div align="center">

<h1>MotionSight: Boosting Fine-Grained Motion Understanding in Multimodal LLMs</h1>

<div style="font-size: 1.2em">
    <a href="https://github.com/natsunoshion">Yipeng Du</a><sup>1*</sup> &nbsp;&nbsp;
    <a href="https://scholar.google.com/citations?user=F11LXvYAAAAJ&hl=en">Tiehan Fan</a><sup>1*</sup> &nbsp;&nbsp;
    <a href="https://scholar.google.com/citations?user=PXmlku8AAAAJ&hl=en">Kepan Nan</a><sup>1,2</sup> &nbsp;&nbsp;
    <a href="https://scholar.google.com/citations?user=Dzr3D_EAAAAJ&hl=en">Rui Xie</a><sup>1,2</sup> &nbsp;&nbsp;
    <a href="https://scholar.google.com/citations?user=yWq1Fd4AAAAJ&hl=zh-CN">Penghao Zhou</a><sup>2</sup> &nbsp;&nbsp;
    <a href="https://implus.github.io/">Xiang Li</a><sup>3</sup> &nbsp;&nbsp;
    <a href="https://scholar.google.com/citations?user=6CIDtZQAAAAJ&&hl=en">Jian Yang</a><sup>1</sup> &nbsp;&nbsp;
    <a href="https://zhenheny.github.io/">Zhengheng Yang</a><sup>2</sup> &nbsp;&nbsp;
    <a href="https://tyshiwo.github.io/">Ying Tai</a><sup>1†</sup> &nbsp;&nbsp;
</div>

<div style="font-size: 1em; margin-top: 10px">
    <sup>1</sup> Nanjing University &nbsp;&nbsp;
    <sup>2</sup> ByteDance &nbsp;&nbsp;
    <sup>3</sup> Nankai University
</div>

<div style="font-size: 0.9em; margin-top: 8px; font-style: italic">
    * Equal contribution. &nbsp;&nbsp; † Corresponding author.
</div>

[![Paper](https://img.shields.io/badge/📝%20Paper-arXiv-red)](https://arxiv.org/abs/2506.01674)
[![Dataset](https://img.shields.io/badge/🤗%20Dataset-Coming%20Soon-blue)](https://github.com/NJU-PCALab/MotionSight)
[![Website](https://img.shields.io/badge/🌐%20Website-Project%20Page-green)](https://nju-pcalab.github.io/projects/MotionSight/)
</div>

Welcome to **MotionSight**, a cutting-edge framework for fine-grained motion understanding. This guide provides instructions for environment setup, model preparation, and evaluation.

---

## 📣 News

- **[2025.06.03]** 🚀 Initial release of MotionSight framework

---

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Model Preparation](#model-preparation)
4. [Evaluation](#evaluation)
5. [MotionVid Examples](#motionvid-examples)
6. [Troubleshooting & FAQ](#troubleshooting--faq)
7. [Citation](#citation)

---

## 🛠️ Prerequisites

- **Operating System:** Linux (Ubuntu 20.04/22.04 recommended)
- **Python:** 3.8 or higher
- **CUDA:** 11.3+ (for GPU acceleration)
- **Hardware:** GPU with at least 24GB VRAM recommended

---

## 🔧 Environment Setup

1. **Clone the Repository**

   ```bash
   git clone https://github.com/NJU-PCALab/MotionSight
   cd MotionSight
   ```

2. **Install Python Dependencies**

   It is highly recommended to use a virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Install Additional Dependencies**

   Some dependencies (e.g., `flash-attn`) may require specific versions. Please refer to `requirements.txt` and ensure compatibility with your CUDA version.

---

## 📦 Model Preparation

1. **Download and Integrate GroundedSAM2**

   - Clone the [GroundedSAM2](https://github.com/IDEA-Research/Grounded-SAM-2) repository:
     ```bash
     git clone https://github.com/IDEA-Research/Grounded-SAM-2
     ```
   - Download all required checkpoints as specified in the GroundedSAM2 documentation.
   - Place the entire `GroundedSAM2` folder (with checkpoints) into the root of the MotionSight project directory, like `MotionSight/Grounded-SAM-2`.

2. **Prepare Tracking Utilities**

   - Move `track_utils.py` into the `GroundedSAM2/` directory:
     ```bash
     mv track_utils.py GroundedSAM2/
     ```
   - Launch the tracking server (adjust `--p` and `--step` as needed for your setup):
     ```bash
     cd GroundedSAM2
     python track_utils.py --p 1 --step 10000
     cd ..
     ```

3. **Prepare Multimodal Large Language Model (MLLM) Checkpoints**

   - Download the MLLM checkpoints (e.g., Qwen2.5-VL-7B-Instruct) and place them in the appropriate directory.
   - You can selectively start the LLM server using [lmdeploy](https://github.com/InternLM/lmdeploy), for example:
     ```bash
     lmdeploy serve api_server '/path/to/Qwen2.5-VL-7B-Instruct' --server-port 23333 --tp 1
     ```
   - Ensure the server is running and accessible at the specified port.

---

## 📊 Evaluation

- To evaluate the results on the MotionBench or FAVOR-Bench benchmark:
    ```bash
    python -m eval.motionsight.eval_motionbench
    python -m eval.motionsight.eval_favorbench
    ```
- Ensure all evaluation datasets and configuration files are properly set up.

---

## 🎬 MotionVid Examples

MotionVid is our specialized module for processing and analyzing fine-grained motion in video content. It provides tools for detailed motion tracking, temporal understanding, and multi-object interaction analysis.

### 📊 Sample Videos and Analysis

Our framework includes several sample videos that demonstrate MotionVid's capabilities:

| Video | Description | Focus Area |
|-------|-------------|------------|
| [📹 pexels_landscape_landscape_7895832_002.mp4](MotionVid/samples/pexels_landscape_landscape_7895832_002.mp4) | Train moving through desert landscape | Object tracking across complex terrain |
| [📹 pixabay_Beach_Sunrise_37084_001.mp4](MotionVid/samples/pixabay_Beach_Sunrise_37084_001.mp4) | Vehicles driving in desert with dust trails | Camera movement and environmental effects |
| [📹 v_JNr0oI927ng_t0.13-5.64.mp4](MotionVid/samples/v_JNr0oI927ng_t0.13-5.64.mp4) | Person on diving board | Subtle human motion analysis |
| [📹 -eq3I7gRqTI_000100_000110.mp4](MotionVid/samples/-eq3I7gRqTI_000100_000110.mp4) | Person mowing lawn with passing vehicle | Multi-object interaction |
| [📹 DKZPW.mp4](MotionVid/samples/DKZPW.mp4) | Person interacting with pet and objects | Complex sequence analysis |

The module includes a [`show.json`](MotionVid/samples/show.json) file that pairs videos with question-answer examples.


### 🎦 Working with Sample Videos

To see these videos in action with MotionSight analysis:

```bash
# Run the MotionBench evaluation pipeline
python -m eval.motionsight.eval_motionbench

# When implemented, you'll be able to process individual videos
# python process_video.py --input MotionVid/samples/DKZPW.mp4 --output results/
```

More detailed examples of video processing will be provided in upcoming documentation.

## ❓ Troubleshooting & FAQ

- **Q:** I encounter CUDA or dependency errors.
  - **A:** Double-check your CUDA version and ensure all dependencies are installed with compatible versions.
- **Q:** The LLM server is not responding.
  - **A:** Verify that the server is running and the port matches the one specified in your scripts.

---

## 📝 Citation

If you use MotionSight in your research, please cite our paper:

```
@misc{du2025motionsightboostingfinegrainedmotion,
      title={MotionSight: Boosting Fine-Grained Motion Understanding in Multimodal LLMs}, 
      author={Yipeng Du and Tiehan Fan and Kepan Nan and Rui Xie and Penghao Zhou and Xiang Li and Jian Yang and Zhenheng Yang and Ying Tai},
      year={2025},
      eprint={2506.01674},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2506.01674}, 
}
```
