# MotionSight

Welcome to **MotionSight**, a cutting-edge framework for fine-grained motion understanding. This guide provides instructions for environment setup, model preparation, and evaluation.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Model Preparation](#model-preparation)
4. [Evaluation](#evaluation)
5. [Troubleshooting & FAQ](#troubleshooting--faq)
6. [Citation](#citation)

---

## Prerequisites

- **Operating System:** Linux (Ubuntu 20.04/22.04 recommended)
- **Python:** 3.8 or higher
- **CUDA:** 11.3+ (for GPU acceleration)
- **Hardware:** GPU with at least 24GB VRAM recommended

---

## Environment Setup

1. **Clone the Repository**

   ```bash
   git clone https://github.com/your-org/MotionSight.git
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

## Model Preparation

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

## Evaluation

- To evaluate the results on the MotionBench or FAVOR-Bench benchmark:
    ```bash
    python -m eval.motionsight.eval_motionbench
    python -m eval.motionsight.eval_favorbench
    ```
- Ensure all evaluation datasets and configuration files are properly set up.

---

## Troubleshooting & FAQ

- **Q:** I encounter CUDA or dependency errors.
  - **A:** Double-check your CUDA version and ensure all dependencies are installed with compatible versions.
- **Q:** The LLM server is not responding.
  - **A:** Verify that the server is running and the port matches the one specified in your scripts.

---

## Citation

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
