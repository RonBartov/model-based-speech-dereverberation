# Model-Based Speech Dereverberation

A deep learning approach for speech dereverberation using neural beamforming, comparing modern model-based techniques to classical signal processing baselines.

## 📌 About the Project

In reverberant environments, speech signals recorded by microphones are significantly degraded due to reflections from surrounding surfaces. Dereverberation is crucial for improving speech clarity and the performance of downstream tasks like speaker identification and ASR.

This project investigates whether model-based deep learning techniques, specifically the Beamformer-Only Speech Dereverberation (BSD) model, can outperform the classical Weighted Prediction Error (WPE) algorithm. BSD is derived from the BSSD architecture [Pfeifenberger & Pernkopf, 2021], isolating the beamforming and dereverberation components into a dedicated neural model.

We perform evaluation using clean mono speech from WSJ0 convolved with simulated Room Impulse Responses (RIRs). The BSD model is trained to minimize SI-SDR loss and is compared to both the full BSSD model and WPE across STOI, SI-SDR, and PESQ metrics.

## Built With

- Python 3.8+
- TensorFlow / Keras
- NumPy & SciPy
- matplotlib & seaborn
- [nara_wpe](https://github.com/fgnt/nara_wpe) – classical dereverberation
- [pystoi](https://github.com/mpariente/pystoi) – STOI evaluation
- [pesq](https://github.com/ludlows/python-pesq) – PESQ evaluation
- Custom RIR simulation using the image-source method

## Getting Started

### Prerequisites

- Python ≥ 3.8
- Git
- Recommended: virtual environment (`venv` or `conda`)

### Installation

```bash
git clone https://github.com/RonBartov/model-based-speech-dereverberation.git
cd model-based-speech-dereverberation
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Running Inference or Training

Example usage:
```bash
python bsd_td.py --mode "train"
python bsd_td.py --mode "valid_with_wpe_for_bsd"
```

Make sure to configure the `config.json` file with the correct dataset paths and settings.

---

## Acknowledgement

This project builds upon the work of several foundational research papers in the field of speech dereverberation:

- Pfeifenberger, L. & Pernkopf, F., *Blind Speech Separation and Dereverberation using Neural Beamforming*, Speech Communication, 2022. https://doi.org/10.1016/j.specom.2022.03.004
- Drude, L. et al., *NARA-WPE: A Python package for weighted prediction error dereverberation*, ITG-Symposium, 2018.
- Taal, C. H. et al., *A short-time objective intelligibility measure for time-frequency weighted noisy speech*, ICASSP, 2010.
- Rix, A. W. et al., *Perceptual evaluation of speech quality (PESQ)*, ICASSP, 2001.
- Le Roux, J. et al., *SDR--half-baked or well done?*, ICASSP, 2019.

© IEEE — All rights reserved to the original authors.  
This repository references their work for academic, research, and educational purposes only.

---

For detailed methodology and results, please refer to our [Final Report (PDF)](./Project_Final_Report-_Model_based_deep_learning.pdf).
