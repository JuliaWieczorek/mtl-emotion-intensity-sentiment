# mtl-emotion-intensity-sentiment
Implementation of a multi-task learning framework for joint modeling of emotion, emotion-conditioned intensity, and sentiment using transformer-based architectures.

> ⚠️ This repository contains research code and may not be fully optimized for production use.

Official implementation of the paper:
**“Multi-Task Aware Learning for Joint Emotion, Intensity, and Sentiment Analysis”**

---

## 📌 Overview

This repository provides the codebase supporting a three-task multi-task learning (MTL) framework for:

* **Multi-label emotion classification**
* **Emotion-conditioned intensity prediction**
* **Sentiment analysis**

The project focuses on modeling complex affective states in conversational data, particularly in **mental health support settings**, where emotions are often implicit, co-occurring, and expressed with varying intensity.

---

## 🧠 Key Contributions

* Joint modeling of **emotion, intensity, and sentiment** in a unified framework
* Explicit formulation of **emotion-conditioned intensity prediction**
* Analysis of **task interactions and negative transfer** in multi-task learning
* Comparison of multiple parameter-sharing strategies:

  * Hard parameter sharing
  * Soft parameter sharing
  * Adapter-based models
  * Mixture-of-Experts (MMoE)
* Integration of **data augmentation and domain adaptation** techniques

---

## 📁 Repository Structure

```
.
├── data/               # Datasets and processed data
├── EMOTIA-DA/          # Data augmentation and domain adaptation
├── EMOTIA-ML/          # Multi-task learning models and training
├── README.md
```

---

## 📊 Datasets

The experiments are based on:

* **MEISD** – multi-label conversational dataset with emotion and intensity annotations
* **ESConv** – mental health support conversations dataset

---

## ⚙️ Methods

The repository includes implementations of:

* Transformer-based encoders (e.g., BERT, RoBERTa)
* Multi-task learning architectures:

  * Hard sharing
  * Soft sharing
  * Adapter-based learning
  * Multi-gate Mixture-of-Experts (MMoE)
* Emotion-conditioned intensity prediction
* Domain adaptation via data augmentation
* Multi-label classification pipelines

---

## 🔁 Reproducibility Notes

Results may vary depending on:

* Random seed
* Data splits
* Model backbone (BERT, RoBERTa, etc.)
* Parameter-sharing strategy
* Augmentation settings

---

## 📚 Citation

If you use this code, please cite:

```bibtex
@article{wieczorek2021multitask,
  title={Multi-Task Aware Learning for Joint Emotion, Intensity, and Sentiment Analysis},
  author={Wieczorek, Julia and Jiang, Xiaorui and Palade, Vasile and Trela, Joanna},
  year={2026}
}
```

---

## 🤝 Acknowledgements

This work explores multi-task learning for affective computing in conversational AI and mental health applications.

---

## 📬 Contact

For questions or collaboration, feel free to reach out.
