# ViT-Empathy

This repository contains code and experiments designed to explore how Vision Transformer (ViT) models perceive and manipulate emotions in facial images. The primary goals of the project are:

1. **Emotion Detection**: Investigate whether ViT models can detect emotions from images and identify which parts of the model or which visual features contribute to that detection.
2. **Activation Patching for Emotion Editing**: Develop and evaluate methods for altering emotional predictions by patching activations within the ViT model. The aim is to understand if and where modifications can change the inferred emotion of an image.

---

## Repository Structure

- `src/` - Source code for datasets, model definitions, ETL processing, analysis, and patching utilities.
- `scripts/` - Utility scripts to run various experiments (feature extraction, patching, CKA analysis, etc.).
- `analysis/`, `data/`, `results/`, `experiments/`, `notebooks/` - Folders containing datasets, experimental outputs, and exploratory notebooks.

Key components in `src/`:
- `dataset/emotion_dataset.py` - Dataset loader for emotion-labeled images.
- `models/vit_backbones.py` - ViT backbone definitions and loading utilities.
- `etl/etl_processing.py` - ETL pipeline for preparing train/val/test splits by person.
- `analysis/` - Tools for linear probe analysis and CKA comparisons.
- `patching/` - Utilities and experiments related to activation patching.

## Getting Started

1. **Setup Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
2. **Prepare Data**
   Place raw emotion image files in `data/` directory, then run ETL pipeline:
   ```bash
   python scripts/run_etl.py --image-dir data/raw --processed-dir data/processed
   ```
3. **Run Experiments**
   Several scripts are available to execute different experiments. Examples:
   ```bash
   python scripts/run_cls_emotion_vector.py    # extract class predictions or embeddings
   python scripts/run_cls_patching_exp.py      # activation patching experiments
   python scripts/run_cka.py                   # compute centered kernel alignment
   ```

   For more details on parameters, check the script docstrings or relevant notebooks.

4. **Analysis and Visualization**
   Work with notebooks under `notebooks/` to visualize results, analyze linear probes, perform CKA analysis, and review patching outcomes.

## Project Goals

- Determine if and how ViT models encode emotional information from facial images.
- Identify which layers, heads, or neurons are responsible for emotion recognition.
- Experimentally modify activations (patching) to change the predicted emotion and evaluate the feasibility of emotion editing.

---

*This repository is intended for research purposes.*
