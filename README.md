# ReguFinder: A Deep Learning Framework for identifying regulators of cell-state transitions through graph-based latent-space perturbation

## Introduction
Cell-state transition regulators are critical for guiding cellular trajectories in complex biological processes, making them high-priority targets in regenerative and preventive medicine. However, conventional methods often neglect intercellular contextual influences and struggle to resolve stage-specific regulatory factors driving sequential cell-state transitions.

To address these limitations, we present **ReguFinder**—a deep learning framework built on a Cell Graph Convolutional Layer-Variational Autoencoder (CGCL-VAE) integrated with a generative adversarial network (GAN). By combining time-stage-specific graph neural network autoencoder modeling, systematic latent dimension perturbation, and decoder-driven prioritization scoring, ReguFinder identifies core regulatory dimensions and their associated master regulators governing cell-state transitions, which we functionally validate to participate in these processes. It further enables the discovery of gene regulatory modules that mediate transitions through coordinated interactions among these regulators.

We benchmarked ReguFinder on five real-world datasets: dentate gyrus development, pancreatic endocrinogenesis, liver development and maturation, hematopoiesis, and COVID-19. In all cases, ReguFinder consistently outperformed conventional methods in accuracy and biological interpretability. ReguFinder thus provides a robust computational framework for prioritizing candidate regulators via systematic in silico perturbation, generating mechanistic hypotheses for downstream experimental validation.

---

## Workflow & Usage

### 1. Preprocessing
First, select highly variable genes to reduce computational burden and annotate your `h5ad` file with a `name.simple` column representing the biological stages of interest.

Run the preprocessing script:
```bash
python preprocessing/data_preprocessing_addSimp.py
```

### 2. Model Training
Initiate training of the CGCL-VAE model using the dataset-specific entry script (example for dentate gyrus):
```bash
python run_dentateGyrus.py
```

### 3. Latent Space Perturbation
1.  Use the classifier script to map cell-type labels to time-stage embeddings and train a cell-type classifier:
    ```bash
    python Emb_Cell_type_Classifier_Dentate_11.py
    ```
2.  Perform systematic perturbation of latent dimensions. This script also computes the difference matrix between perturbed and unperturbed embeddings, then aggregates results across all time stages.

### 4. Regulator Identification
1.  Reconstruct the gene prioritization score matrix:
    ```bash
    python run_GSE132188_resume_argparse.py
    ```
2.  Extract and visualize top regulators for each cell-type transition:
    ```bash
    python run_regus_for_allType_heatmap_fixed.py
    ```

### 5. Downstream Analysis
Reproduce all figures and perform additional biological analyses using scripts in the `downstream_analysis` directory.
