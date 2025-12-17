# Surrogate Stress Prediction in 3D Finite Element Meshes Using Graph Neural Networks

This project explores the application of **Geometric Deep Learning** as a surrogate model for Finite Element Analysis (FEA). The objective is to predict the **Von Mises stress distribution** on complex 3D mechanical meshes, significantly accelerating the simulation process while maintaining high accuracy.

We implement and benchmark **four distinct Graph Neural Network (GNN) architectures**, ranging from multi-scale encoders to equivariant and attention-based models. Each architecture is implemented in a dedicated Jupyter Notebook to facilitate reproducibility and comparative analysis.

---

## 🏗️ Implemented Architectures

This repository contains the implementation of the following state-of-the-art architectures:

### 1. Graph U-Net
* **Type:** Multi-scale Encoder-Decoder.
* **Mechanism:** Adapts the classic U-Net architecture to graph data. It utilizes graph pooling (`gPool`) and unpooling (`gUnpool`) operations to create a bottleneck, capturing both local high-resolution features and global context through hierarchical representations.

### 2. EGNN (E(n) Equivariant Graph Neural Network)
* **Type:** Equivariant GNN.
* **Mechanism:** Explicitly models the network to be equivariant to rotations, translations, and reflections ($E(n)$ symmetry). This ensures that the stress prediction respects the physical geometric laws of the 3D objects regardless of their orientation in space, without requiring data augmentation.

### 3. ClofNet (Coordinate-independent Local Frame Network)
* **Type:** $SE(3)$-Invariant / Local Frame GNN.
* **Mechanism:** Utilizes local coordinate frames constructed on the mesh surface to achieve coordinate independence. This approach is particularly effective at capturing complex local geometric details (anisotropy) in 3D meshes without relying on a global coordinate system.

### 4. Graph Transformer
* **Type:** Attention-based GNN.
* **Mechanism:** Employs **Multi-Head Attention** mechanisms combined with explicit geometric edge features (relative positions and Euclidean distances). This allows the model to capture long-range dependencies and learn which nodes are most relevant to each other, irrespective of their graph distance.

---

## ⚠️ Hardware & Environment Requirements

**This project is specifically optimized for Google Colab Pro**.

Due to the complexity of the 3D meshes and the depth of the networks, the computational requirements are significant.

* **Dataset Size:** The processed dataset is approximately **9 GB** (compressed) and contains roughly 48,000 graphs.
* **Memory:** Loading the graph data and training deep GNNs requires **High-RAM** runtimes (>25 GB System RAM).
* **GPU:** A high-end GPU (A100 or V100) is recommended for reasonable training times.

> **❌ Standard (Free) Google Colab:** The free tier will likely crash due to "Out of Memory" (OOM) errors during data loading or training.
>
> **❌ Local Execution:** Downloading the dataset and running this locally is discouraged unless you possess a workstation with significant VRAM (24GB+) and System RAM (32GB+).

---

## 📂 Repository Structure

The project is organized by architecture. Each notebook is self-contained.

> **⚠️ Note on Data:** The dataset is **NOT** included in this repository due to its size (~9 GB). You must upload the data to your own Google Drive before running the notebooks (see instructions below).

```bash
├── notebooks/
│   ├── 01_GraphUNet.ipynb
│   ├── 02_EGNN.ipynb
│   ├── 03_ClofNet.ipynb
│   └── 04_GraphTransformer.ipynb
└── README.md
```

## ⚙️ Installation & Dependencies

There is no `requirements.txt` file to manage.

Each notebook is self-contained and includes a **setup cell** at the beginning that automatically installs the required environment specific to that architecture.

**Key Libraries Used:**
* `torch` (PyTorch)
* `torch-geometric` (PyG)
* `torch-scatter`, `torch-sparse`
* `plotly` (for 3D visualization - *Already available in Colab environment*)

Simply run the first cell of any notebook to set up the environment.

---

## 💾 Dataset Setup (Critical)

This project utilizes the synthetic dataset provided by **Ezemba et al.**, consisting of approximately **16,000 unique 3D geometries** (tetrahedral meshes).

### 1. Download the Data
The dataset is hosted on **Hugging Face**. You do not need the entire repository, only the pre-processed PyTorch files.

* **Source:** [Hugging Face - cmudrc/SFEM](https://huggingface.co/datasets/cmudrc/SFEM)
* **Files to Download:**
    * `train_data.pt`
    * `val_data.pt`

### 2. Upload to Google Drive
Since the processed dataset is large (approx. **9 GB**) and cannot be easily uploaded to a temporary Colab session, it **must** be hosted on your personal Google Drive.

**⚠️ Exact Folder Structure Required:**
To match the paths defined in the notebooks, please follow these steps exactly:

1.  Go to the **root** of your Google Drive.
2.  Create a folder named: `DataSetML4`
3.  Inside `DataSetML4`, create a subfolder named: `data`
4.  Upload the downloaded `.pt` files into this `data` folder.

**Your final file paths on Drive must look like this:**
* `My Drive/DataSetML4/data/train_data.pt`
* `My Drive/DataSetML4/data/val_data.pt`

> **Note:** The notebooks are configured to mount Drive and look for data at `/content/gdrive/MyDrive/DataSetML4/data/...`. If you use a different folder name, you will need to update the `Config` class in every notebook.

> **Action Required:** Before running any notebook, ensure these files exist in your Drive to avoid `FileNotFoundError`.

---

## 🚀 Usage

1.  **Select an Architecture:** Open the corresponding notebook in the `notebooks/` folder using Google Colab Pro.
2.  **Mount Drive:** Run the initialization cell to mount your Google Drive.
3.  **Run Pipeline:** Execute the cells sequentially. The notebooks handle:
    * Environment Installation.
    * Efficient Data Loading (directly from Drive).
    * Dynamic Feature Computation.
    * Training & Validation loops.
4.  **Inference:** The best model state is saved automatically as `best_model.pt`.

---

## 📊 Results and Discussion

We benchmarked the four architectures on a validation set. The results highlight the superiority of geometric-aware models (EGNN, ClofNet) over standard topological models (U-Net).

### Quantitative Metrics

| Metric | Graph U-Net | EGNN | ClofNet | Transformers |
| :--- | :---: | :---: | :---: | :---: |
| **RMSE** | 0.794 | **0.743** | 0.749 | 0.748 |
| **R² (Median)** | 0.41 | 0.49 | **0.50** | 0.49 |
| **R² (90th percentile)**| 0.74 | 0.79 | **0.81** | 0.80 |
| **R² (10th percentile)**| -0.08 | **0.00** | -0.05 | -0.05 |

* **Best Performance:** **ClofNet** achieves the highest top-tier precision ($R^2_{90\\%}$).
* **Robustness:** **EGNN** demonstrates the best stability on complex geometries (10th percentile), avoiding catastrophic failure ($R^2_{10\\%}=0.0$) unlike the U-Net ($R^2_{10\\%}=-0.08$).
* **Efficiency:** The **Graph Transformer** yields competitive results while offering superior computational efficiency.

### Qualitative Analysis

The project includes an evaluation pipeline using **Plotly** to visualize the stress fields directly on the 3D mesh.

* **Over-smoothing:** All models exhibit a "regression to the mean" behavior, struggling to capture sharp stress singularities in the most complex cases.
* **Visual Comparison:** Predictions are displayed side-by-side with Ground Truth (FEM). Visualization uses **independent color scaling** to emphasize stress distribution patterns (topology) over absolute magnitude errors.

---

## 👤 Authors

* **Alexandre Vallet** - EPFL
* **Gaspard Lafont** - EPFL
* **Tomas Jouven** - EPFL

## 📚 References

* **[1]** J. Ezemba, C. McComb, and C. Tucker, "Neural Network Surrogate Modeling for Stochastic Finite Element Method Using Three-Dimensional Graph Representations," ASME IDETC/CIE, 2023.
* **[2]** H. Gao and S. Ji, "Graph U-Nets," ICML, 2019.
* **[3]** V. G. Satorras et al., "E(n) Equivariant Graph Neural Networks," ICML, 2021.
* **[4]** W. Du et al., "SE(3) Equivariant Graph Neural Networks with Complete Local Frames," ICML, 2022.
