# MachineUnlearning_Facial_Recognition


## 🌟 Project Summary
This project demonstrates the practical application of **Machine Unlearning (MU)** on a pre-trained **ResNet-18** model used for age/gender classification. The goal was to selectively remove (forget) specific data samples from the model's memory without undergoing a full, expensive retraining process, while ensuring the model maintained high utility on its remaining data.

We successfully implemented a **Negative Gradient Ascent** approach and validated the unlearning efficacy using a **Membership Inference Attack (MIA)** and loss distribution analysis.

---

## 🎯 Goal and Methodology

### Goal: Selective Forgetting
To remove the influence of a designated **Forget Set** of image samples from the pre-trained model while preserving high prediction accuracy on the **Retain Set** and **Unseen Set**.

### Methodology: Negative Gradient Ascent
We utilized the Negative Gradient Ascent technique, often referred to as '**Unlearning via Fine-Tuning with Negative Loss**,' to achieve unlearning:

1.  **Initial State**: Loaded a pre-trained ResNet-18 model (fine-tuned for 8 age classes).
2.  **Unlearning**: Iterated over the **Forget Set**, performing **gradient ascent** by **minimizing the negative Cross-Entropy Loss** (i.e., *maximizing the loss*). This effectively pushes the model weights away from the original classification boundary for the forgotten samples.
3.  **Utility Preservation**: The model's remaining parameters were refined based on the gradient ascent step.

---

## 📊 Key Results

The experiment achieved **successful unlearning**, evidenced by an MIA score below the random chance baseline ($\text{MIA} < 0.5$).

### Final Performance Metrics

| Metric | Score | Interpretation |
| :--- | :--- | :--- |
| **Test (Retain) Accuracy** | **0.9884** | **Utility preserved**: Model maintains high accuracy on the retained dataset. |
| **Unseen Accuracy** | **0.9778** | **Generalization maintained**: Model performs well on new, unseen data. |
| **MIA Score** | **0.4698** | **Successful Unlearning**: The model cannot reliably distinguish between forgotten and unseen samples (Random chance is $0.5$). |

### Visual Proof: Loss Distribution Comparison

The most critical evidence of unlearning is the complete overlap of the Cross-Entropy Loss distributions for the two groups. This plot confirms that the model treats **forgotten samples identically to genuinely unseen samples**, successfully mitigating membership leakage risk.

---

## 🔧 Technical Details & Data Pipeline Fix

### Environment
The entire pipeline was successfully executed on a **CPU-only environment** using PyTorch.

### Data Challenge & Solution
The initial code, designed for a full dataset, failed with `FileNotFoundError` when using a small sample.

> **Problem**: Metadata CSV files referenced thousands of image paths that were not present in the small local image folder.
> **Solution**: Implemented **dynamic metadata filtering**. The data pipeline now loads the full CSV index, checks `os.path.exists()` for every file, and only keeps entries corresponding to images physically present on disk. This was followed by a strategic re-split of the surviving data into new Train, Validation, and Unseen sets.

---

## 🚀 Getting Started

### Prerequisites
* Python 3.8+
* PyTorch, Torchvision
* Pandas, Numpy, Scikit-learn

### Setup
1.  **Clone this Repository**:
    ```bash
    git clone [https://github.com/YourUsername/YourProjectName.git](https://github.com/YourUsername/YourProjectName.git)
    cd YourProjectName
    ```
2.  **Place Checkpoint**: Ensure the pre-trained weights (`pre_trained_last_checkpoint_epoch_30.pth`) are accessible in the correct path relative to the notebook.

### Run the Experiment
The entire analysis, from data loading to metric calculation and plot generation, is contained in the main notebook:

* [`Your_Unlearning_Notebook.ipynb`](Your_Unlearning_Notebook.ipynb)
