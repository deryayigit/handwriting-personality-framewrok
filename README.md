# Handwriting-Based Personality Prediction

Handwriting is an indicator of personality traits represented by neurological patterns in the brain. In other words, our brain and subconscious actually shape our character as a result of our habits. It is, therefore, a unique form of biometric data that contains various clues regarding an individual’s cognitive processes, motor skills and personality traits.  

Studies linking handwriting to personality analysis are examined under the science of graphology. Graphology is a field that investigates possible relationships between handwriting characteristics and personality traits. The subject of personality analysis through handwriting is seen to be utilised in various contexts, such as criminal investigations, human resources (the recruitment process), psychological counselling and guidance services, and education. 

The aim of this project is to predict personality traits within the framework of the Five-Factor Personality Model (OCEAN) using handwriting images. As part of the study, a deep learning model based on the Vision Transformer (ViT) was developed, with the aim of predicting the dominant personality trait using a classification approach. To enhance model performance, comparative experiments were conducted by applying various datasets, data splitting strategies (Train-Test Split, Train-Validation-Test Split and Stratified K-Fold Cross Validation), data balancing methods (Class Weight, Weighted Random Sampler) and data augmentation techniques. Furthermore, the effects of preprocessing methods aimed at image enhancement on model performance were investigated. To facilitate the analysis of the results, a graphical user interface was developed that allows for the visualisation of experiment management, model comparisons, personality predictions and performance metrics. 

## Repository Structure

```text
handwriting-personality-framework/
│
├── datasets/
│   ├── ds1/
│   ├── ds1_square/
│   ├── ds2/
│   ├── ds3/
│   ├── ds4_regression/
│   ├── ds5/
│   └── hienwrite/
│
├── gui/
│   ├── app.py
│   └── controller.py
│
├── preprocessing_module/
│   ├── image_enhancer.cpp
│   ├── create_ds5.py
│   ├── create_square_dataset.py
│   └── prepare_ds4.py
│
├── runs/
│   ├── EXP-001/
│   ├── EXP-002/
│   ├── EXP-003/
│   └── ...
│
├── src/
│   ├── classification/
│   ├── regression/
│   └── common/
│
├── main.py
├── dataset_registry.json
├── experiment_registry.json
└── requirements.txt
```

---

## System Overview

The system follows a straightforward pipeline from input to prediction. The user uploads a handwriting sample image through the GUI, which is resized and normalized to match the input requirements of the Vision Transformer architecture (224×224×3). The image is then divided into 16×16 patches and converted into numerical embeddings that the model can process.

The selected ViT model pretrained on ImageNet and fine-tuned on the handwriting dataset analyzes the input and predicts the dominant personality trait. The prediction results are displayed in the GUI alongside a confusion matrix summarizing the model's overall performance, and the analyzed handwriting sample is shown for reference.

## Training Strategy & Experiments

Multiple training strategies and data handling techniques were explored to identify the most effective configuration. These include class weighting, weighted random sampling, strong data augmentation, and preprocessing based image enhancement, each evaluated and compared across separate experiments.

The best-performing configuration—based on accuracy and Macro-F1 score—was selected as the default model used for handwriting analysis in the GUI. All experiment results remain accessible through the interface, allowing users to compare different training strategies and their effects on model performance.

<p align="center">
  <img src="https://github.com/user-attachments/assets/96d3d7f0-52e3-4f9b-8c1f-8c5c29dd1a29" width="650">
</p>

<p align="center">
  <b>General Flowchart</b>
</p>

---
 
## Data Collection Stages and Data Splitting Strategies

This project utilizes publicly available handwriting datasets collected from open-source repositories (see Dataset Sources below). The datasets are labeled according to the Big Five (OCEAN) personality model, with handwriting samples associated with personality scores obtained through the IPIP questionnaire.

To investigate model performance under different experimental settings, three data splitting strategies were applied across the experiments:

 **Train-Test Split** a simple 80/20 split used for initial baseline experiments.
 **Train-Validation-Test Split**  a three-way split (typically 60–80% train, 10–20% validation, 10–20% test) used for larger datasets to monitor performance during training.
 **Stratified K-Fold Cross Validation (k=5)**  used on smaller datasets to obtain more reliable performance estimates while preserving class distribution across folds.

---

### Model Architecture

The Vision Transformer (ViT) architecture, introduced by Dosovitskiy et al. in "An Image is Worth 16×16 Words" (ICLR 2021), processes an image as a sequence of fixed-size patches rather than individual pixels. This property is particularly well suited for handwriting analysis: what matters for personality prediction isn't the shape of a single letter, but holistic patterns such as line spacing, word spacing, and overall page layout. By modeling relationships between patches through multi-head self-attention, ViT can learn these global, style-level features directly from the data—without hand-crafted rules.

**Pipeline:**

**Resizing** — Input images are resized to 224×224×3 (RGB), matching the ImageNet-pretrained ViT's expected input.
**Patch Embedding** — The image is split into 196 patches of 16×16 pixels (224/16 = 14 per side, 14×14 = 196). Each patch (16×16×3 = 768 values) is projected into a fixed-size embedding via a linear layer.
**CLS Token** — A learnable classification token is prepended to the 196 patch embeddings, resulting in 197 tokens total. This token aggregates information from all patches through the encoder.
**Positional Embedding** — Since Transformers have no inherent sense of spatial position, a learnable positional embedding is added to each token to preserve layout information.
**Transformer Encoder** — A stack of encoder blocks, each consisting of Layer Normalization, Multi-Head Self-Attention (MSA), residual connections, and an MLP layer:

```math
Attention(Q,K,V)=Softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V
```

  Unlike CNNs, which build hierarchical features through local filters (e.g., 3×3, 5×5), ViT evaluates all patches simultaneously from the start—allowing it to capture long-range relationships, such as consistency between line slant at the top of a page and writing density at the bottom.

- **Classification Head** — The CLS token's final representation is passed through an MLP head to produce logits for the five OCEAN classes, converted to probabilities via Softmax:

```math
\sigma(z)_i=\frac{e^{z_i}}{\sum_j e^{z_j}}
```

**Two-Stage Fine-Tuning:** Given the limited dataset size, the model is fine-tuned in two stages to avoid overfitting and preserve pretrained representations:

1. **Linear Probing** — The backbone is frozen; only the classification head is trained.
2. **Full Fine-Tuning** — The entire model is unfrozen and fine-tuned with a low learning rate; the checkpoint with the best Macro-F1 score is saved.

Resizing: The input image is resized to 224 × 224 × 3 (RGB), which is the expected input size of the ViT model pretrained on ImageNet. 

Patch Embedding: The image is divided into 196 patches of size 16 × 16 pixels. Each patch (16×16×3 = 768 dimensions) is transformed into a fixed-size embedding vector through linear projection. 

CLS Token: A special [CLS] token used for classification is prepended to the sequence of 196 patch tokens, resulting in a total of 197 tokens. 

Positional Embedding: A learnable positional embedding vector is added to each token to preserve spatial information. 

Transformer Encoder: Consists of multiple encoder blocks, each containing Layer Normalization, Multi-Head Self-Attention (MSA), residual connections, and MLP layers. 


```math
Attention(Q,K,V)=Softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V
```

Classification Layer: The [CLS] token from the encoder output is passed through the MLP head to generate logits corresponding to five classes, which are then converted into a probability distribution using Softmax. 

```math
\sigma(z)_i=\frac{e^{z_i}}{\sum_j e^{z_j}}
```

<p align="center">
    <img src="https://github.com/user-attachments/assets/1e5c4733-d90f-4caf-a50b-634523aed69c" width="850">
</p>

<p align="center">
  <b>Vision Transformer (ViT) Architecture Overview</b>
</p>

---

## Code Implementation

ViT Two-Stage Fine-Tuning Strategy 

Due to the limited size of the dataset, directly fine-tuning all ImageNet pretrained weights increases the risk of overfitting. To reduce this risk, a two-stage fine-tuning strategy was applied: 

Stage 1 — Linear Probing: The backbone (Transformer encoder) is frozen, and only the classification head is trained. 

Stage 2 — Full Fine-Tuning: The backbone is unfrozen and the entire model is fine-tuned with a low learning rate; the checkpoint with the best Macro-F1 score is saved. 

#### ViT Pseudocode 

```text

Algorithm: Two-Stage Fine-Tuning Strategy 
 
Input: 
 
D          → Training dataset 
 
M_pre      → ImageNet pretrained ViT model 
 
θ_backbone → Transformer backbone parameters 
 
θ_head     → Classification head parameters 
 
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 
 
STAGE 1 — Linear Probing (Head Training) 
 
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 
 
1. Load the pretrained ViT model: 
 
 M ← M_pre 
 
2. Freeze backbone parameters: 
 
     θ_backbone.requires_grad ← False 
 
3. Optimize only the classification head parameters: 
 
     Optimize(θ_head) 
 
4. Run head training epochs: 
 
     for epoch = 1 → E_head do 
 
           logits ← M(x) 
 
           loss ← CrossEntropyLoss(logits, y) 
 
           Backpropagation(loss) 
 
           θ_head ← Update(θ_head) 
 
     end for 
 
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 
 
STAGE 2 — Full Fine-Tuning 
 
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 
 
5. Unfreeze backbone parameters: 
 
     θ_backbone.requires_grad ← True 
 
6. Optimize all model parameters with a low learning rate: 
 
     Optimize(θ_backbone ∪ θ_head) 
 
7. Run fine-tuning epochs: 
 
     for epoch = 1 → E_finetune do 
 
           logits ← M(x) 
 
           loss ← CrossEntropyLoss(logits, y) 
 
           Backpropagation(loss) 
 
           θ ← Update(θ) 
 
           Calculate validation metrics: 
 
                 - Accuracy 
                 - Macro-F1 
                 - Loss 
 
     end for 
 
8. Save the model with the best validation score: 
 
     M_best ← argmax(Macro-F1) 
 
Output: 
 
M_best → Fine-tuned Vision Transformer model

```

In the system design, Python was selected as the programming language due to its ease of use for vectorized operations, readability, open-source nature, extensive ecosystem support, and ability to accelerate development through rapid prototyping. In addition, Python is widely used in the modern Deep Learning ecosystem, making it a suitable choice for this project. 

Since Python is a runtime language, excessive use of for-loops can reduce performance and efficiency. This becomes particularly significant in matrix operations commonly used in Deep Learning. This issue is addressed through vectorization provided by libraries, resulting in a more efficient system. 

Visual Studio Code (Windows ecosystem) was used as the development environment (IDE). PyTorch, torchvision, timm, NumPy, scikit-learn, and Pillow were selected as the primary libraries. For GUI development, PyQt6, the modern Python-compatible version of the Qt framework, was used. 

The software architecture was designed in a modular manner. Within the scope of the project, the data layer, model training layer, and user interface layer were separated to create a maintainable and extensible system architecture. This structure enables easy integration of different datasets and training strategies. The codebase generally consists of data preparation, model training, evaluation, preprocessing, inference, and graphical user interface components. 

Each Python module is responsible for a specific task. The application is executed through a single entry point (GUI module), where model loading, inference, and user interactions are managed. The GUI does not perform model training; instead, it presents the outputs generated during training. 

The outputs of training are stored in the runs/ directory. A separate subfolder (e.g., EXP-001) is created for each training experiment. Config.yaml stores training configurations. Metrics.json stores performance metrics for each epoch. Model.pt contains the best saved model. Label.json contains class labels and their corresponding mappings. Splits.json stores the file paths of images allocated to training and validation sets. 

main.py is the main execution file of the project. Core operations such as training, evaluation, and launching the graphical user interface are managed through this file. 

dataset_registry.json and experiment_registry.json are registry files that define the datasets and experiment configurations used in the project. The dataset, training strategy, and parameter configuration for each experiment are determined through these files. 

The src/classification/ directory contains the training, model, and evaluation code used for classification-based personality prediction. Training the Vision Transformer model, saving checkpoints, and calculating test/validation metrics are performed in this section. 

The src/regression/ directory contains regression experiments aimed at predicting personality scores as continuous values. 

The src/common/ directory contains shared code used across multiple modules, including dataset loading, experiment registry management, utility functions, and preprocessing integrations. 

preprocessing_module/image_enhancer.cpp is a C++-based preprocessing module developed to apply image enhancement techniques to handwriting images. Operations such as grayscale conversion and background cleaning are implemented in a manner similar to the CLAHE algorithm. 

The runs/ directory contains trained model weights, metric files, confusion matrix results, and temporary processed images generated during GUI execution. 

The datasets/ directory contains the datasets used for classification and regression experiments. 

---

## Key Observations  

Within the scope of this study, open-source datasets were utilized because collecting and labeling handwriting data requires considerable time, cost, and expert involvement. The experimental results showed that the Vision Transformer architecture was able to learn specific visual patterns from handwriting images that may be associated with personality traits. In addition, data balancing strategies, data augmentation techniques, and the K-Fold cross-validation approach were found to have a significant impact on model performance. While image enhancement methods improved performance in certain experiments, variations in model predictions were also observed across different handwriting samples from the same individual. This finding highlights both the complexity of handwriting-based personality prediction and the importance of dataset diversity when developing deep learning models for this task.

---

## Dataset Sources

1.	Personality Prediction Using Handwriting Images Dataset
Chaubey, G. Personality Prediction Using Handwriting Images Dataset. Kaggle.

https://www.kaggle.com/datasets/gyanendrachaubey/personality-prediction-using-handwriting-images

2.	Big Five Personality Computer Vision Dataset
Big Five Personality Computer Vision Dataset. Roboflow Universe.

https://universe.roboflow.com/tugas-j4mkm/big-five-personality-5hbsp

3.	HiEnWrite Dataset
Checker, S. HiEnWrite Dataset.

https://github.com/sakshamchecker/HiEnWrite-Dataset

---

## References

1.	Detection of Personality Features From Handwriting By Machine Learning Methods
Müsevitoğlu, H., Öztürk, A., & Başünal, F. N. (2023). *Detection of Personality Features From Handwriting By Machine Learning Methods*. Gazi Journal of Engineering Sciences, 9(2), 200–212.

2.	IPIP Big-Five Factor Markers
Open-Source Psychometrics Project. *IPIP Big-Five Factor Markers*.

https://openpsychometrics.org/tests/IPIP-BFFM/

3.	An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale
Dosovitskiy, A., et al. (2021). *An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale*. ICLR.

4.	Attention Is All You Need
Vaswani, A., et al. (2017). *Attention Is All You Need*. NeurIPS.

5.	A Survey on Vision Transformers
Khan, S., et al. (2022). *A Survey on Vision Transformers*. ACM Computing Surveys, 54(10s), 1–41.

6.	Transfer Learning, Fine-Tuning and Hyperparameter Tuning
Development Seed. *Transfer Learning, Fine-Tuning and Hyperparameter Tuning*.

https://developmentseed.org/tensorflow-eo-training-2/docs/Lesson7c_transfer_learning_hyperparam_opt.html

7.	Neural Networks and Deep Learning
Nielsen, M. *Neural Networks and Deep Learning*.

http://neuralnetworksanddeeplearning.com

8.	 Artificial Neural Systems
Zurada, J. M. (1992). *Artificial Neural Systems*. West Publishing Company.
