# Handwriting Personality

## Project Scope  

Handwriting is an indicator of personality traits represented by neurological patterns in the brain. In other words, our brain and subconscious shape our character as a result of our habits. Therefore, handwriting is a unique biometric trait that contains various clues about an individual's cognitive processes, motor skills, and personality characteristics. 

Studies associating handwriting with personality analysis are examined under the field of Graphology. Graphology is the field of study that encompasses inferences about personality and character based on a person's handwriting. Personality analysis through handwriting has been used in various areas such as forensic investigations, human resources (recruitment processes), psychological counseling and guidance services, and education. 

In this project, the aim is to predict personality traits from handwriting images within the scope of the Five-Factor Personality Model (OCEAN). A Vision Transformer (ViT)-based deep learning model was developed, and a classification approach was employed to predict the dominant personality trait. To improve model performance, comparative experiments were conducted using different datasets, data splitting strategies (Train-Test Split, Train-Validation-Test Split, and Stratified K-Fold Cross Validation), data balancing methods (Class Weight, Weighted Random Sampler), and data augmentation techniques. In addition, the effects of CLAHE-based image enhancement and various preprocessing methods on model performance were investigated. To analyze the obtained results, a PyQt-based graphical user interface was developed, enabling experiment management, model comparisons, personality predictions, and visualization of performance metrics. 

## Definition, Input Format  

The system uses handwriting images as input. In order for the model to process the data effectively, a square image is provided as input. The handwriting image uploaded by the user can optionally undergo an image enhancement stage and is then passed to the Vision Transformer model. The model predicts the dominant personality trait from the image and presents the result through the graphical user interface. 

<p align="center">
  <img src="https://github.com/user-attachments/assets/da18de40-12e3-46e4-a09b-c803daf40ad6" 
       alt="System Workflow"
       width="500">
  <br>
  <em> General workflow of System.</em>
</p>

---
 
## Data Collection Stages and Data Splitting Strategies 

Train-Test Split 

Train-Validation-Test Split 

Stratified K-Fold Cross Validation 

### Model Details, Model Definition, and Adaptation to the Project 

Resizing: The input image is resized to 224 × 224 × 3 (RGB), which is the expected input size of the ViT model pretrained on ImageNet. 

Patch Embedding: The image is divided into 196 patches of size 16 × 16 pixels. Each patch (16×16×3 = 768 dimensions) is transformed into a fixed-size embedding vector through linear projection. 

CLS Token: A special [CLS] token used for classification is prepended to the sequence of 196 patch tokens, resulting in a total of 197 tokens. 

Positional Embedding: A learnable positional embedding vector is added to each token to preserve spatial information. 

Transformer Encoder: Consists of multiple encoder blocks, each containing Layer Normalization, Multi-Head Self-Attention (MSA), residual connections, and MLP layers. 

Attention(Q, K, V) = Softmax(Q·Kᵀ / √dₖ) · V 

Classification Layer: The [CLS] token from the encoder output is passed through the MLP head to generate logits corresponding to five classes, which are then converted into a probability distribution using Softmax. 

σ(z)ᵢ = e^(zᵢ) / Σⱼ e^(zⱼ) 

Two-Stage Fine-Tuning Strategy 

Due to the limited size of the dataset, directly fine-tuning all ImageNet pretrained weights increases the risk of overfitting. To reduce this risk, a two-stage fine-tuning strategy was applied: 

Stage 1 — Linear Probing: The backbone (Transformer encoder) is frozen, and only the classification head is trained. 

Stage 2 — Full Fine-Tuning: The backbone is unfrozen and the entire model is fine-tuned with a low learning rate; the checkpoint with the best Macro-F1 score is saved. 

--- 

## Code Implementation → What is the architectural and implementation approach? 

In the system design, Python was selected as the programming language due to its ease of use for vectorized operations, readability, open-source nature, extensive ecosystem support, and ability to accelerate development through rapid prototyping. In addition, Python is widely used in the modern Deep Learning ecosystem, making it a suitable choice for this project. 

Since Python is a runtime language, excessive use of for-loops can reduce performance and efficiency. This becomes particularly significant in matrix operations commonly used in Deep Learning. This issue is addressed through vectorization provided by libraries, resulting in a more efficient system. 

Visual Studio Code (Windows ecosystem) was used as the development environment (IDE). PyTorch, torchvision, timm, NumPy, scikit-learn, and Pillow were selected as the primary libraries. For GUI development, PyQt6, the modern Python-compatible version of the Qt framework, was used. 

The software architecture was designed in a modular manner. Within the scope of the project, the data layer, model training layer, and user interface layer were separated to create a maintainable and extensible system architecture. This structure enables easy integration of different datasets and training strategies. The codebase generally consists of data preparation, model training, evaluation, preprocessing, inference, and graphical user interface components. 

handwriting-personality-framework/ 
│ 
├── datasets/ 
│   ├── classification/ 
│   └── regression/ 
│ 
├── preprocessing_module/ 
│   └── image_enhancer.cpp 
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
---

## Key Observations  

Within the scope of this study, open-source datasets were utilized because collecting and labeling handwriting data requires considerable time, cost, and expert involvement. The experimental results showed that the Vision Transformer architecture was able to learn specific visual patterns from handwriting images that may be associated with personality traits. In addition, data balancing strategies, data augmentation techniques, and the K-Fold cross-validation approach were found to have a significant impact on model performance. While image enhancement methods improved performance in certain experiments, variations in model predictions were also observed across different handwriting samples from the same individual. This finding highlights both the complexity of handwriting-based personality prediction and the importance of dataset diversity when developing deep learning models for this task.

---

## References 

1. Müsevitoğlu, H., Öztürk, A., & Başünal, F. N. (2023). Detection of Personality Features From Handwriting By Machine Learning Methods. Gazi Journal of Engineering Sciences, 9(2), 200–212.
2. Open-Source Psychometrics Project. IPIP Big-Five Factor Markers. https://openpsychometrics.org/tests/IPIP-BFFM/
3. Chaubey, G. Personality Prediction Using Handwriting Images. Kaggle.
4. Checker, S. HiEnWrite Dataset. GitHub. https://github.com/sakshamchecker/HiEnWrite-Dataset
5. Dosovitskiy, A., et al. (2021). An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale. ICLR.
6. Vaswani, A., et al. (2017). Attention Is All You Need. NeurIPS.
7. Khan, S., et al. (2022). A Survey on Vision Transformers. ACM Computing Surveys, 54(10s), 1–41.
8. Development Seed. Transfer Learning, Fine-Tuning and Hyperparameter Tuning. https://developmentseed.org/tensorflow-eo-training-2/docs/Lesson7c_transfer_learning_hyperparam_opt.html
9. Nielsen, M. Neural Networks and Deep Learning. http://neuralnetworksanddeeplearning.com
10. Zurada, J. M. (1992). Artificial Neural Systems. West Publishing Company.
