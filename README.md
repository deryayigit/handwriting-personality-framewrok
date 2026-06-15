# Handwriting Personality Prediction

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

In this project, Data Collection, Model Development, GUI Integration, and Testing and Validation stages were carried out. Within this scope, multiple datasets were used, these datasets were analyzed, and various methods were applied to identify the most successful model.

The user uploads a handwriting sample image through the developed Graphical User Interface (GUI). The input image is resized and normalized according to the requirements of the Vision Transformer (ViT) architecture (224×224×3). The image is then divided into 16×16 patches and converted into numerical embeddings that can be processed by the model.

A ViT model pretrained on the ImageNet dataset and fine tuned on the handwriting dataset analyzes the input image and predicts the dominant personality trait. The prediction results are presented to the user through the GUI together with a confusion matrix visualizing model performance.

## Training Strategies and Experiments

During the study, class weighting, weighted random sampling, strong data augmentation, and preprocessing based image enhancement techniques were applied. Each technique was tested and compared through separate experiments.

Based on Accuracy and Macro F1 score, the best performing configuration was selected as the default model used for handwriting analysis in the GUI. In addition, the results of all experiments remain accessible through the interface. This allows users to compare different training strategies and evaluate their impact on model performance.


<p align="center">
  <img src="https://github.com/user-attachments/assets/96d3d7f0-52e3-4f9b-8c1f-8c5c29dd1a29" width="650">
</p>

<p align="center">
  <b>General Flowchart</b>
</p>

---
 
## Datasets and Data Splitting Strategies

This project utilizes publicly available handwriting datasets collected from open-source repositories (see Dataset Sources below). The datasets are labeled according to the Big Five (OCEAN) personality model, with handwriting samples associated with personality scores obtained through the IPIP questionnaire.

To investigate model performance under different experimental settings, three data splitting strategies were applied across the experiments:

 **Train-Test Split** a simple 80/20 split used for initial baseline experiments.
 **Train-Validation-Test Split**  a three-way split (typically 60–80% train, 10–20% validation, 10–20% test) used for larger datasets to monitor performance during training.
 **Stratified K-Fold Cross Validation (k=5)**  used on smaller datasets to obtain more reliable performance estimates while preserving class distribution across folds.

---

### Model Architecture

The Vision Transformer (ViT) architecture was introduced by Dosovitskiy et al. in "An Image is Worth 16×16 Words" (ICLR, 2021). Instead of processing an image pixel by pixel, ViT splits it into fixed size patches and treats each patch as a token, similar to words in a sentence. 
In this project, every input image is first resized to 224×224×3 (RGB), which matches the resolution expected by the ImageNet-pretrained ViT. The image is then divided into patches of 16×16 pixels, giving 14 patches along each side and 196 patches in total. Each patch contains 16×16×3 = 768 values, and these are projected through a linear layer into fixed-size embedding vectors. This step is generally referred to as patch embedding.
A special classification token, usually called the CLS token, is added at the beginning of this sequence of 196 patch embeddings, bringing the total to 197 tokens. As the data passes through the encoder, the CLS token interacts with all the patch tokens and gradually builds up a representation of the whole image. Since the Transformer itself has no builtin notion of where each patch sits in the image, a learnable positional embedding is added to every token so that spatial layout information is not lost.
The core of the model is a stack of Transformer encoder blocks, each made up of layer normalization, multi-head self-attention, residual connections, and a small MLP. The attention mechanism computes, for every patch, how strongly it should attend to every other patch:

```math
\text{Attention}(Q,K,V)=\text{Softmax}\left(\frac{QK^{T}}{\sqrt{d_k}}\right)V
```

This is where ViT really differs from a typical CNN. A convolutional network builds up its understanding gradually, starting from small local filters (3×3 or 5×5) that detect edges and textures before combining them into larger shapes. ViT instead looks at all 196 patches at once from the very first layer, so it can directly relate, say, the slant of the first line of text to the spacing of the last line, even though they are far apart in the image.

After passing through all the encoder blocks, the final representation of the CLS token is fed into an MLP classification head, which produces a logit for each of the five OCEAN personality classes. These logits are converted into probabilities using softmax:

```math
\sigma(z_i)=\frac{e^{z_i}}{\sum_{j=1}^{C} e^{z_j}}
```

Because the dataset used in this project is relatively small, the model is trained in two stages rather than fine tuning everything at once. In the first stage, the pretrained backbone is frozen and only the classification head is trained, so the model learns to map its existing visual features to the five personality classes. In the second stage, the backbone is unfrozen and the whole model is fine tuned with a low learning rate, while the checkpoint with the best Macro-F1 score on the validation set is kept. This two step approach helps the model adapt to handwriting images without losing the general visual knowledge it picked up from ImageNet.

<p align="center">
    <img src="https://github.com/user-attachments/assets/1e5c4733-d90f-4caf-a50b-634523aed69c" width="850">
</p>

<p align="center">
  <b>Vision Transformer (ViT) Architecture Overview</b>
</p>

---

## Code Implementation

### ViT Two-Stage Fine-Tuning Strategy 

Due to the limited size of the dataset, directly fine-tuning all ImageNet pretrained weights increases the risk of overfitting. To reduce this risk, a two stage fine tuning strategy was applied: 

Stage 1 — Linear Probing: The backbone (Transformer encoder) is frozen, and only the classification head is trained. 

Stage 2 — Full Fine-Tuning: The backbone is unfrozen and the entire model is fine-tuned with a low learning rate, the checkpoint with the best Macro-F1 score is saved. 

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
### Development Environment and Libraries

Python was selected as the main programming language because of its simplicity, readability, and strong support for Deep Learning applications. Its extensive ecosystem and rapid development capabilities made it a suitable choice for this project.

Since Deep Learning models rely heavily on matrix operations, vectorized computations provided by libraries were preferred instead of manually written loops. This approach improves both efficiency and training performance.

Visual Studio Code was used as the development environment. PyTorch, torchvision, timm, NumPy, scikit learn, and Pillow were used during model development. PyQt6 was used to develop the graphical user interface.

### Software Architecture

The software was designed using a modular architecture. The data layer, model training layer, and user interface layer were separated to create a maintainable and extensible system.

This structure allows different datasets, training strategies, and model configurations to be integrated easily. The codebase mainly consists of data preparation, model training, evaluation, preprocessing, inference, and GUI components.

Each module is responsible for a specific task. Model loading, inference, and user interactions are managed through the GUI. Model training is performed separately, and the GUI is used to present the generated results.

Project Structure

Training outputs are stored inside the runs/ directory. A separate folder is created for each experiment.

config.yaml stores training configurations.
metrics.json stores epoch based performance metrics.
model.pt stores the best saved model.
labels.json stores class labels and label mappings.
splits.json stores dataset split information.

main.py is the main entry point of the project. Training, evaluation, and GUI execution are managed through this file.

dataset_registry.json and experiment_registry.json define the datasets and experiment configurations used throughout the project.

src/classification/ contains the training, model, and evaluation modules used for classification based personality prediction.

src/regression/ contains experiments designed to predict personality scores as continuous values.

src/common/ contains shared utilities, dataset loading functions, experiment management tools, and preprocessing integrations.

preprocessing_module/image_enhancer.cpp is a C++ based preprocessing module developed for handwriting image enhancement. Operations such as background cleaning and contrast improvement are implemented in this module.

The runs/ directory contains trained models, performance metrics, confusion matrices, and temporary images generated during GUI execution.

The datasets/ directory contains the datasets used in classification and regression experiments.

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

1. Detection of Personality Features From Handwriting By Machine Learning Methods  
   Müsevitoğlu, H., Öztürk, A., & Başünal, F. N. (2023). *Detection of Personality Features From Handwriting By Machine Learning Methods*. Gazi Journal of Engineering Sciences, 9(2), 200–212.

2. IPIP Big-Five Factor Markers  
   Open-Source Psychometrics Project. *IPIP Big-Five Factor Markers*.

   https://openpsychometrics.org/tests/IPIP-BFFM/

3. An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale  
   Dosovitskiy, A., et al. (2021). *An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale*. ICLR.

4. Attention Is All You Need  
   Vaswani, A., et al. (2017). *Attention Is All You Need*. NeurIPS.

5. A Survey on Vision Transformers  
   Khan, S., et al. (2022). *A Survey on Vision Transformers*. ACM Computing Surveys, 54(10s), 1–41.

6. Transfer Learning, Fine-Tuning and Hyperparameter Tuning  
   Development Seed. *Transfer Learning, Fine-Tuning and Hyperparameter Tuning*.

   https://developmentseed.org/tensorflow-eo-training-2/docs/Lesson7c_transfer_learning_hyperparam_opt.html

7. Neural Networks and Deep Learning  
   Nielsen, M. *Neural Networks and Deep Learning*.

   http://neuralnetworksanddeeplearning.com

8. Artificial Neural Systems  
   Zurada, J. M. (1992). *Artificial Neural Systems*. West Publishing Company.

---

## Acknowledgements

This project was carried out under the guidance and support of my advisor, Assoc. Prof. Dr. Selen Ayas. I would like to thank the members of my final project jury Prof. Dr. Murat Ekinci, Assoc. Prof. Dr. Selen Ayas and Res. Asst. Mustafa Yazıcı, for the time they devoted to the project evaluation process.

I would especially like to thank Prof. Dr. Murat Ekinci for helping me develop a strong engineering mindset through his lectures and assignments and for emphasizing the importance of analytical thinking. His approach significantly contributed to my ability to examine and understand the technologies, methods and research processes involved in this study in greater depth.
