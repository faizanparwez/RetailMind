# RetailMind AI

## Product Vision
RetailMind AI is an incremental machine learning project focused on building intelligent solutions for retail businesses.

## Day 1 — Customer Churn Prediction

### Objective
Build an end-to-end machine learning pipeline to predict whether a telecom customer is likely to leave the service.

### Dataset
Telco Customer Churn Dataset — Kaggle  
Source: https://www.kaggle.com/datasets/blastchar/telco-customer-churn

### Technologies
- Python
- Pandas
- Scikit-learn
- Joblib

### ML Pipeline
1. Data ingestion and inspection.
2. Data cleaning and handling missing values.
3. Train-test split with stratification.
4. Categorical encoding using OneHotEncoder.
5. Numerical scaling using StandardScaler.
6. Classification using Logistic Regression.
7. Model evaluation.
8. Model persistence using Joblib.
9. Customer churn prediction using a reusable `predict()` function.

### Baseline Evaluation

| Metric | Score |
|---|---:|
| Accuracy | 80.55% |
| Precision | 65.72% |
| Recall | 55.88% |
| ROC-AUC | 84.21% |

### Model Artifact
`models/churn_pipeline.joblib`

### Prediction Service
A reusable `predict()` helper accepts customer information and returns:
- Predicted churn status (`Yes` or `No`).
- Estimated churn probability.

### Future Development
This baseline serves as the starting point for future improvements to RetailMind AI.


## Day 2 — Model Comparison

### Objective
Compare Decision Tree, Random Forest and Support Vector Machine (SVM) models for customer churn prediction.

### Model Comparison

| Model | Accuracy | Precision | Recall | ROC-AUC |
|---|---:|---:|---:|---:|
| Decision Tree | 72.11% | 47.55% | 49.20% | 64.77% |
| Random Forest | 78.64% | 62.54% | 48.66% | 81.85% |
| SVM | 79.13% | 64.18% | 48.40% | 79.02% |

### Champion Model: Decision Tree

Decision Tree was selected as the Day 2 champion because it achieved the highest recall (49.20%) among the three models.

Although Random Forest and SVM achieved better accuracy, precision and ROC-AUC, the selection prioritised recall to identify as many potential churners as possible.

### Model Artifact
`models/churn_champion.joblib`

### Observation
The Day 1 Logistic Regression baseline outperformed all three Day 2 models on the evaluated metrics. This demonstrates that more complex algorithms do not necessarily guarantee better performance.

## Day 3 — Hyperparameter Tuning and Bias-Variance Analysis

### Objective
Tune Decision Tree, Random Forest and Support Vector Machine (SVM) models using GridSearchCV and evaluate them using 5-fold stratified cross-validation.

### Model Comparison

| Model | Training Recall | Cross-Validation Recall |
|---|---:|---:|
| Decision Tree | 55.17% | 51.37% |
| Random Forest | 60.84% | 51.77% |
| SVM | 53.34% | 53.38% |

### Champion Model: SVM

SVM was selected as the Day 3 champion because it achieved the highest cross-validation recall of 53.38%.

### Final Test Set Evaluation

| Metric | Score |
|---|---:|
| Accuracy | 78.78% |
| Precision | 61.76% |
| Recall | 52.67% |
| ROC-AUC | 82.46% |

### Bias-Variance Analysis

The learning curve shows that training and validation recall gradually converge as the training dataset increases.

The small difference between the final training and validation scores suggests limited overfitting. However, the relatively modest recall indicates potential underfitting or limitations in the current feature set.

Further feature engineering and model tuning may improve performance.

### Hyperparameter Tuning

GridSearchCV was used with 5-fold StratifiedKFold cross-validation. The models were tuned using recall as the scoring metric, with `"Yes"` designated as the positive class.

The best hyperparameters for each model were identified and printed during execution.

### Model Artifact

`models/churn_v3.joblib`

### Learning Curve

`models/churn_v3_learning_curve.png`


## Day 4 — Feature Engineering and Ensemble Learning

### Objective
Improve customer churn prediction through domain-specific feature engineering, PCA dimensionality reduction, and ensemble learning.

### Feature Engineering
The following features were added:
- Tenure groups: categorised customers based on subscription duration.
- Average monthly charge: calculated using total charges and tenure.
- Service count: calculated the number of subscribed services.
- Contract flags: identified month-to-month, one-year and two-year contracts.
- Customer flags: identified online security, technical support and dependents.

Total features after engineering: 28.

### Dimensionality Reduction
Applied Principal Component Analysis (PCA) to numerical features, retaining 95.45% of their explained variance.

### Ensemble Models
- Random Forest: Bagging ensemble.
- XGBoost: Boosting ensemble.

Both models were compared against the Day 3 SVM champion using the same train-test split.

### Day 4 Model Comparison

| Model | Accuracy | Precision | Recall | ROC-AUC |
|---|---:|---:|---:|---:|
| Day 3 SVM | 78.78% | 61.76% | 52.67% | 82.46% |
| Random Forest | 76.79% | 54.87% | 70.86% | 83.60% |
| XGBoost | 75.16% | 52.20% | 76.20% | 83.75% |

### Final Champion: XGBoost

XGBoost was selected as the final champion because it achieved the highest recall (76.20%) and ROC-AUC (83.75%) among the evaluated models.

Compared with the Day 3 SVM, recall improved by 23.53 percentage points.

The lower precision and accuracy represent a trade-off: the model identifies more potential churners but also produces more false-positive predictions.

### Cumulative Model Baseline

| Day | Model | Accuracy | Precision | Recall | ROC-AUC |
|---|---|---:|---:|---:|---:|
| Day 1 | Logistic Regression | 80.55% | 65.72% | 55.88% | 84.21% |
| Day 2 | Decision Tree | 72.11% | 47.55% | 49.20% | 64.77% |
| Day 3 | SVM | 78.78% | 61.76% | 52.67% | 82.46% |
| Day 4 | XGBoost | 75.16% | 52.20% | 76.20% | 83.75% |

### Production Model Artifact

`models/churn_production.joblib`

The saved pipeline includes the fitted preprocessing steps and XGBoost classifier, allowing predictions on raw customer features without manually repeating preprocessing.

### Phase 1 Milestone
Completed the initial churn prediction pipeline, model comparison, hyperparameter tuning, feature engineering, dimensionality reduction and ensemble learning.



## Day 5 — Neural Networks and Artificial Neural Networks

### Objective
Build an Artificial Neural Network (ANN) using TensorFlow/Keras to predict customer churn and compare its performance against the Day 4 XGBoost champion.

### ANN Architecture
- Input layer: Preprocessed customer features.
- Hidden layer 1: 64 neurons, ReLU activation.
- Dropout: 30%.
- Hidden layer 2: 32 neurons, ReLU activation.
- Dropout: 20%.
- Output layer: 1 neuron, sigmoid activation.

### Training Configuration
- Optimizer: Adam
- Loss function: Binary Crossentropy
- Batch size: 32
- Maximum epochs: 100
- Early stopping: Enabled, patience of 10 epochs.
- Actual training duration: 40 epochs.
- Class weights: Balanced to account for churn class imbalance.

### Model Comparison

| Metric | Day 4 XGBoost | Day 5 ANN |
|---|---:|---:|
| Accuracy | 75.16% | 75.73% |
| Precision | 52.20% | 53.01% |
| Recall | 76.20% | 75.40% |
| ROC-AUC | 83.75% | 83.57% |

### Observations

The ANN achieved slightly higher accuracy and precision, while XGBoost achieved slightly higher recall and ROC-AUC.

The performance difference was small, indicating that the ANN learned useful patterns but did not outperform the classical ensemble model.

### Production Decision

XGBoost remains the RetailMind production champion because it achieved higher recall and ROC-AUC in this comparison.

The ANN is retained as an alternative model for future experimentation.

### When to Use Which

- **XGBoost:** Preferred for structured tabular data, strong predictive performance and relatively efficient training.
- **ANN:** Useful when relationships are complex and sufficient data is available, particularly when working with unstructured data or combining multiple data types.

### Model Artifacts
- `models/churn_ann.keras`
- `models/churn_ann_preprocessor.joblib`
- `models/day5_model_comparison.csv`
- `models/churn_ann_learning_curve.png`


## Day 6 — CNN Image Classification

### Objective
Developed a Convolutional Neural Network (CNN) to automatically classify retail product images for the RetailMind AI catalog.

### Datasets
- Fashion-MNIST for CNN prototyping.
- Fashion Product Images (Small) for the product classification module.
- Target variable: `articleType`.

### CNN Architecture
- Input: 128 × 128 RGB images
- Conv2D: 32 filters + MaxPooling
- Conv2D: 64 filters + MaxPooling
- Conv2D: 128 filters + MaxPooling
- Flatten
- Dropout (0.4)
- Dense: 128 neurons
- Dropout (0.3)
- Softmax output layer

### Training Configuration
- Optimizer: Adam
- Loss: Sparse Categorical Crossentropy
- Batch size: 32
- Maximum epochs: 20
- Early stopping based on validation loss
- Data augmentation: Random Flip, Rotation and Zoom

### Evaluation Results

| Metric | Result |
|---|---:|
| Test Loss | 0.6173 |
| Test Accuracy | 82.53% |

### Generated Artifacts
- `models/retailmind_cnn.keras`
- `models/retailmind_cnn_best.keras`
- `models/retailmind_cnn_classes.txt`
- `models/day6_confusion_matrix.csv`
- `models/day6_cnn_accuracy.png`

### Outcome
Successfully developed and saved the first computer vision module of RetailMind AI. The CNN achieved 82.53% test accuracy on the product image classification task.


## Day 7 — Transfer Learning & Object Detection

### Objective
Improve fashion product classification using transfer learning with a pretrained convolutional neural network.

### Features

1. **Transfer Learning with VGG16**
   - Uses VGG16 pretrained on ImageNet.
   - Removes the original classification head.
   - Adds a custom classification head for fashion product categories.

2. **Feature Extraction**
   - Freezes the pretrained VGG16 layers.
   - Trains the newly added classification layers.

3. **Fine-Tuning**
   - Unfreezes the final VGG16 convolutional block.
   - Uses a lower learning rate to fine-tune pretrained features.

4. **Model Evaluation**
   - Evaluates the model on the held-out test dataset.
   - Compares transfer-learning performance with the Day 6 CNN.

### Technologies
- TensorFlow / Keras
- VGG16
- ImageNet pretrained weights

### Main File
`src/day7_transfer_learning.py`

### Model Architecture

- Pretrained VGG16 (without original classification head)
- Global Average Pooling
- Dense layer (256 units, ReLU)
- Dropout (0.4)
- Softmax classification output

### Training Strategy

**Phase 1 — Feature Extraction**
- Freeze pretrained VGG16 layers.
- Train the custom classification head.

**Phase 2 — Fine-Tuning**
- Unfreeze the final convolutional block.
- Continue training with a smaller learning rate.

### YOLO Object Detection

YOLO-based object detection was postponed following the training instructions because of the additional bounding-box annotation requirements.

### Day 7 Deliverable
A transfer-learning-based fashion product classification pipeline using pretrained VGG16, with feature extraction and fine-tuning stages.

### Status
Implementation created. Full training and final performance evaluation remain pending.


## Day 8 — Segmentation & End-to-End Computer Vision

### Objective
Build an explainable computer-vision module for fashion product catalog images.

### Features

1. **Product Classification**
   - Reuses the trained Day 6 CNN model.
   - Predicts the product category and confidence score.

2. **Image Segmentation**
   - Uses OpenCV GrabCut to separate the product from its background.
   - Generates a binary foreground mask.

3. **Background Removal**
   - Removes the original background.
   - Composites the extracted product onto a uniform white background.

4. **Product Detection**
   - Extracts the product's bounding box from the segmentation mask.
   - YOLO-based detection remains postponed.

5. **Grad-CAM Explainability**
   - Generates activation heatmaps from the CNN's final convolutional layer.
   - Highlights image regions influencing classification predictions.

### Technologies
- TensorFlow / Keras
- OpenCV
- NumPy

### Main File
`src/day8_cv_module.py`

### Model Dependency
`models/retailmind_cnn.keras`

### Output Structure

```text
outputs/
└── day8/
    ├── segmented/
    │   ├── *_clean.jpg
    │   └── *_mask.png
    └── gradcam/
        └── *_gradcam.jpg
```

### Methodology
The segmentation pipeline uses OpenCV GrabCut rather than a supervised U-Net because the Fashion Product Images dataset does not include pixel-level segmentation masks.

Grad-CAM provides visual explanations of classification predictions.

### Limitations
- GrabCut may not perfectly separate products with complex backgrounds.
- Bounding boxes are derived from segmentation masks rather than an independent object detector.
- Grad-CAM explains classification decisions and does not validate segmentation accuracy.

### Day 8 Deliverable
An integrated computer-vision module combining product classification, segmentation, background removal, bounding-box extraction, and explainability.



## Day 9 — RNN / LSTM Demand Forecasting

### Objective
Build an LSTM-based demand forecasting module to predict
daily product demand and support retail inventory planning.

### Dataset
Store Item Demand Forecasting Challenge (Kaggle).

Selected the 20 highest-selling store-item combinations
for forecasting.

### Data Preparation
- Aggregated sales into daily time series using pandas.
- Filled missing dates with zero sales.
- Normalized each store-item series using its training-period maximum.
- Used a 30-day look-back window to predict next-day sales.
- Reserved the most recent 28 days for testing.

### Model Architecture
- Input: 30 daily observations
- LSTM: 64 units, return_sequences=True
- Dropout: 20%
- LSTM: 32 units
- Dense: 16 units, ReLU
- Output: 1 unit

### Training Configuration
- Optimizer: Adam
- Loss: Mean Absolute Error (MAE)
- Batch size: 64
- Maximum epochs: 30
- Early stopping: validation loss, patience 5

### Evaluation

| Metric | Result |
|---|---|
| MAE | Add actual result |
| RMSE | Add actual result |

### Outputs
- `models/demand_forecaster.keras`
- `outputs/day9/test_predictions.csv`
- `outputs/day9/evaluation_metrics.csv`
- `outputs/day9/next_day_forecast.csv`
- `outputs/day9/series_metadata.csv`
- `outputs/day9/actual_vs_predicted.svg`

### Key Learning
- Time-series forecasting
- Sliding-window supervised learning
- LSTM sequence modelling
- Time-based train/test splitting
- MAE and RMSE evaluation
- Demand forecasting for inventory planning

### Next Step
Extend the forecasting module to multi-horizon prediction
using Seq2Seq architecture and attention.



## Day 10 — Seq2Seq & Attention Demand Forecasting

### Objective
Extend the Day 9 demand forecasting module from single-step
prediction to multi-horizon forecasting using Seq2Seq and attention.

### Dataset
Store Item Demand Forecasting Challenge (Kaggle).

Selected the 20 highest-selling store-item combinations.

### Data Preparation
- Aggregated sales into daily time series.
- Filled missing dates with zero sales.
- Normalized each series using its training-period maximum.
- Historical input window: 30 days.
- Forecast horizon: 7 days.
- Reserved the most recent 28 days for testing.

### Model Architecture

**Encoder**
- LSTM with 64 units.
- Returns historical sequence outputs and hidden states.

**Attention**
- Keras Attention layer.
- Decoder outputs attend to encoder outputs.
- Context vectors combined with decoder representations.

**Decoder**
- RepeatVector for seven forecast steps.
- LSTM with 64 units.
- Dense layer with 32 ReLU units.
- Dense output layer for daily demand.

### Baseline Comparison
Compared:
- Single-step LSTM with recursive seven-day predictions.
- Seq2Seq LSTM with attention and seven-day output.

### Evaluation

| Model | MAE | RMSE |
|---|---|---|
| Single-step LSTM | Add result | Add result |
| Seq2Seq Attention | Add result | Add result |

### Outputs
- `models/day10_single_step_baseline.keras`
- `models/demand_seq2seq_attention.keras`
- `outputs/day10/multi_horizon_predictions.csv`
- `outputs/day10/model_comparison.csv`
- `outputs/day10/attention_weights.svg`

### Key Learning
- Sequence-to-sequence learning.
- Encoder-decoder architecture.
- Multi-horizon forecasting.
- Attention mechanisms.
- Attention weight visualization.
- Recursive forecasting vs direct multi-step forecasting.

### Next Step
Explore GAN/VAE-based synthetic product image generation
to balance rare catalog categories.


## Day 11 — GANs & VAEs: Synthetic Fashion Image Augmentation

### Objective
Generate synthetic product images using a Generative Adversarial Network (GAN) to augment under-represented fashion categories and improve the balance of the RetailMind computer vision dataset.

### Dataset
**Fashion Product Images (Kaggle)**

Dataset location:
- `data/fashion-product-images-small/styles.csv`
- `data/fashion-product-images-small/images/`

Category information is extracted from `articleType` (or `masterCategory` if unavailable) in `styles.csv`.

### Approach
Implemented a DCGAN-style architecture consisting of two neural networks:

**1. Generator**
- Takes random noise as input.
- Latent dimension: 100.
- Uses Dense and Conv2DTranspose layers.
- Batch Normalization and LeakyReLU activation.
- Generates 64 × 64 RGB product images.
- Uses Tanh activation in the output layer.

**2. Discriminator**
- Receives real and generated images.
- Uses Conv2D layers with LeakyReLU activation.
- Dropout: 0.3.
- Final Dense layer produces a real/fake score.

### Training Configuration

| Parameter | Value |
|---|---|
| Framework | TensorFlow / Keras |
| Image size | 64 × 64 |
| Channels | 3 (RGB) |
| Latent dimension | 100 |
| Batch size | 32 |
| Epochs | 100 |
| Optimizer | Adam |
| Learning rate | 0.0002 |
| Beta 1 | 0.5 |
| Loss | Binary Crossentropy |
| Default synthetic images | 500 |

### Data Preprocessing
- Loaded category metadata from `styles.csv`.
- Matched product IDs with their corresponding image files.
- Identified category distribution to locate under-represented categories.
- Resized images to 64 × 64 pixels.
- Normalized pixel values from [0, 255] to [-1, 1].
- Created a shuffled TensorFlow dataset.

### GAN Training
Implemented a custom Keras training loop using `GradientTape`.

Each training step:
1. Generates fake images from random noise.
2. Trains the discriminator using real and generated images.
3. Trains the generator to produce images that fool the discriminator.
4. Tracks generator and discriminator losses.

### Synthetic Image Generation
- Automatically selects the smallest category by default.
- Supports manually selecting a category through command-line arguments.
- Generates synthetic product images using the trained generator.
- Saves generated images in PNG format.

### Model and Output Files

**Saved model:**
- `models/day11_generator.keras`

**Generated images:**
- `data/fashion-product-images-small/synthetic_images/<category>/`

**Run metadata:**
- `data/fashion-product-images-small/synthetic_images/day11_run_metadata.csv`

### Key Learning
- Generative Adversarial Networks (GANs).
- Generator and discriminator architecture.
- Transposed convolution for image generation.
- Latent space representation.
- Adversarial training using GradientTape.
- Synthetic image generation and dataset augmentation.

### Important Observation
Generated images must be visually inspected for realism and mode collapse before being added to the original training dataset.

The generated images have not yet been integrated into the Day 6/7 classifier, and the impact on per-class accuracy remains to be evaluated.

### Next Step
Begin text intelligence by cleaning product reviews and developing a classical sentiment analysis baseline.
