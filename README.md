# Chicken Disease Classification System

An end-to-end deep learning application for detecting chicken diseases using convolutional neural networks (CNN). This system uses transfer learning with VGG16 to classify chicken fecal images into two categories: **Healthy** and **Coccidiosis**.

## Table of Contents

1. [Features](#features)
2. [Project Overview](#project-overview)
   - [Objective](#objective)
   - [Technology Stack](#technology-stack)
   - [Key Metrics](#key-metrics)
3. [Project Structure](#project-structure)
4. [Installation](#installation)
   - [Prerequisites](#prerequisites)
   - [Step-by-step Setup](#step-1-clone-the-repository)
   - [Verification](#step-5-verify-installation)
5. [Quick Start](#quick-start)
   - [Running Web Application](#running-the-web-application)
   - [Using Docker](#using-docker)
6. [Usage](#usage)
   - [Web Interface Features](#web-interface-features)
   - [API Usage](#api-usage)
   - [Training](#training)
   - [TensorBoard Monitoring](#monitor-training-with-tensorboard)
7. [Model Architecture](#model-architecture)
   - [Base Architecture](#base-architecture)
   - [Transfer Learning Approach](#transfer-learning-approach)
   - [Model Layers](#model-layers)
   - [Training Configuration](#training-configuration)
8. [Configuration](#configuration)
   - [Model Hyperparameters](#model-hyperparameters-paramsyaml)
   - [Project Configuration](#project-configuration-configconfigyaml)
   - [Modifying Parameters](#modifying-hyperparameters)
9. [Dataset](#dataset)
   - [Overview](#overview)
   - [Data Structure](#data-directory-structure)
   - [Data Augmentation](#data-augmentation)
   - [Preprocessing](#preprocessing)
10. [API Documentation](#api-documentation)
    - [Health Check Endpoint](#health-check-endpoint-optional)
    - [Training Endpoint](#training-endpoint)
    - [Prediction Endpoint](#prediction-endpoint)
    - [Error Handling](#error-handling)
11. [Troubleshooting](#troubleshooting)
    - [Common Issues](#issue-importerror-for-cnnclassifier)
    - [Solutions](#solution-install-package-in-development-mode)
12. [Performance Metrics](#performance-metrics)
    - [Model Evaluation](#model-evaluation)
    - [Key Metrics](#key-metrics-1)
    - [Performance Factors](#performance-factors)
13. [Contributing](#contributing)
    - [How to Contribute](#how-to-contribute)
    - [Areas for Contribution](#areas-for-contribution)
    - [Code Style](#code-style)
14. [License](#license)
15. [Additional Resources](#additional-resources)
    - [Documentation](#documentation)
    - [Related Projects](#related-projects)
    - [Tutorials](#tutorials)
16. [Contact & Support](#contact--support)
17. [Version History](#version-history)
18. [Acknowledgments](#acknowledgments)

---

## Features

✨ **Core Features:**
- 🧠 **Transfer Learning**: Uses pre-trained VGG16 model for efficient learning
- 📸 **Image Classification**: Binary classification (Healthy vs Coccidiosis)
- 🌐 **Web Interface**: Modern, responsive UI with drag-and-drop support
- 🔌 **REST API**: Easy integration with other applications
- 📊 **Confidence Scores**: Returns prediction probability for each class
- 🎯 **Data Augmentation**: Improves model generalization
- 📈 **TensorBoard Monitoring**: Track training progress in real-time
- 💾 **Model Checkpointing**: Saves best model during training
- 🐳 **Docker Support**: Containerized deployment ready

---

## Project Overview

### Objective
Develop an automated system to detect chicken diseases (specifically Coccidiosis) from chicken fecal images using deep learning, enabling early diagnosis and treatment.

### Technology Stack
- **Deep Learning Framework**: TensorFlow/Keras
- **Backend**: Flask with Flask-CORS
- **Frontend**: HTML5, CSS3, JavaScript
- **Data Pipeline**: DVC (Data Version Control)
- **Model Base**: VGG16 (pre-trained on ImageNet)
- **Containerization**: Docker
- **Python Version**: 3.x

### Key Metrics
- **Input Size**: 224×224×3 pixels
- **Classes**: 2 (Healthy, Coccidiosis)
- **Base Model**: VGG16 (ImageNet weights)
- **Batch Size**: 16 images
- **Epochs**: Configurable (default: 1)

---

## Project Structure

```
End_to_End_ML_project_Chiken_disease_classification/
│
├── app.py                          # Flask application entry point
├── main.py                         # Training pipeline orchestrator
├── setup.py                        # Package setup and dependencies
├── Dockerfile                      # Docker configuration
├── dvc.yaml                        # DVC pipeline configuration
├── params.yaml                     # Model hyperparameters
├── config/
│   └── config.yaml                # Project configuration
│
├── src/cnnClassifier/              # Main package
│   ├── __init__.py
│   ├── components/                 # Pipeline components
│   │   ├── data_ingestion.py       # Data download and extraction
│   │   ├── prepare_base_model.py   # Load and freeze base model
│   │   ├── prepare_callbacks.py    # TensorBoard and checkpoint callbacks
│   │   ├── training.py             # Model training logic
│   │   └── evaluation.py           # Model evaluation metrics
│   │
│   ├── config/
│   │   └── configuration.py        # Configuration loader
│   │
│   ├── constants/
│   │   └── __init__.py             # Project constants
│   │
│   ├── entity/
│   │   └── config_entity.py        # Data class configurations
│   │
│   ├── pipeline/                   # ML pipelines
│   │   ├── stage_01_data_ingestion.py
│   │   ├── stage_02_prepare_base_model.py
│   │   ├── stage_03_training.py
│   │   ├── stage_04_evaluation.py
│   │   └── predict.py              # Prediction pipeline
│   │
│   └── utils/
│       └── common.py               # Utility functions
│
├── templates/
│   └── index.html                  # Web UI
│
├── research/                       # Jupyter notebooks
│   ├── 01_data_ingestion.ipynb
│   ├── 02_prepare_base_model.ipynb
│   ├── 03_prepare_callbacks.ipynb
│   ├── 04_training.ipynb
│   └── 05_model_evaluation.ipynb
│
├── artifacts/                      # Generated artifacts
│   ├── data_ingestion/             # Downloaded dataset
│   ├── prepare_base_model/         # Base model files
│   ├── prepare_callbacks/          # Callbacks and logs
│   │   ├── checkpoint_dir/
│   │   └── tensorboard_log_dir/
│   └── training/                   # Trained model
│
├── requirements.txt                # Python dependencies
└── README_DETAILED.md              # This file
```

---

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)
- 4GB RAM minimum
- GPU support recommended (NVIDIA CUDA)

### Step 1: Clone the Repository
```bash
git clone <https://github.com/ajaychaudhary8104/End_to_End_ML_project_Chiken_disease_classification.git>
cd End_to_End_ML_project_Chiken_disease_classification
```

### Step 2: Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Install Package
```bash
pip install -e .
```

### Step 5: Verify Installation
```bash
python -c "import tensorflow as tf; print(tf.__version__)"
```

---

## Quick Start

### Running the Web Application

#### 1. Start the Flask Server
```bash
python app.py
```

Expected output:
```
 * Running on http://0.0.0.0:8080
 * Debug mode: on
```

#### 2. Open Web Interface
Navigate to: `http://localhost:8080`

#### 3. Use the Application
1. **Upload Image**: Click "Choose Image" or drag & drop a chicken image
2. **Make Prediction**: Click "Predict" button
3. **View Results**: See disease classification with confidence scores

## Workflows

1. Update config.yaml
2. Update secrets.yaml [Optional]
3. Update params.yaml
4. Update the entity
5. Update the configuration manager in src config
6. Update the components
7. Update the pipeline 
8. Update the main.py
9. Update the dvc.yaml

# AWS-CICD-Deployment-with-Github-Actions

## 1. Login to AWS console.

## 2. Create IAM user for deployment

	#with specific access

	1. EC2 access : It is virtual machine

	2. ECR: Elastic Container registry to save your docker image in aws


	#Description: About the deployment

	1. Build docker image of the source code

	2. Push your docker image to ECR

	3. Launch Your EC2 

	4. Pull Your image from ECR in EC2

	5. Lauch your docker image in EC2

	#Policy:

	1. AmazonEC2ContainerRegistryFullAccess

	2. AmazonEC2FullAccess

	
## 3. Create ECR repo to store/save docker image
    - Save the URI: 566373416292.dkr.ecr.us-east-1.amazonaws.com/chicken

	
## 4. Create EC2 machine (Ubuntu) 

## 5. Open EC2 and Install docker in EC2 Machine:
	
	
	#optinal

	sudo apt-get update -y

	sudo apt-get upgrade
	
	#required

	curl -fsSL https://get.docker.com -o get-docker.sh

	sudo sh get-docker.sh

	sudo usermod -aG docker ubuntu

	newgrp docker
	
# 6. Configure EC2 as self-hosted runner:
    setting>actions>runner>new self hosted runner> choose os> then run command one by one


# 7. Setup github secrets:

    AWS_ACCESS_KEY_ID=

    AWS_SECRET_ACCESS_KEY=

    AWS_REGION = us-east-1

    AWS_ECR_LOGIN_URI = demo>>  566373416292.dkr.ecr.ap-south-1.amazonaws.com

    ECR_REPOSITORY_NAME = simple-app




# AZURE-CICD-Deployment-with-Github-Actions

## Save pass:

s3cEZKH5yytiVnJ3h+eI3qhhzf9q1vNwEi6+q+WGdd+ACRCZ7JD6


## Run from terminal:

docker build -t chickenapp.azurecr.io/chicken:latest .

docker login chickenapp.azurecr.io

docker push chickenapp.azurecr.io/chicken:latest


## Deployment Steps:

1. Build the Docker image of the Source Code
2. Push the Docker image to Container Registry
3. Launch the Web App Server in Azure 
4. Pull the Docker image from the container registry to Web App server and run 

---

## Usage

### Web Interface Features

#### Image Upload
- **Drag & Drop**: Drag image onto preview area
- **File Browser**: Click "Choose Image" button
- **Supported Formats**: PNG, JPG, JPEG, GIF, BMP

#### Prediction Results
- **Classification**: Primary prediction class
- **Confidence Score**: Probability for each class (%)
- **Visual Progress Bar**: Shows relative confidence
- **Processed Image**: Display model output if available

#### Controls
- **Upload**: Select new image
- **Predict**: Run disease detection
- **Clear**: Reset interface for new prediction

### API Usage

#### Endpoint: `/predict`

**Method**: `POST`

**Headers**:
```
Content-Type: application/json
```

**Request Body**:
```json
{
  "image": "base64_encoded_image_string"
}
```

**Response**:
```json
[
  {
    "image": "Healthy",
    "prediction": {
      "Coccidiosis": 0.15,
      "Healthy": 0.85
    }
  }
]
```

**Example using cURL**:
```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d "{\"image\":\"your_base64_image_here\"}"
```

**Example using Python**:
```python
import requests
import base64

# Read image file
with open('chicken_image.jpg', 'rb') as f:
    image_data = base64.b64encode(f.read()).decode()

# Send prediction request
response = requests.post(
    'http://localhost:8080/predict',
    json={'image': image_data}
)

# Parse response
result = response.json()
prediction = result[0]['image']
confidence = result[0]['prediction']
print(f"Prediction: {prediction}")
print(f"Confidence: {confidence}")
```

### Training

#### Train New Model
```bash
python main.py
```

This executes the complete pipeline:
1. ✅ Data Ingestion (download & extract)
2. ✅ Base Model Preparation (load VGG16)
3. ✅ Callbacks Setup (TensorBoard, checkpoint)
4. ✅ Model Training
5. ✅ Model Evaluation

#### Monitor Training with TensorBoard
```bash
tensorboard --logdir=artifacts/prepare_callbacks/tensorboard_log_dir
```

Then open: `http://localhost:6006`

---

## Model Architecture

### Base Architecture
- **Model**: VGG16 (pre-trained on ImageNet)
- **Input Shape**: (224, 224, 3)
- **Output Classes**: 2

### Transfer Learning Approach
```
Pre-trained VGG16 Layers (Frozen)
        ↓
Custom Dense Layers
        ↓
Dropout Layers (Regularization)
        ↓
Output Layer (Softmax, 2 classes)
```

### Model Layers
```
- Input: 224×224×3
- Conv Blocks: Frozen from VGG16
- Global Average Pooling
- Dense(512, activation='relu')
- Dropout(0.5)
- Dense(256, activation='relu')
- Dropout(0.3)
- Output: Dense(2, activation='softmax')
```

### Training Configuration
- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Learning Rate**: 0.01
- **Batch Size**: 16
- **Image Augmentation**: Enabled (flip, rotate, zoom)

---

## Configuration

### Model Hyperparameters (`params.yaml`)

```yaml
AUGMENTATION: True          # Enable/disable data augmentation
IMAGE_SIZE: [224, 224, 3]   # Input image dimensions
BATCH_SIZE: 16              # Samples per batch
INCLUDE_TOP: False          # Exclude VGG16 top layers
EPOCHS: 1                   # Training epochs
CLASSES: 2                  # Number of output classes
WEIGHTS: imagenet           # Pre-trained weights
LEARNING_RATE: 0.01         # Adam optimizer learning rate
```

### Project Configuration (`config/config.yaml`)

```yaml
artifacts_root: artifacts

data_ingestion:
  root_dir: artifacts/data_ingestion
  source_URL: https://raw.githubusercontent.com/ajaychaudhary8104/data/main/Chicken-fecal-images.7z
  local_data_file: artifacts/data_ingestion/data.7z

prepare_base_model:
  root_dir: artifacts/prepare_base_model
  base_model_path: artifacts/prepare_base_model/base_model.h5
  updated_base_model_path: artifacts/prepare_base_model/base_model_updated.h5

training:
  root_dir: artifacts/training
  trained_model_path: artifacts/training/model.h5
```

### Modifying Hyperparameters
Edit `params.yaml` before training:

```bash
# Change epochs
EPOCHS: 25

# Change batch size
BATCH_SIZE: 32

# Disable augmentation
AUGMENTATION: False
```

---

## Dataset

### Overview
- **Source**: Chicken fecal microscopy images
- **Classes**: 2
  - **Healthy**: Normal chicken fecal samples
  - **Coccidiosis**: Infected with Coccidiosis parasite
- **Image Format**: JPG, PNG
- **Typical Size**: Variable (resized to 224×224)
- **Train/Test Split**: 80% training, 20% validation

### Data Directory Structure
```
artifacts/data_ingestion/Chicken-fecal-images/
├── Healthy/
│   ├── image_1.jpg
│   ├── image_2.jpg
│   └── ...
└── Coccidiosis/
    ├── image_1.jpg
    ├── image_2.jpg
    └── ...
```

### Data Augmentation
Applied during training:
- **Random Horizontal Flip**: 50% probability
- **Random Rotation**: ±10 degrees
- **Random Zoom**: ±10%

### Preprocessing
1. Load image at 224×224 size
2. Convert to numpy array
3. Normalize: divide by 255.0
4. Apply augmentation (if enabled)
5. One-hot encode labels

---

## API Documentation

### Health Check Endpoint (Optional)

```bash
GET /
```

Returns the web interface (index.html)

### Training Endpoint

```bash
POST /train
```

Starts the complete training pipeline.

**Response**:
```json
"Training done successfully!"
```

### Prediction Endpoint

```bash
POST /predict
```

Makes a disease prediction on uploaded image.

**Request**:
```json
{
  "image": "base64_string"
}
```

**Response**:
```json
[
  {
    "image": "Healthy",
    "prediction": {
      "Coccidiosis": 0.05,
      "Healthy": 0.95
    }
  }
]
```

**Status Codes**:
- `200`: Successful prediction
- `400`: Invalid request
- `500`: Server error

### Error Handling

The API returns descriptive error messages:

```json
{
  "error": "Invalid image format",
  "status": 400
}
```

---

## Troubleshooting

### Issue: ImportError for cnnClassifier
**Solution**: Install package in development mode:
```bash
pip install -e .
```

### Issue: Model Not Found (FileNotFoundError)
**Solution**: Train the model first:
```bash
python main.py
```

### Issue: Out of Memory (OOM) Error
**Solutions**:
1. Reduce batch size in `params.yaml`:
   ```yaml
   BATCH_SIZE: 8
   ```
2. Reduce image size (not recommended)
3. Use GPU with more VRAM

### Issue: Predictions Always Return Same Class
**Solution**: Verify image normalization in prediction.py:
```python
# Images must be normalized by 255.0
test_image = test_image / 255.0
```

### Issue: Flask Port Already in Use
**Solution**: Change port in app.py or kill existing process:
```bash
# Windows
netstat -ano | findstr :8080
taskkill /PID <PID> /F

# Linux/Mac
lsof -i :8080
kill -9 <PID>
```

### Issue: TensorBoard Not Loading
**Solution**: Verify log directory path and try:
```bash
tensorboard --logdir=./artifacts/prepare_callbacks/tensorboard_log_dir --host=localhost
```

### Issue: CUDA Not Detected
**Solutions**:
1. Install NVIDIA GPU drivers
2. Install CUDA Toolkit
3. Install cuDNN
4. Force CPU: `export CUDA_VISIBLE_DEVICES=-1`

---

## Performance Metrics

### Model Evaluation

After training, metrics are saved to `scores.json`:

```json
{
  "loss": 0.245,
  "accuracy": 0.92
}
```

### Key Metrics
- **Accuracy**: Percentage of correct predictions
- **Loss**: Categorical crossentropy loss
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall

### Performance Factors
- **Image Quality**: Clear, well-lit images improve accuracy
- **Dataset Size**: More images = better generalization
- **Training Epochs**: More epochs may improve accuracy (watch for overfitting)
- **Batch Size**: Affects convergence and GPU memory usage

---

## Contributing

### How to Contribute
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make your changes
4. Commit: `git commit -m "Add your feature"`
5. Push: `git push origin feature/your-feature`
6. Submit a Pull Request

### Areas for Contribution
- [ ] Improve model accuracy
- [ ] Add more disease classifications
- [ ] Enhance UI/UX
- [ ] Add real-time monitoring
- [ ] Create mobile app
- [ ] Improve documentation
- [ ] Add more test cases

### Code Style
- Follow PEP 8 guidelines
- Use meaningful variable names
- Add docstrings to functions
- Include type hints where possible

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Additional Resources

### Documentation
- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [VGG16 Paper](https://arxiv.org/abs/1409.1556)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [DVC Documentation](https://dvc.org/doc)

### Related Projects
- [PlantVillage Dataset](https://github.com/spMohanty/PlantVillageDataset)
- [Animal Disease Detection Models](https://github.com/topics/disease-detection)

### Tutorials
- [Transfer Learning with TensorFlow](https://www.tensorflow.org/tutorials/images/transfer_learning)
- [Building REST APIs with Flask](https://blog.miguelgrinberg.com/post/designing-a-restful-api-with-flask)

---

## Contact & Support

For issues, questions, or suggestions:
- Create an issue on GitHub
- Contact the development team
- Check existing documentation first

---

## Version History

- **v1.0.0** (January 2026)
  - Initial release
  - Binary classification (Healthy vs Coccidiosis)
  - Web interface and REST API
  - VGG16 transfer learning model

---

## Acknowledgments

- TensorFlow/Keras team for excellent deep learning framework
- Flask team for lightweight web framework
- ImageNet contributors for pre-trained weights
- Data contributors and researchers in poultry disease detection

---

**Last Updated**: January 1, 2026
**Maintainer**: Your Name/Organization
**Status**: Active Development
