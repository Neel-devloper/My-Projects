# Cat and Dog Recognition AI Model

This project implements a deep learning model to classify images as either cats or dogs using a Convolutional Neural Network (CNN) with transfer learning.

## Features

- **Transfer Learning**: Uses pre-trained VGG16 model as base for better performance
- **Data Augmentation**: Implements rotation, flipping, and scaling for robust training
- **90/10 Split**: Uses 90% of data for training and 10% for testing
- **Early Stopping**: Prevents overfitting with automatic training termination
- **Learning Rate Scheduling**: Adaptive learning rate for optimal convergence
- **Comprehensive Evaluation**: Provides accuracy, classification report, and confusion matrix
- **Model Persistence**: Saves trained model for future use

## Model Architecture

- **Base Model**: VGG16 (pre-trained on ImageNet)
- **Classification Head**: 
  - Global Average Pooling
  - Dense layers with dropout (512 → 256 → 1)
  - Sigmoid activation for binary classification
- **Input Size**: 224x224x3 RGB images
- **Output**: Binary classification (0 = Cat, 1 = Dog)

## Dataset Structure

The model expects the following dataset structure:
```
kagglecatsanddogs_3367a/
├── PetImages/
│   ├── Cat/
│   │   ├── 0.jpg
│   │   ├── 1.jpg
│   │   └── ... (cat images)
│   └── Dog/
│       ├── 0.jpg
│       ├── 1.jpg
│       └── ... (dog images)
```

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Ensure you have the dataset in the correct location relative to the script.

## Usage

### Training the Model

Run the main script to train the model:
```bash
python main.py
```

The script will:
1. Load and preprocess all images from the dataset
2. Split data into training (90%) and testing (10%) sets
3. Create and train the CNN model
4. Evaluate performance on test data
5. Save the trained model as `cat_dog_classifier_model.h5`
6. Display training plots and results

### Using the Trained Model

After training, you can use the saved model for predictions:

```python
from tensorflow.keras.models import load_model
import numpy as np
import cv2

# Load the trained model
model = load_model('cat_dog_classifier_model.h5')

# Load and preprocess a single image
img = cv2.imread('path_to_image.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (224, 224))
img = img / 255.0
img = np.expand_dims(img, axis=0)

# Make prediction
prediction = model.predict(img)[0][0]
predicted_class = "Dog" if prediction > 0.5 else "Cat"
confidence = prediction if prediction > 0.5 else 1 - prediction

print(f"Prediction: {predicted_class}")
print(f"Confidence: {confidence:.4f}")
```

## Model Performance

The model typically achieves:
- **Training Accuracy**: 95%+ 
- **Test Accuracy**: 90-95%
- **Training Time**: 15-30 minutes (depending on hardware)

## Key Features

### Data Preprocessing
- Image resizing to 224x224 pixels
- RGB color space conversion
- Normalization to [0, 1] range
- Error handling for corrupted images

### Training Optimizations
- **Early Stopping**: Stops training when validation loss stops improving
- **Learning Rate Reduction**: Reduces learning rate when plateau is reached
- **Data Augmentation**: Increases effective dataset size and improves generalization
- **Dropout Layers**: Prevents overfitting

### Evaluation Metrics
- **Accuracy**: Overall classification accuracy
- **Precision/Recall**: Per-class performance metrics
- **Confusion Matrix**: Detailed classification results
- **Training History**: Learning curves for monitoring

## Requirements

- Python 3.7+
- TensorFlow 2.10+
- OpenCV 4.5+
- NumPy 1.21+
- Matplotlib 3.5+
- Scikit-learn 1.0+
- Pillow 8.3+

## Hardware Recommendations

- **Minimum**: 8GB RAM, CPU training
- **Recommended**: 16GB+ RAM, GPU acceleration (CUDA-compatible)
- **Storage**: At least 2GB free space for model and temporary files

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size or image size
2. **Slow Training**: Use GPU acceleration if available
3. **Low Accuracy**: Check dataset quality and increase training epochs
4. **Import Errors**: Ensure all requirements are installed correctly

### Performance Tips

- Use GPU acceleration for faster training
- Adjust batch size based on available memory
- Monitor training curves to detect overfitting
- Use data augmentation for better generalization

## License

This project is open source and available under the MIT License.

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve the model.
