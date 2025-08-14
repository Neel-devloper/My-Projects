import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import cv2
import random

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# Configure TensorFlow for better performance
tf.config.optimizer.set_jit(True)  # Enable XLA optimization
tf.config.threading.set_inter_op_parallelism_threads(4)  # Parallel processing
tf.config.threading.set_intra_op_parallelism_threads(4)

class CatDogClassifier:
    def __init__(self, data_path, img_size=(160, 160)):  # Reduced from 224x224 to 160x160
        self.data_path = data_path
        self.img_size = img_size
        self.model = None
        self.history = None
        
    def get_data_generators(self, batch_size=128):  # Increased from 32 to 128
        """Create data generators for training and validation"""
        print("Creating data generators...")
        
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rotation_range=15,  # Reduced from 20
            width_shift_range=0.15,  # Reduced from 0.2
            height_shift_range=0.15,  # Reduced from 0.2
            horizontal_flip=True,
            zoom_range=0.15,  # Reduced from 0.2
            fill_mode='nearest',
            validation_split=0.1,  # 10% for validation
            rescale=1./255  # Normalize to [0, 1]
        )
        
        # No augmentation for validation
        val_datagen = ImageDataGenerator(
            validation_split=0.1,
            rescale=1./255
        )
        
        # Training generator
        train_generator = train_datagen.flow_from_directory(
            self.data_path + '/PetImages',
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='binary',
            subset='training',
            shuffle=True
        )
        
        # Validation generator
        val_generator = val_datagen.flow_from_directory(
            self.data_path + '/PetImages',
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='binary',
            subset='validation',
            shuffle=False
        )
        
        print(f"Found {train_generator.samples} training samples")
        print(f"Found {val_generator.samples} validation samples")
        print(f"Class indices: {train_generator.class_indices}")
        print(f"Batch size: {batch_size}")
        print(f"Image size: {self.img_size}")
        
        return train_generator, val_generator
    
    def create_model(self):
        """Create a faster, lighter CNN model for cat/dog classification"""
        print("Creating optimized model...")
        
        # Use MobileNetV2 instead of VGG16 (much faster and lighter)
        base_model = MobileNetV2(
            weights='imagenet', 
            include_top=False, 
            input_shape=(*self.img_size, 3),
            alpha=0.75  # Smaller model for speed
        )
        
        # Freeze the base model layers
        base_model.trainable = False
        
        # Create a lighter, faster model
        self.model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.3),  # Reduced from 0.5
            layers.Dense(256, activation='relu'),  # Reduced from 512
            layers.Dropout(0.2),  # Reduced from 0.3
            layers.Dense(128, activation='relu'),  # Reduced from 256
            layers.Dropout(0.2),
            layers.Dense(1, activation='sigmoid')  # Binary classification
        ])
        
        # Use a faster optimizer
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),  # Slightly higher learning rate
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        print("Model summary:")
        self.model.summary()
        
        return self.model
    
    def train_model(self, train_generator, val_generator, epochs=10, batch_size=128):  # Reduced epochs from 15 to 10
        """Train the model using data generators"""
        print("Training model...")
        
        # Calculate steps per epoch
        steps_per_epoch = train_generator.samples // batch_size
        validation_steps = val_generator.samples // batch_size
        
        print(f"Steps per epoch: {steps_per_epoch}")
        print(f"Validation steps: {validation_steps}")
        print(f"Total training steps: {steps_per_epoch * epochs}")
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=3,  # Reduced from 5 for faster training
            restore_best_weights=True
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.3,  # More aggressive reduction
            patience=2,  # Reduced from 3
            min_lr=1e-6
        )
        
        # Train the model
        self.history = self.model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=val_generator,
            validation_steps=validation_steps,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        return self.history
    
    def evaluate_model(self, test_generator):
        """Evaluate the model on test data"""
        print("Evaluating model...")
        
        # Reset generator
        test_generator.reset()
        
        # Make predictions
        predictions = self.model.predict(test_generator, verbose=1)
        y_pred = (predictions > 0.5).astype(int).flatten()
        
        # Get true labels
        test_generator.reset()
        y_true = test_generator.classes[:len(y_pred)]
        
        # Calculate accuracy
        accuracy = accuracy_score(y_true, y_pred)
        print(f"\nTest Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=['Cat', 'Dog']))
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        print("\nConfusion Matrix:")
        print(cm)
        
        return accuracy, y_pred
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            print("No training history available")
            return
        
        try:
            # Set matplotlib to use non-interactive backend to avoid hanging
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # Plot accuracy
            ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
            ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
            ax1.set_title('Model Accuracy')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Accuracy')
            ax1.legend()
            ax1.grid(True)
            
            # Plot loss
            ax2.plot(self.history.history['loss'], label='Training Loss')
            ax2.plot(self.history.history['val_loss'], label='Validation Loss')
            ax2.set_title('Model Loss')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss')
            ax2.legend()
            ax2.grid(True)
            
            plt.tight_layout()
            
            # Save the plot instead of showing it to avoid hanging
            plot_filename = 'training_history.png'
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            plt.close()  # Close the figure to free memory
            
            print(f"✅ Training history plot saved as '{plot_filename}'")
            print("   (Plot saved to avoid display hanging issues)")
            
        except Exception as e:
            print(f"⚠️  Could not create training plot: {e}")
            print("   Continuing without visualization...")
    
    def predict_single_image(self, image_path):
        """Predict a single image"""
        try:
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.img_size)
            img = img / 255.0
            img = np.expand_dims(img, axis=0)
            
            prediction = self.model.predict(img, verbose=0)[0][0]
            predicted_class = "Dog" if prediction > 0.5 else "Cat"
            confidence = prediction if prediction > 0.5 else 1 - prediction
            
            print(f"Prediction: {predicted_class}")
            print(f"Confidence: {confidence:.4f}")
            
            return predicted_class, confidence
            
        except Exception as e:
            print(f"Error predicting image: {e}")
            return None, None

def main():
    # Initialize the classifier
    data_path = "/Users/neelvorani/Desktop/Python Projects Main Dir/kagglecatsanddogs_3367a"
    classifier = CatDogClassifier(data_path)
    
    # Create data generators with optimized settings
    train_generator, val_generator = classifier.get_data_generators(batch_size=128)
    
    # Create and train the model
    model = classifier.create_model()
    history = classifier.train_model(train_generator, val_generator, epochs=10, batch_size=128)
    
    # Create test generator for evaluation
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        data_path + '/PetImages',
        target_size=(160, 160),  # Match the new image size
        batch_size=128,  # Larger batch size for faster evaluation
        class_mode='binary',
        shuffle=False
    )
    
    # Evaluate the model
    accuracy, predictions = classifier.evaluate_model(test_generator)
    
    # Plot training history
    print("\n📊 Creating training history visualization...")
    classifier.plot_training_history()
    
    # Save the model
    print("\n💾 Saving trained model...")
    model.save('cat_dog_classifier_model.h5')
    print("✅ Model saved as 'cat_dog_classifier_model.h5'")
    
    print("\n🎯 Training and evaluation completed successfully!")
    print("=" * 60)
    
    return classifier, accuracy

if __name__ == "__main__":
    try:
        classifier, accuracy = main()
        print(f"\n🎉 Training completed successfully!")
        print(f"Final test accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print("\n🚀 You can now use the trained model!")
        print("   - Model file: cat_dog_classifier_model.h5")
        print("   - Training plot: training_history.png")
        print("   - Run 'python test_model.py' to test individual images")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
