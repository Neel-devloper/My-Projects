#!/usr/bin/env python3
"""
Test script for the trained cat and dog classification model.
Use this script to test individual images after training the model.
"""

import os
import sys
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

def load_and_preprocess_image(image_path, target_size=(160, 160)):
    """Load and preprocess a single image for prediction"""
    try:
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize image
        img_resized = cv2.resize(img, target_size)
        
        # Normalize to [0, 1]
        img_normalized = img_resized / 255.0
        
        # Add batch dimension
        img_batch = np.expand_dims(img_normalized, axis=0)
        
        return img_batch, img_resized
        
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None, None

def predict_image(model, image_path):
    """Predict whether an image contains a cat or dog"""
    try:
        # Load and preprocess image
        img_batch, img_original = load_and_preprocess_image(image_path)
        
        if img_batch is None:
            return None, None, None
        
        # Make prediction
        prediction = model.predict(img_batch, verbose=0)[0][0]
        
        # Determine class and confidence
        predicted_class = "Dog" if prediction > 0.5 else "Cat"
        confidence = prediction if prediction > 0.5 else 1 - prediction
        
        return predicted_class, confidence, img_original
        
    except Exception as e:
        print(f"Error making prediction: {e}")
        return None, None, None

def display_prediction(image_path, predicted_class, confidence, img_display):
    """Display the image with prediction results"""
    try:
        # Set matplotlib to use non-interactive backend to avoid hanging
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        
        plt.figure(figsize=(10, 8))
        
        # Display image
        plt.imshow(img_display)
        plt.axis('off')
        
        # Add prediction text
        title = f"Prediction: {predicted_class}\nConfidence: {confidence:.4f}"
        
        # Color code the title
        if predicted_class == "Dog":
            plt.title(title, color='blue', fontsize=16, fontweight='bold')
        else:
            plt.title(title, color='orange', fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        # Just close the figure without saving - no image files created
        plt.close()  # Close the figure to free memory
        
        print(f"✅ Prediction visualization created (no files saved)")
        
    except Exception as e:
        print(f"⚠️  Could not create prediction visualization: {e}")
        print("   Continuing without visualization...")

def main():
    # Check if model file exists
    model_path = 'cat_dog_classifier_model.h5'
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found!")
        print("Please train the model first by running 'python main.py'")
        return
    
    # Load the trained model
    print("Loading trained model...")
    try:
        model = load_model(model_path)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Test with sample images from the dataset
    dataset_path = "/Users/neelvorani/Desktop/Python Projects Main Dir/kagglecatsanddogs_3367a/PetImages"
    
    if os.path.exists(dataset_path):
        print("\nTesting with sample images from the dataset...")
        
        # Test a cat image
        cat_path = os.path.join(dataset_path, 'Cat')
        if os.path.exists(cat_path):
            cat_files = [f for f in os.listdir(cat_path) if f.endswith('.jpg')]
            if cat_files:
                test_cat = os.path.join(cat_path, cat_files[0])
                print(f"\nTesting cat image: {test_cat}")
                
                predicted_class, confidence, img_display = predict_image(model, test_cat)
                if predicted_class:
                    print(f"Prediction: {predicted_class}")
                    print(f"Confidence: {confidence:.4f}")
                    display_prediction(test_cat, predicted_class, confidence, img_display)
        
        # Test a dog image
        dog_path = os.path.join(dataset_path, 'Dog')
        if os.path.exists(dog_path):
            dog_files = [f for f in os.listdir(dog_path) if f.endswith('.jpg')]
            if dog_files:
                test_dog = os.path.join(dog_path, dog_files[0])
                print(f"\nTesting dog image: {test_dog}")
                
                predicted_class, confidence, img_display = predict_image(model, test_dog)
                if predicted_class:
                    print(f"Prediction: {predicted_class}")
                    print(f"Confidence: {confidence:.4f}")
                    display_prediction(test_dog, predicted_class, confidence, img_display)
    
    # Interactive testing
    print("\n" + "="*50)
    print("Interactive Testing Mode")
    print("="*50)
    print("Enter the path to an image file to test (or 'quit' to exit):")
    
    while True:
        try:
            user_input = input("\nImage path: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not user_input:
                continue
            
            if not os.path.exists(user_input):
                print(f"Error: File '{user_input}' not found!")
                continue
            
            if not user_input.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                print("Error: Please provide an image file (.jpg, .jpeg, .png, .bmp)")
                continue
            
            # Make prediction
            print(f"\n🔍 Analyzing image: {os.path.basename(user_input)}")
            predicted_class, confidence, img_display = predict_image(model, user_input)
            
            if predicted_class:
                print(f"\n🎯 Results:")
                print(f"   Prediction: {predicted_class}")
                print(f"   Confidence: {confidence:.4f}")
                print(f"   Image: {os.path.basename(user_input)}")
                
                # Display image with prediction
                display_prediction(user_input, predicted_class, confidence, img_display)
                
                print(f"\n✅ Successfully classified as {predicted_class}!")
            else:
                print("❌ Failed to make prediction for this image.")
                print("   Please try a different image or check the file format.")
                
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
