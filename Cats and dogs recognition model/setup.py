#!/usr/bin/env python3
"""
Setup script for the Cat and Dog Recognition AI Model.
This script helps install dependencies and prepare the environment.
"""

import subprocess
import sys
import os

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 7):
        print("❌ Error: Python 3.7 or higher is required!")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True

def install_requirements():
    """Install required packages from requirements.txt"""
    print("\n📦 Installing required packages...")
    
    try:
        # Upgrade pip first
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        
        # Install requirements
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All packages installed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        return False

def check_dataset():
    """Check if the dataset is available"""
    print("\n🔍 Checking dataset availability...")
    
    dataset_path = "/Users/neelvorani/Desktop/Python Projects Main Dir/kagglecatsanddogs_3367a/PetImages"
    
    if os.path.exists(dataset_path):
        cat_path = os.path.join(dataset_path, "Cat")
        dog_path = os.path.join(dataset_path, "Dog")
        
        if os.path.exists(cat_path) and os.path.exists(dog_path):
            cat_count = len([f for f in os.listdir(cat_path) if f.endswith('.jpg')])
            dog_count = len([f for f in os.listdir(dog_path) if f.endswith('.jpg')])
            
            print(f"✅ Dataset found!")
            print(f"   Cat images: {cat_count}")
            print(f"   Dog images: {dog_count}")
            print(f"   Total: {cat_count + dog_count}")
            return True
        else:
            print("❌ Dataset structure is incomplete!")
            print("   Expected: Cat/ and Dog/ folders with .jpg images")
            return False
    else:
        print("❌ Dataset not found!")
        print(f"   Expected path: {dataset_path}")
        print("   Please ensure the dataset is in the correct location")
        return False

def main():
    """Main setup function"""
    print("🚀 Setting up Cat and Dog Recognition AI Model")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return
    
    # Check if requirements.txt exists
    if not os.path.exists("requirements.txt"):
        print("❌ requirements.txt not found!")
        return
    
    # Install requirements
    if not install_requirements():
        return
    
    # Check dataset
    if not check_dataset():
        print("\n⚠️  Dataset not found. You can still install the model, but training won't work.")
        print("   Please download the dataset and place it in the correct location.")
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Run 'python main.py' to train the model")
    print("2. Run 'python test_model.py' to test the trained model")
    print("\nFor more information, see README.md")

if __name__ == "__main__":
    main()
