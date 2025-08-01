#!/usr/bin/env python3
"""
Script to pull the smaller phi:2.7b model for Veda
This model is only ~1.7GB and perfect for low-end computers.
"""

import requests
import subprocess
import sys

def pull_phi_model():
    """Pull the phi:2.7b model"""
    print("🚀 Pulling phi:2.7b model (1.7GB)...")
    print("This model is perfect for low-end computers!")
    
    try:
        # Check if Ollama is running
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code != 200:
            print("❌ Ollama is not running. Please start it with: ollama serve")
            return False
            
        # Pull the model
        print("📥 Downloading phi:2.7b... This may take a few minutes.")
        response = requests.post("http://localhost:11434/api/pull", 
                               json={"name": "phi:2.7b"})
        
        if response.status_code == 200:
            print("✅ Successfully pulled phi:2.7b model!")
            print("🎉 You can now use Veda with the lightweight model.")
            return True
        else:
            print(f"❌ Failed to pull model: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to Ollama. Make sure it's running.")
        print("💡 Start Ollama with: ollama serve")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    print("=" * 50)
    print("📦 Veda - Small Model Setup")
    print("=" * 50)
    print("This will download phi:2.7b (~1.7GB) for Veda.")
    print("Perfect for low-end computers!")
    print("=" * 50)
    
    if pull_phi_model():
        print("\n🎉 Setup complete!")
        print("📋 Next steps:")
        print("1. Start Veda: streamlit run app_simple.py")
        print("2. Upload a dataset and start analyzing!")
    else:
        print("\n❌ Setup failed. Please check the errors above.")

if __name__ == "__main__":
    main() 