#!/usr/bin/env python3
"""
Setup script for Ollama integration with Veda
This script helps users install and configure Ollama for local LLM processing.
"""

import os
import sys
import subprocess
import platform
import requests
import time

def print_banner():
    """Print setup banner"""
    print("=" * 60)
    print("🚀 Veda - Ollama Setup")
    print("=" * 60)
    print("This script will help you set up Ollama for local LLM processing.")
    print("Ollama allows you to run large language models locally on your computer.")
    print("=" * 60)

def check_ollama_installed():
    """Check if Ollama is already installed"""
    try:
        result = subprocess.run(['ollama', '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"✅ Ollama is already installed: {result.stdout.strip()}")
            return True
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return False

def install_ollama():
    """Install Ollama based on the operating system"""
    system = platform.system().lower()
    
    print(f"🔧 Installing Ollama for {system}...")
    
    if system == "windows":
        print("📥 Downloading Ollama for Windows...")
        print("Please visit: https://ollama.ai/download")
        print("Download and install the Windows installer.")
        print("After installation, restart your terminal and run this script again.")
        return False
        
    elif system == "darwin":  # macOS
        try:
            subprocess.run(['brew', 'install', 'ollama'], check=True)
            print("✅ Ollama installed successfully via Homebrew")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ Homebrew not found. Please install Homebrew first:")
            print("Visit: https://brew.sh")
            return False
            
    elif system == "linux":
        try:
            # Install using the official install script
            install_cmd = "curl -fsSL https://ollama.ai/install.sh | sh"
            print("📥 Installing Ollama using official install script...")
            subprocess.run(install_cmd, shell=True, check=True)
            print("✅ Ollama installed successfully")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to install Ollama automatically.")
            print("Please install manually:")
            print("Visit: https://ollama.ai/download")
            return False
    
    else:
        print(f"❌ Unsupported operating system: {system}")
        print("Please install Ollama manually from: https://ollama.ai/download")
        return False

def start_ollama_service():
    """Start the Ollama service"""
    try:
        print("🚀 Starting Ollama service...")
        subprocess.run(['ollama', 'serve'], check=True)
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to start Ollama service")
        return False

def check_ollama_running():
    """Check if Ollama service is running"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print("✅ Ollama service is running")
            return True
    except requests.exceptions.RequestException:
        pass
    return False

def pull_recommended_model():
    """Pull a recommended lightweight model"""
    models = [
        "llama2:7b",      # Good balance of performance and resource usage
        "llama2:13b",     # Better performance, more resources
        "codellama:7b",   # Good for code-related tasks
        "mistral:7b",     # Fast and efficient
        "phi:2.7b"        # Very lightweight, good for low-end computers
    ]
    
    print("\n📚 Available lightweight models:")
    for i, model in enumerate(models, 1):
        print(f"  {i}. {model}")
    
    print("\n💡 Recommended for low-end computers:")
    print("  - phi:2.7b (very lightweight, ~1.7GB)")
    print("  - mistral:7b (good performance, ~4GB)")
    print("  - llama2:7b (balanced, ~4GB)")
    
    try:
        choice = input("\nEnter model number to download (or press Enter for phi:2.7b): ").strip()
        if not choice:
            selected_model = "phi:2.7b"
        else:
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(models):
                selected_model = models[choice_idx]
            else:
                selected_model = "phi:2.7b"
    except (ValueError, KeyboardInterrupt):
        selected_model = "phi:2.7b"
    
    print(f"\n📥 Pulling {selected_model}...")
    print("This may take several minutes depending on your internet connection.")
    
    try:
        subprocess.run(['ollama', 'pull', selected_model], check=True)
        print(f"✅ Successfully downloaded {selected_model}")
        
        # Update the LLM processor configuration
        update_config(selected_model)
        
        return selected_model
    except subprocess.CalledProcessError:
        print(f"❌ Failed to download {selected_model}")
        return None

def update_config(model_name):
    """Update the LLM processor configuration to use the selected model"""
    config_content = f'''# Ollama Configuration
# This file contains the default model configuration for Veda

DEFAULT_MODEL = "{model_name}"
OLLAMA_BASE_URL = "http://localhost:11434"

# Available models for different use cases:
# - phi:2.7b: Very lightweight, good for low-end computers (~1.7GB)
# - mistral:7b: Fast and efficient (~4GB)
# - llama2:7b: Balanced performance (~4GB)
# - llama2:13b: Better performance, more resources (~7GB)
# - codellama:7b: Good for code-related tasks (~4GB)
'''
    
    try:
        with open('ollama_config.py', 'w') as f:
            f.write(config_content)
        print("✅ Configuration updated")
    except Exception as e:
        print(f"⚠️ Could not update configuration: {e}")

def test_model():
    """Test the installed model"""
    print("\n🧪 Testing the model...")
    
    try:
        from llm_processor import OllamaLLMProcessor
        
        # Initialize with the default model
        processor = OllamaLLMProcessor()
        
        # Test with a simple prompt
        test_prompt = "Hello, can you help me with data analysis?"
        response = processor._generate_response(test_prompt, max_tokens=50)
        
        if response:
            print("✅ Model is working correctly!")
            print(f"Test response: {response[:100]}...")
            return True
        else:
            print("❌ Model test failed")
            return False
            
    except Exception as e:
        print(f"❌ Error testing model: {e}")
        return False

def main():
    """Main setup function"""
    print_banner()
    
    # Check if Ollama is already installed
    if not check_ollama_installed():
        print("\n📦 Ollama not found. Installing...")
        if not install_ollama():
            print("\n❌ Setup incomplete. Please install Ollama manually.")
            print("Visit: https://ollama.ai/download")
            return
    
    # Check if Ollama service is running
    if not check_ollama_running():
        print("\n🚀 Starting Ollama service...")
        print("Note: Ollama service needs to run in the background.")
        print("You can start it manually with: ollama serve")
        
        # Try to start the service
        if not start_ollama_service():
            print("❌ Could not start Ollama service automatically.")
            print("Please start it manually with: ollama serve")
            return
    
    # Pull a recommended model
    model_name = pull_recommended_model()
    if not model_name:
        print("❌ Failed to download model. Setup incomplete.")
        return
    
    # Test the model
    if test_model():
        print("\n🎉 Setup completed successfully!")
        print("\n📋 Next steps:")
        print("1. Start Veda with: streamlit run app.py")
        print("2. Upload a dataset and start analyzing!")
        print("3. Use voice commands or text input for analysis")
        print("\n💡 Tips:")
        print("- Keep Ollama running in the background")
        print("- You can change models in the app settings")
        print("- For better performance, close other applications")
    else:
        print("\n❌ Setup completed but model test failed.")
        print("Please check if Ollama is running: ollama serve")

if __name__ == "__main__":
    main() 