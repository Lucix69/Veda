#!/usr/bin/env python3
"""
Test script for Ollama integration with Veda
This script tests the local LLM functionality.
"""

import requests
import time
from llm_processor import OllamaLLMProcessor

def test_ollama_connection():
    """Test if Ollama is running and accessible"""
    print("🔍 Testing Ollama connection...")
    
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            print(f"✅ Ollama is running. Found {len(models)} models:")
            for model in models:
                print(f"   - {model['name']}")
            return True
        else:
            print(f"❌ Ollama responded with status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to Ollama. Make sure it's running with: ollama serve")
        return False
    except Exception as e:
        print(f"❌ Error connecting to Ollama: {e}")
        return False

def test_model_generation():
    """Test if a model can generate responses"""
    print("\n🧪 Testing model generation...")
    
    try:
        # Initialize processor with a lightweight model
        processor = OllamaLLMProcessor(model_name="phi:2.7b")
        
        # Test with a simple prompt
        test_prompt = "Hello, can you help me with data analysis? Please respond briefly."
        print(f"📝 Testing prompt: {test_prompt}")
        
        response = processor._generate_response(test_prompt, max_tokens=100)
        
        if response:
            print("✅ Model generated a response successfully!")
            print(f"📄 Response: {response[:200]}...")
            return True
        else:
            print("❌ Model did not generate a response")
            return False
            
    except Exception as e:
        print(f"❌ Error testing model generation: {e}")
        return False

def test_dataset_analysis():
    """Test dataset analysis functionality"""
    print("\n📊 Testing dataset analysis...")
    
    try:
        import pandas as pd
        import numpy as np
        
        # Create a sample dataset
        data = {
            'age': np.random.randint(20, 70, 100),
            'salary': np.random.randint(30000, 100000, 100),
            'department': np.random.choice(['IT', 'HR', 'Sales', 'Marketing'], 100),
            'experience': np.random.randint(1, 20, 100)
        }
        df = pd.DataFrame(data)
        
        # Initialize processor
        processor = OllamaLLMProcessor(model_name="phi:2.7b")
        
        # Test dataset analysis
        analysis = processor.analyze_dataset(df)
        
        if analysis and 'data_quality' in analysis:
            print("✅ Dataset analysis completed successfully!")
            print(f"📈 Dataset has {analysis['data_quality']['total_rows']} rows and {analysis['data_quality']['total_columns']} columns")
            if analysis.get('insights'):
                print("💡 Generated insights available")
            return True
        else:
            print("❌ Dataset analysis failed")
            return False
            
    except Exception as e:
        print(f"❌ Error testing dataset analysis: {e}")
        return False

def test_command_processing():
    """Test command processing functionality"""
    print("\n🎤 Testing command processing...")
    
    try:
        processor = OllamaLLMProcessor(model_name="phi:2.7b")
        
        # Test command processing
        test_command = "show summary statistics"
        available_columns = ["age", "salary", "department", "experience"]
        
        result, success = processor.process_command(test_command, available_columns)
        
        if success and result:
            print("✅ Command processing completed successfully!")
            print(f"📋 Command type: {result.get('command_type', 'unknown')}")
            return True
        else:
            print("❌ Command processing failed")
            return False
            
    except Exception as e:
        print(f"❌ Error testing command processing: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("🧪 Veda - Ollama Integration Test")
    print("=" * 60)
    
    tests = [
        ("Ollama Connection", test_ollama_connection),
        ("Model Generation", test_model_generation),
        ("Dataset Analysis", test_dataset_analysis),
        ("Command Processing", test_command_processing)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with error: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Ollama integration is working correctly.")
        print("\n📋 Next steps:")
        print("1. Start Veda with: streamlit run app.py")
        print("2. Upload a dataset and start analyzing!")
    else:
        print("⚠️ Some tests failed. Please check the issues above.")
        print("\n🔧 Troubleshooting:")
        print("1. Make sure Ollama is running: ollama serve")
        print("2. Check if models are available: ollama list")
        print("3. Pull a model if needed: ollama pull phi:2.7b")
        print("4. Run setup script: python setup_ollama.py")

if __name__ == "__main__":
    main() 