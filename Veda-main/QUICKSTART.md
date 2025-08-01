# 🚀 Quick Start Guide - Veda with Local LLM

Get Veda running with local AI processing in 5 minutes!

## ⚡ Quick Setup (5 minutes)

### 1. Install Ollama
Visit [ollama.ai/download](https://ollama.ai/download) and install Ollama for your system.

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Setup Script
```bash
python setup_ollama.py
```
This will:
- ✅ Check Ollama installation
- ✅ Download a lightweight model (phi:2.7b)
- ✅ Test the setup

### 4. Start Ollama Service
```bash
ollama serve
```
Keep this running in the background.

### 5. Launch Veda
```bash
streamlit run app.py
```

## 🎯 Recommended Models for Different Computers

### Low-End Computers (4GB RAM)
```bash
ollama pull phi:2.7b
```
- **Size**: ~1.7GB
- **Speed**: Fast
- **Quality**: Good for basic tasks

### Mid-Range Computers (8GB RAM)
```bash
ollama pull mistral:7b
```
- **Size**: ~4GB
- **Speed**: Balanced
- **Quality**: Better performance

### High-End Computers (16GB+ RAM)
```bash
ollama pull llama2:13b
```
- **Size**: ~7GB
- **Speed**: Slower but more capable
- **Quality**: Best performance

## 🎤 Quick Voice Commands to Try

1. **"Show summary statistics"** - Basic dataset overview
2. **"Plot histogram of age"** - Create a visualization
3. **"Show missing values"** - Data quality check
4. **"Create correlation matrix"** - Find relationships
5. **"Detect outliers in salary"** - Advanced analysis

## 🔧 Troubleshooting

### Ollama Not Starting
```bash
# Check if Ollama is installed
ollama --version

# Start Ollama service
ollama serve

# Check if it's running
curl http://localhost:11434/api/tags
```

### No Models Available
```bash
# List available models
ollama list

# Pull a lightweight model
ollama pull phi:2.7b
```

### Slow Performance
1. Switch to a lighter model: `phi:2.7b`
2. Close other applications
3. Check system resources

## 📊 Example Workflow

1. **Upload Dataset**: Use the file uploader
2. **Explore Data**: "Show summary statistics"
3. **Clean Data**: "Handle missing values"
4. **Visualize**: "Plot histogram of [column]"
5. **Analyze**: "Run linear regression with [target]"

## 💡 Tips for Best Performance

- **Use phi:2.7b** for low-end computers
- **Keep Ollama running** in the background
- **Close other applications** when analyzing large datasets
- **Use voice commands** for faster interaction
- **Check the sidebar** for quick action buttons

## 🆘 Need Help?

1. Run the test script: `python test_ollama.py`
2. Check the troubleshooting section in README.md
3. Ensure Ollama is running: `ollama serve`
4. Try a different model if performance is poor

---

**Ready to analyze your data with local AI! 🚀** 