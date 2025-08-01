# 📊 Veda - Data Analysis Assistant

Advanced data analysis and visualization platform powered by local AI. Veda now uses Ollama to run large language models locally on your computer, providing privacy and offline capabilities.

## 🚀 Features

- **Local AI Processing**: Run LLMs locally using Ollama
- **Voice Commands**: Natural language data analysis through voice input
- **Text Commands**: Type commands for data analysis
- **Interactive Visualizations**: Create charts, plots, and visualizations
- **Data Preprocessing**: Handle missing values, duplicates, and data cleaning
- **Statistical Analysis**: Comprehensive statistical insights
- **Model Selection**: Choose from various lightweight models for different use cases

## 🎯 Local LLM Models

Veda supports multiple lightweight models that can run on low-end computers:

| Model | Size | Use Case | Performance |
|-------|------|----------|-------------|
| **phi:2.7b** | ~1.7GB | Very lightweight, low-end computers | Fast, basic tasks |
| **mistral:7b** | ~4GB | Balanced performance | Good all-around |
| **llama2:7b** | ~4GB | General purpose | Reliable, well-tested |
| **codellama:7b** | ~4GB | Code-related tasks | Good for data analysis |
| **llama2:13b** | ~7GB | Better performance | Higher resource usage |

## 📦 Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd Veda-main
```

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 3. Install Ollama
Visit [ollama.ai/download](https://ollama.ai/download) and install Ollama for your operating system.

### 4. Run Setup Script
```bash
python setup_ollama.py
```

This script will:
- Check if Ollama is installed
- Install Ollama if needed
- Download a recommended lightweight model
- Test the setup

### 5. Start Ollama Service
```bash
ollama serve
```

Keep this running in the background while using Veda.

## 🚀 Quick Start

1. **Start the Application**:
   ```bash
   streamlit run app.py
   ```

2. **Upload a Dataset**: Use the file uploader to upload your CSV file

3. **Use Voice or Text Commands**: 
   - Click the microphone button for voice input
   - Type commands in the text input field
   - Use the sidebar buttons for quick actions

4. **Explore Your Data**: 
   - View dataset insights
   - Create visualizations
   - Perform statistical analysis

## 🎤 Voice Commands

Veda supports natural language commands for data analysis:

### Basic Analysis
- "Show summary statistics"
- "Show top 5 rows"
- "Show missing values"
- "Show data types"

### Visualizations
- "Plot histogram of [column name]"
- "Create boxplot for [column name]"
- "Show correlation matrix"
- "Create scatter plot of [x] vs [y]"

### Data Preprocessing
- "Handle missing values"
- "Remove duplicates"
- "Normalize [column name]"
- "Encode [column name]"

### Advanced Analysis
- "Detect outliers in [column name]"
- "Run linear regression with [target column]"
- "Show feature importance"
- "Analyze relationships"

## ⚙️ Configuration

### Model Selection
In the app sidebar, expand "🤖 Ollama Settings" to:
- View available models
- Switch between different models
- Check Ollama status
- Get model recommendations

### System Requirements

#### Minimum Requirements (for phi:2.7b)
- **RAM**: 4GB
- **Storage**: 2GB free space
- **CPU**: Dual-core processor
- **OS**: Windows 10+, macOS 10.14+, or Linux

#### Recommended Requirements (for larger models)
- **RAM**: 8GB+
- **Storage**: 8GB+ free space
- **CPU**: Quad-core processor
- **GPU**: Optional, for faster inference

## 🔧 Troubleshooting

### Ollama Not Running
```bash
# Start Ollama service
ollama serve

# Check if it's running
curl http://localhost:11434/api/tags
```

### Model Not Found
```bash
# List available models
ollama list

# Pull a specific model
ollama pull phi:2.7b
```

### Performance Issues
1. **Use a lighter model**: Switch to `phi:2.7b` for low-end computers
2. **Close other applications**: Free up RAM and CPU
3. **Check system resources**: Monitor CPU and memory usage
4. **Restart Ollama**: Sometimes helps with memory issues

### Common Issues

#### "Connection refused" error
- Make sure Ollama is running: `ollama serve`
- Check if port 11434 is available
- Restart Ollama service

#### "Model not found" error
- Pull the model: `ollama pull phi:2.7b`
- Check available models: `ollama list`
- Run setup script: `python setup_ollama.py`

#### Slow performance
- Switch to a lighter model (phi:2.7b)
- Close other applications
- Check system resources
- Consider upgrading RAM

## 📊 Example Usage

### 1. Upload Dataset
Upload a CSV file containing your data.

### 2. Explore Data
```bash
# Voice command: "Show summary statistics"
# This will display basic statistics for all columns
```

### 3. Handle Data Quality
```bash
# Voice command: "Handle missing values"
# This will analyze and suggest ways to handle missing data
```

### 4. Create Visualizations
```bash
# Voice command: "Plot histogram of age"
# This will create a histogram for the age column
```

### 5. Advanced Analysis
```bash
# Voice command: "Run linear regression with target_column"
# This will perform linear regression analysis
```

## 🛠️ Development

### Project Structure
```
Veda-main/
├── app.py              # Main Streamlit application
├── llm_processor.py    # Local LLM processing with Ollama
├── utils.py            # Utility functions
├── setup_ollama.py     # Ollama setup script
├── requirements.txt     # Python dependencies
└── README.md          # This file
```

### Adding New Models
1. Pull the model: `ollama pull model_name`
2. The model will appear in the app's model selection
3. Update the setup script if needed

### Customizing Commands
Edit `utils.py` to add new command patterns and their corresponding actions.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with different models
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- [Ollama](https://ollama.ai) for local LLM infrastructure
- [Streamlit](https://streamlit.io) for the web interface
- [Pandas](https://pandas.pydata.org) for data manipulation
- [Matplotlib](https://matplotlib.org) for visualizations

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Ensure Ollama is running: `ollama serve`
3. Try a different model if performance is poor
4. Check system requirements for your chosen model

---

**Made with ❤️ for data analysis enthusiasts** 
