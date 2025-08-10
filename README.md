# VEDA - Exploratory Data Analysis Assistant

A desktop application for performing Exploratory Data Analysis (EDA) using natural language commands.

## Features

- Upload and analyze CSV datasets
- Voice command recognition using OpenAI Whisper (Currently using VOSK libraries to analyse speech)
- Natural language processing of analysis commands
- Interactive visualizations and statistical analysis
- Clean, minimal GUI built with Streamlit

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```
## Images 
![Screenshot 2025-05-09 160948](https://github.com/user-attachments/assets/b0c44042-1608-4c8d-84a1-e092cc89627f)
![Screenshot 2025-05-09 160912](https://github.com/user-attachments/assets/68019efd-ec32-40b1-b5ce-e24ffc4bd1ea)
![Screenshot 2025-05-09 160858](https://github.com/user-attachments/assets/b91eff90-fb1f-487b-ae41-d1cf74c14c8d)
![Screenshot 2025-05-09 160849](https://github.com/user-attachments/assets/ef316050-7071-47f3-aaf6-6c1b15bc0271)
![Screenshot 2025-05-09 160835](https://github.com/user-attachments/assets/5180075d-afa1-461c-951b-8f10822861e8)
![Screenshot 2025-05-09 154724](https://github.com/user-attachments/assets/694c9218-b174-48e9-8c95-e5d6b4d85485)

## Usage

1. Launch the application
2. Upload your CSV dataset using the file upload widget
3. Click the "Start Listening" button or press the spacebar to begin voice recognition
4. Speak your analysis commands naturally
5. View the results in the main panel

## Example Commands

- "Show summary statistics"
- "Plot a histogram of age"
- "Create a boxplot for salary"
- "Display the correlation matrix"
- "Show missing value heatmap"
- "Run linear regression with price as target and area, rooms as features"

## Requirements

- Python 3.10 or higher
- Microphone for voice input
- Speakers for optional text-to-speech output

## Note

The application runs entirely locally, with no cloud dependencies. Voice recognition is handled by OpenAI's Whisper model running on your machine, (also working on fixing 
The issues regarding speech recognition actively. 
