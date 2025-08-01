import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from llm_processor import LLMProcessor
import io
import base64
import time

# Set page config
st.set_page_config(
    page_title="Veda - Data Analysis Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        margin: 0.5rem 0;
        background-color: #2196F3;
        color: white;
        border: none;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 14px;
        border-radius: 4px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #1976D2;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .stTextInput>div>div>input {
        font-size: 1.2rem;
        padding: 10px;
        border-radius: 4px;
    }
    .command-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
    .suggestion-button {
        margin: 0.2rem;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        background-color: #e0e0e0;
        border: none;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .suggestion-button:hover {
        background-color: #2196F3;
        color: white;
    }
    .dataset-info {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
    .column-selector {
        margin: 1rem 0;
    }
    .result-box {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .settings-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    .command-category {
        font-weight: bold;
        color: #1976D2;
        margin-top: 1rem;
    }
    .command-item {
        padding: 0.5rem;
        margin: 0.2rem 0;
        background-color: #ffffff;
        border-radius: 4px;
        border-left: 4px solid #2196F3;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'llm_processor' not in st.session_state:
    st.session_state.llm_processor = LLMProcessor()
if 'result_text' not in st.session_state:
    st.session_state.result_text = ""
if 'result_fig' not in st.session_state:
    st.session_state.result_fig = None
if 'result_df' not in st.session_state:
    st.session_state.result_df = None
if 'dataset_analysis' not in st.session_state:
    st.session_state.dataset_analysis = None
if 'analysis_settings' not in st.session_state:
    st.session_state.analysis_settings = {
        "show_correlations": True,
        "show_distributions": True,
        "show_outliers": True,
        "max_categories": 10,
        "confidence_level": 0.95
    }
if 'ollama_status' not in st.session_state:
    st.session_state.ollama_status = "unknown"
if 'current_model' not in st.session_state:
    st.session_state.current_model = "phi:2.7b"

def get_download_link(df, filename, text):
    """Generate a download link for a DataFrame"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

def get_figure_download_link(fig, filename, text):
    """Generate a download link for a matplotlib figure"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    b64 = base64.b64encode(buf.getvalue()).decode()
    href = f'<a href="data:image/png;base64,{b64}" download="{filename}">{text}</a>'
    return href

def process_command(command):
    """Process text command and update session state"""
    if st.session_state.df is not None:
        try:
            # Simple command processing
            command_lower = command.lower()
            
            if "summary" in command_lower or "statistics" in command_lower:
                st.session_state.result_text = "Summary Statistics"
                st.session_state.result_df = st.session_state.df.describe()
                st.session_state.result_fig = None
                
            elif "missing" in command_lower:
                st.session_state.result_text = "Missing Values Analysis"
                missing_data = st.session_state.df.isnull().sum()
                st.session_state.result_df = pd.DataFrame({
                    'Column': missing_data.index,
                    'Missing_Count': missing_data.values,
                    'Missing_Percentage': (missing_data.values / len(st.session_state.df)) * 100
                })
                st.session_state.result_fig = None
                
            elif "correlation" in command_lower:
                st.session_state.result_text = "Correlation Matrix"
                numeric_df = st.session_state.df.select_dtypes(include=[np.number])
                if len(numeric_df.columns) > 1:
                    corr_matrix = numeric_df.corr()
                    st.session_state.result_df = corr_matrix
                    
                    # Create correlation heatmap
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
                    plt.title('Correlation Matrix')
                    st.session_state.result_fig = fig
                else:
                    st.session_state.result_text = "Not enough numeric columns for correlation analysis"
                    st.session_state.result_df = None
                    st.session_state.result_fig = None
                    
            elif "histogram" in command_lower:
                # Extract column name from command
                words = command_lower.split()
                if "of" in words:
                    col_idx = words.index("of") + 1
                    if col_idx < len(words):
                        col_name = words[col_idx]
                        if col_name in st.session_state.df.columns:
                            st.session_state.result_text = f"Histogram of {col_name}"
                            fig, ax = plt.subplots(figsize=(10, 6))
                            st.session_state.df[col_name].hist(ax=ax, bins=20)
                            plt.title(f'Histogram of {col_name}')
                            plt.xlabel(col_name)
                            plt.ylabel('Frequency')
                            st.session_state.result_fig = fig
                            st.session_state.result_df = None
                        else:
                            st.session_state.result_text = f"Column '{col_name}' not found"
                    else:
                        st.session_state.result_text = "Please specify a column name for histogram"
                else:
                    st.session_state.result_text = "Please specify a column name for histogram"
                    
            elif "boxplot" in command_lower:
                # Extract column name from command
                words = command_lower.split()
                if "for" in words:
                    col_idx = words.index("for") + 1
                    if col_idx < len(words):
                        col_name = words[col_idx]
                        if col_name in st.session_state.df.columns:
                            st.session_state.result_text = f"Boxplot of {col_name}"
                            fig, ax = plt.subplots(figsize=(10, 6))
                            st.session_state.df.boxplot(column=col_name, ax=ax)
                            plt.title(f'Boxplot of {col_name}')
                            st.session_state.result_fig = fig
                            st.session_state.result_df = None
                        else:
                            st.session_state.result_text = f"Column '{col_name}' not found"
                    else:
                        st.session_state.result_text = "Please specify a column name for boxplot"
                else:
                    st.session_state.result_text = "Please specify a column name for boxplot"
                    
            else:
                st.session_state.result_text = f"Command '{command}' not recognized. Try: summary statistics, missing values, correlation matrix, histogram of [column], boxplot for [column]"
                st.session_state.result_df = None
                st.session_state.result_fig = None
                
        except Exception as e:
            st.session_state.result_text = f"Error processing command: {str(e)}"
            st.session_state.result_df = None
            st.session_state.result_fig = None
    else:
        st.session_state.result_text = "Please upload a dataset first."
        st.session_state.result_fig = None
        st.session_state.result_df = None

def analyze_dataset():
    """Analyze the dataset and update session state"""
    if st.session_state.df is not None:
        try:
            # Basic dataset analysis
            analysis = {
                "data_quality": {
                    "total_rows": len(st.session_state.df),
                    "total_columns": len(st.session_state.df.columns),
                    "duplicate_rows": st.session_state.df.duplicated().sum(),
                    "missing_values": st.session_state.df.isnull().sum().to_dict()
                },
                "column_types": {},
                "insights": []
            }
            
            # Analyze each column
            for col in st.session_state.df.columns:
                col_type = str(st.session_state.df[col].dtype)
                unique_vals = st.session_state.df[col].nunique()
                null_count = st.session_state.df[col].isnull().sum()
                null_percent = (null_count / len(st.session_state.df)) * 100
                
                analysis["column_types"][col] = {
                    "type": col_type,
                    "unique_values": unique_vals,
                    "null_percentage": null_percent
                }
            
            # Generate insights using LLM
            if st.session_state.llm_processor:
                dataset_summary = f"""
                Dataset Summary:
                - Total Rows: {len(st.session_state.df)}
                - Total Columns: {len(st.session_state.df.columns)}
                - Column Types: {st.session_state.df.dtypes.to_dict()}
                - Missing Values: {st.session_state.df.isnull().sum().to_dict()}
                - Sample Data: {st.session_state.df.head().to_dict()}
                """
                analysis["insights"] = st.session_state.llm_processor.generate_insights(dataset_summary)
            
            st.session_state.dataset_analysis = analysis
            
        except Exception as e:
            st.error(f"Error analyzing dataset: {str(e)}")
            st.session_state.dataset_analysis = None

# Sidebar for commands and help
with st.sidebar:
    st.title("📚 Quick Actions")
    
    # Ollama Settings
    with st.expander("🤖 Ollama Settings", expanded=False):
        st.markdown("**Local LLM Configuration**")
        
        # Check Ollama status
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags", timeout=3)
            if response.status_code == 200:
                st.success("✅ Ollama is running")
                st.session_state.ollama_status = "running"
                
                # Get available models
                models = response.json().get("models", [])
                model_names = [model["name"] for model in models]
                
                if model_names:
                    st.markdown("**Available Models:**")
                    for model in model_names:
                        st.write(f"• {model}")
                    
                    # Model selection
                    selected_model = st.selectbox(
                        "Select Model",
                        model_names,
                        index=0 if not model_names else model_names.index(st.session_state.current_model) if st.session_state.current_model in model_names else 0
                    )
                    
                    if selected_model != st.session_state.current_model:
                        if st.button("Change Model"):
                            try:
                                st.session_state.llm_processor.change_model(selected_model)
                                st.session_state.current_model = selected_model
                                st.success(f"✅ Switched to {selected_model}")
                            except Exception as e:
                                st.error(f"❌ Error changing model: {str(e)}")
                else:
                    st.warning("⚠️ No models found. Run setup_ollama.py to install models.")
            else:
                st.error("❌ Ollama is not responding")
                st.session_state.ollama_status = "error"
        except requests.exceptions.RequestException:
            st.error("❌ Ollama is not running")
            st.session_state.ollama_status = "not_running"
            st.markdown("""
            **To set up Ollama:**
            1. Install Ollama: https://ollama.ai/download
            2. Run: `python setup_ollama.py`
            3. Start Ollama: `ollama serve`
            """)
        
        # Model recommendations
        st.markdown("**💡 Model Recommendations:**")
        st.markdown("""
        - **phi:2.7b**: Very lightweight (~1.7GB), good for low-end computers
        - **mistral:7b**: Fast and efficient (~4GB)
        - **llama2:7b**: Balanced performance (~4GB)
        - **codellama:7b**: Good for code-related tasks (~4GB)
        """)

    # Quick Action Buttons
    st.markdown("### 🚀 Quick Actions")
    
    if st.button("📊 Show Summary Statistics", help="Display basic statistics for all columns"):
        process_command("show summary statistics")
        
    if st.button("🔍 Show Missing Values", help="Analyze missing data in the dataset"):
        process_command("show missing values")
        
    if st.button("📈 Show Correlation Matrix", help="Display the correlation matrix for numeric columns"):
        process_command("show correlation matrix")
        
    if st.button("📋 Show Data Types", help="Display the data type of each column"):
        if st.session_state.df is not None:
            st.session_state.result_text = "Data Types"
            st.session_state.result_df = pd.DataFrame({
                'Column': st.session_state.df.columns,
                'Data Type': [str(dtype) for dtype in st.session_state.df.dtypes]
            })
            st.session_state.result_fig = None
        else:
            st.session_state.result_text = "Please upload a dataset first."
            
    if st.button("👥 Show Top 5 Rows", help="Display the first 5 rows of the dataset"):
        if st.session_state.df is not None:
            st.session_state.result_text = "Top 5 Rows"
            st.session_state.result_df = st.session_state.df.head()
            st.session_state.result_fig = None
        else:
            st.session_state.result_text = "Please upload a dataset first."

    # Column-specific actions
    if st.session_state.df is not None:
        st.markdown("### 📊 Column Analysis")
        
        selected_column = st.selectbox(
            "Select a column for analysis",
            options=st.session_state.df.columns.tolist(),
            help="Choose a column to analyze"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button(f"📊 Histogram of {selected_column}"):
                process_command(f"histogram of {selected_column}")
                
        with col2:
            if st.button(f"📦 Boxplot of {selected_column}"):
                process_command(f"boxplot for {selected_column}")

# Main content
st.title("📊 Veda - Data Analysis Assistant")
st.markdown("""
    Advanced data analysis and visualization platform powered by local AI.
    Upload your dataset and explore its insights using text commands or quick action buttons.
""")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])
if uploaded_file is not None:
    try:
        # Read CSV with improved error handling
        df = pd.read_csv(uploaded_file, encoding='utf-8', on_bad_lines='warn')
        
        # Basic data cleaning
        df = df.replace([np.inf, -np.inf], np.nan)  # Replace inf values with NaN
        
        # Store in session state
        st.session_state.df = df
        st.success("File uploaded successfully!")
        
        # Analyze dataset
        analyze_dataset()
        
        # Display dataset information
        st.markdown("### 📊 Dataset Information")
        with st.expander("View Dataset Details"):
            if st.session_state.dataset_analysis:
                st.write("**Dataset Overview:**")
                st.write(f"Total Rows: {st.session_state.dataset_analysis['data_quality']['total_rows']}")
                st.write(f"Total Columns: {st.session_state.dataset_analysis['data_quality']['total_columns']}")
                st.write(f"Duplicate Rows: {st.session_state.dataset_analysis['data_quality']['duplicate_rows']}")
                
                st.write("\n**Column Information:**")
                for col, info in st.session_state.dataset_analysis['column_types'].items():
                    st.write(f"\n{col}:")
                    st.write(f"- Type: {info['type']}")
                    st.write(f"- Unique Values: {info['unique_values']}")
                    st.write(f"- Null Percentage: {info['null_percentage']:.2f}%")
                
                if st.session_state.dataset_analysis.get('insights'):
                    st.write("\n**AI Insights:**")
                    st.write(st.session_state.dataset_analysis['insights'])
        
        st.write("**Dataset Preview:**")
        st.dataframe(st.session_state.df.head())

    except Exception as e:
        st.error(f"Error reading file: {str(e)}")

# Text Input Section
st.markdown("### ⌨️ Text Commands")
text_command = st.text_input("Type your command here (e.g., 'show summary statistics', 'histogram of age')")

# Process text command if entered
if text_command:
    process_command(text_command)

# Results Section
st.markdown("### 📊 Results")

# Show result
if st.session_state.result_text:
    st.markdown("**Analysis Result:**")
    st.write(st.session_state.result_text)

# Display results in proper format
if st.session_state.result_df is not None:
    st.markdown("### 📋 Data Table")
    st.dataframe(st.session_state.result_df)
    
    # Add download button for the table
    st.markdown(get_download_link(
        st.session_state.result_df,
        "analysis_results.csv",
        "Download Table as CSV"
    ), unsafe_allow_html=True)

if st.session_state.result_fig is not None:
    st.markdown("### 📈 Visualization")
    st.pyplot(st.session_state.result_fig)
    
    # Add download button for the figure
    st.markdown(get_figure_download_link(
        st.session_state.result_fig,
        "visualization.png",
        "Download Visualization as PNG"
    ), unsafe_allow_html=True)

# Help section
with st.expander("💡 Available Commands"):
    st.markdown("""
    Here are some example commands you can use:
    
    **Basic Analysis:**
    - "Show summary statistics"
    - "Show missing values"
    - "Show data types"
    
    **Visualizations:**
    - "Histogram of [column name]"
    - "Boxplot for [column name]"
    - "Show correlation matrix"
    
    **Quick Actions:**
    - Use the sidebar buttons for instant analysis
    - Select columns from the dropdown for specific analysis
    
    **Tips:**
    - Upload a CSV file to get started
    - Use the sidebar for quick actions
    - Type commands in the text input for custom analysis
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; font-size: 1.1em;'>
  <a href="https://github.com/Kirissh" target="_blank" style="margin-right: 20px; text-decoration: none;">
    <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/github/github-original.svg" height="20" style="vertical-align: middle; margin-right: 5px;"/> @Kirissh
  </a>
  <a href="https://www.linkedin.com/in/kirissh/" target="_blank" style="text-decoration: none;">
    <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/linkedin/linkedin-original.svg" height="20" style="vertical-align: middle; margin-right: 5px;"/> @kirissh
  </a>
</div>
""", unsafe_allow_html=True) 