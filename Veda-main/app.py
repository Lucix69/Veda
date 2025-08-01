import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from utils import VoiceProcessor, CommandParser
from llm_processor import LLMProcessor
import io
import base64
import time
import numpy as np

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
    .recording {
        animation: pulse 1.5s infinite;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
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
if 'voice_processor' not in st.session_state:
    st.session_state.voice_processor = VoiceProcessor()
if 'command_parser' not in st.session_state:
    st.session_state.command_parser = CommandParser()
if 'llm_processor' not in st.session_state:
    st.session_state.llm_processor = LLMProcessor()
if 'last_recognized' not in st.session_state:
    st.session_state.last_recognized = ""
if 'llm_interpretation' not in st.session_state:
    st.session_state.llm_interpretation = ""
if 'result_text' not in st.session_state:
    st.session_state.result_text = ""
if 'result_fig' not in st.session_state:
    st.session_state.result_fig = None
if 'result_df' not in st.session_state:
    st.session_state.result_df = None
if 'is_recording' not in st.session_state:
    st.session_state.is_recording = False
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
    st.session_state.current_model = "llama2"

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

def process_voice_command(command):
    """Process voice command and update session state"""
    st.session_state.last_recognized = command
    if st.session_state.df is not None:
        result, fig, df, llm_msg = st.session_state.command_parser.parse_command_with_llm_feedback(
            command, st.session_state.df)
        st.session_state.llm_interpretation = llm_msg
        st.session_state.result_text = result
        st.session_state.result_fig = fig
        st.session_state.result_df = df
    else:
        st.session_state.result_text = "Please upload a dataset first."
        st.session_state.result_fig = None
        st.session_state.result_df = None

def toggle_recording():
    """Toggle recording state"""
    if not st.session_state.is_recording:
        st.session_state.voice_processor.start_recording()
        st.session_state.is_recording = True
    else:
        recognized = st.session_state.voice_processor.stop_recording()
        st.session_state.is_recording = False
        if recognized:
            process_voice_command(recognized)

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
    st.title("📚 Available Commands")
    
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

    # Helper to store pending command and columns
    if 'pending_command' not in st.session_state:
        st.session_state.pending_command = None
    if 'pending_command_type' not in st.session_state:
        st.session_state.pending_command_type = None
    if 'pending_column_count' not in st.session_state:
        st.session_state.pending_column_count = 1

    def sidebar_command_button(label, command_template, help_text, needs_col=False, needs_two_cols=False, needs_value=False, value_label=None):
        cmd = command_template
        if needs_value and value_label:
            cmd = cmd.replace("[mean/median/mode]", value_label)
        if st.button(label, key=f"sidebar_btn_{label}_{cmd}", help=help_text):
            if needs_col:
                st.session_state.pending_command = cmd
                st.session_state.pending_command_type = 'single_col'
                st.session_state.pending_column_count = 1
            elif needs_two_cols:
                st.session_state.pending_command = cmd
                st.session_state.pending_command_type = 'two_cols'
                st.session_state.pending_column_count = 2
            else:
                process_voice_command(cmd)

    # Data Preprocessing
    with st.expander("Data Preprocessing", expanded=False):
        sidebar_command_button("Handle Missing Values", "handle missing values", "Show missing value analysis for all columns.")
        sidebar_command_button("Remove Duplicates", "remove duplicates", "Remove duplicate rows from the dataset.")
        sidebar_command_button("Drop Rows with Missing Values", "drop rows with missing values", "Drop all rows that contain any missing values.")
        sidebar_command_button("Normalize [column]", "normalize [column]", "Normalize the selected numeric column.", needs_col=True)
        sidebar_command_button("Standardize [column]", "standardize [column]", "Standardize the selected numeric column.", needs_col=True)
        sidebar_command_button("Encode [column]", "encode [column]", "Encode the selected categorical column.", needs_col=True)
        for fill_type in ["mean", "median", "mode"]:
            sidebar_command_button(f"Fill Missing with {fill_type.title()}", f"fill missing with [mean/median/mode]", f"Fill missing values using the {fill_type} of each column.", needs_value=True, value_label=fill_type)
        sidebar_command_button("Filter Rows", "filter rows", "Filter rows based on a condition (you will be prompted for details).")
        sidebar_command_button("Sort Rows", "sort rows", "Sort rows by a selected column.", needs_col=True)
        sidebar_command_button("Sample Rows", "sample rows", "Randomly sample a subset of rows from the dataset.")

    # Basic Analysis
    with st.expander("Basic Analysis", expanded=False):
        sidebar_command_button("Show Summary Statistics", "show summary statistics", "Display basic statistics for all columns.")
        sidebar_command_button("Show Data Types", "show data types", "Display the data type of each column.")
        sidebar_command_button("Show Top [n] Rows", "show top 5 rows", "Display the first n rows of the dataset.")
        sidebar_command_button("Show Unique Values in [column]", "show unique values in [column]", "Display unique values in the selected column.", needs_col=True)
        sidebar_command_button("Show Missing Values", "show missing values", "Analyze missing data in the dataset.")

    # Visualizations
    with st.expander("Visualizations", expanded=False):
        sidebar_command_button("Plot Histogram of [column]", "plot histogram of [column]", "Create a histogram for the selected column.", needs_col=True)
        sidebar_command_button("Create Boxplot for [column]", "create boxplot for [column]", "Create a boxplot for the selected column.", needs_col=True)
        sidebar_command_button("Show Correlation Matrix", "show correlation matrix", "Display the correlation matrix for numeric columns.")
        sidebar_command_button("Create Scatter Plot of [x] vs [y]", "create scatter plot of [x] vs [y]", "Create a scatter plot for two selected columns.", needs_two_cols=True)
        sidebar_command_button("Create Pairplot", "create pairplot", "Create a pairplot for all numeric columns.")
        sidebar_command_button("Plot Heatmap", "plot heatmap", "Create a heatmap of the correlation matrix.")
        sidebar_command_button("Plot Time Series of [column]", "plot time series of [column]", "Plot a time series for the selected column.", needs_col=True)

    # Advanced Analysis
    with st.expander("Advanced Analysis", expanded=False):
        sidebar_command_button("Run Linear Regression with [target]", "run linear regression with [column]", "Perform linear regression using the selected column as the target.", needs_col=True)
        sidebar_command_button("Show Feature Importance", "show feature importance", "Display feature importance from a regression model.")
        sidebar_command_button("Detect Outliers in [column]", "detect outliers in [column]", "Identify outliers in the selected column.", needs_col=True)
        sidebar_command_button("Show Distribution Statistics", "show distribution statistics", "Display distribution statistics for all columns.")
        sidebar_command_button("Analyze Relationships", "analyze relationships", "Analyze relationships between columns.")

# Main content
st.title("📊 Veda")
st.markdown("""
    Advanced data analysis and visualization platform powered by AI.
    Upload your dataset and explore its insights.
""")

# File uploader with improved CSV handling
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
        
        # Dataset Settings
        st.markdown("### ⚙️ Dataset Settings")
        with st.expander("Configure Dataset Settings"):
            col1, col2 = st.columns(2)
            with col1:
                # Data preprocessing options
                st.markdown("**Data Preprocessing**")
                handle_missing = st.selectbox(
                    "Handle Missing Values",
                    ["None", "Drop", "Fill with Mean", "Fill with Median", "Fill with Mode", "Fill with Forward", "Fill with Backward"]
                )
                if handle_missing != "None":
                    if handle_missing == "Drop":
                        st.session_state.df = st.session_state.df.dropna()
                    elif handle_missing == "Fill with Forward":
                        st.session_state.df = st.session_state.df.fillna(method='ffill')
                    elif handle_missing == "Fill with Backward":
                        st.session_state.df = st.session_state.df.fillna(method='bfill')
                    else:
                        st.session_state.df = st.session_state.df.fillna(
                            st.session_state.df.mean() if handle_missing == "Fill with Mean"
                            else st.session_state.df.median() if handle_missing == "Fill with Median"
                            else st.session_state.df.mode().iloc[0]
                        )
                
                # Data type conversion
                st.markdown("**Data Type Conversion**")
                for col in st.session_state.df.columns:
                    current_type = str(st.session_state.df[col].dtype)
                    new_type = st.selectbox(
                        f"Convert {col} ({current_type}) to:",
                        ["Keep Current", "Numeric", "String", "Category", "Datetime", "Boolean"],
                        key=f"type_{col}"
                    )
                    if new_type != "Keep Current":
                        try:
                            if new_type == "Numeric":
                                st.session_state.df[col] = pd.to_numeric(st.session_state.df[col], errors='coerce')
                            elif new_type == "String":
                                st.session_state.df[col] = st.session_state.df[col].astype(str)
                            elif new_type == "Category":
                                st.session_state.df[col] = st.session_state.df[col].astype('category')
                            elif new_type == "Datetime":
                                st.session_state.df[col] = pd.to_datetime(st.session_state.df[col], errors='coerce')
                            elif new_type == "Boolean":
                                st.session_state.df[col] = st.session_state.df[col].astype(bool)
                        except Exception as e:
                            st.error(f"Error converting {col}: {str(e)}")
            
            with col2:
                # Analysis settings
                st.markdown("**Analysis Settings**")
                st.session_state.analysis_settings = {
                    "show_correlations": st.checkbox("Show Correlations", value=True),
                    "show_distributions": st.checkbox("Show Distributions", value=True),
                    "show_outliers": st.checkbox("Show Outliers", value=True),
                    "max_categories": st.slider("Max Categories to Show", 5, 50, 10),
                    "confidence_level": st.slider("Statistical Confidence Level", 0.8, 0.99, 0.95, 0.01),
                    "normalize_data": st.checkbox("Normalize Numeric Data", value=False),
                    "remove_outliers": st.checkbox("Remove Outliers", value=False)
                }
        
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
        
        # Column selector with enhanced information
        st.markdown("### 🔍 Column Selection")
        col1, col2 = st.columns(2)
        with col1:
            selected_column = st.selectbox(
                "Select a column for analysis",
                options=st.session_state.df.columns.tolist(),
                help="Choose a column to analyze"
            )
        with col2:
            if selected_column:
                col_type = str(st.session_state.df[selected_column].dtype)
                st.write(f"**Column Type:** {col_type}")
                if "int" in col_type or "float" in col_type:
                    st.write(f"**Range:** {st.session_state.df[selected_column].min():.2f} to {st.session_state.df[selected_column].max():.2f}")
                    st.write(f"**Mean:** {st.session_state.df[selected_column].mean():.2f}")
                    st.write(f"**Median:** {st.session_state.df[selected_column].median():.2f}")
                    st.write(f"**Std Dev:** {st.session_state.df[selected_column].std():.2f}")
                else:
                    st.write(f"**Unique Values:** {st.session_state.df[selected_column].nunique()}")
                    if st.session_state.df[selected_column].nunique() <= st.session_state.analysis_settings['max_categories']:
                        st.write("**Value Counts:**")
                        st.write(st.session_state.df[selected_column].value_counts())

        # ---
        # Handle pending command (column selection after button click)
        if st.session_state.pending_command:
            col_count = st.session_state.pending_column_count
            st.markdown("### Select Required Column(s) for Command")
            # Determine if this command needs multi-select (by command type or keywords)
            multi_col_commands = [
                'pairplot', 'plot heatmap', 'run linear regression', 'show feature importance', 'analyze relationships', 'sort rows', 'filter rows'
            ]
            cmd_lower = st.session_state.pending_command.lower()
            if any(cmd in cmd_lower for cmd in multi_col_commands):
                # Allow multi-select for these commands
                cols = st.multiselect("Select columns", st.session_state.df.columns, key="pending_multi_cols_select")
                if cols and st.button("Run Command", key="run_pending_cmd_multi"):
                    # For regression, assume first is target, rest are features
                    if 'run linear regression' in cmd_lower and len(cols) >= 2:
                        cmd = st.session_state.pending_command.replace("[column]", cols[0])
                        features = ','.join(cols[1:])
                        cmd += f" with features {features}"
                    elif 'pairplot' in cmd_lower or 'plot heatmap' in cmd_lower or 'analyze relationships' in cmd_lower:
                        cmd = st.session_state.pending_command + " for columns " + ','.join(cols)
                    elif 'sort rows' in cmd_lower:
                        cmd = st.session_state.pending_command + " by " + ','.join(cols)
                    elif 'filter rows' in cmd_lower:
                        cmd = st.session_state.pending_command + " on " + ','.join(cols)
                    else:
                        cmd = st.session_state.pending_command.replace("[column]", ','.join(cols))
                    process_voice_command(cmd)
                    st.session_state.pending_command = None
            elif col_count == 2:
                cols = st.multiselect("Select two columns", st.session_state.df.columns, max_selections=2, key="pending_cols_select")
                if len(cols) == 2 and st.button("Run Command", key="run_pending_cmd"):
                    cmd = st.session_state.pending_command.replace("[x]", cols[0]).replace("[y]", cols[1])
                    process_voice_command(cmd)
                    st.session_state.pending_command = None
            elif col_count == 1:
                col = st.selectbox("Select column", st.session_state.df.columns, key="pending_col_select")
                if st.button("Run Command", key="run_pending_cmd_single"):
                    cmd = st.session_state.pending_command.replace("[column]", col)
                    process_voice_command(cmd)
                    st.session_state.pending_command = None

    except Exception as e:
        st.error(f"Error reading file: {str(e)}")

# Voice Input Section
st.markdown("### 🎤 Voice Input")
col1, col2 = st.columns([1, 3])

with col1:
    if st.button("🎤 Start/Stop Recording", key="record_button", 
                 help="Click to start/stop voice recording"):
        toggle_recording()

with col2:
    if st.session_state.is_recording:
        st.markdown('<div class="recording">🎤 Recording...</div>', unsafe_allow_html=True)

# Text Input Section
st.markdown("### ⌨️ Text Input")
text_command = st.text_input("Type your command here (if voice recognition fails)")

# Process text command if entered
if text_command:
    process_voice_command(text_command)

# Results Section
st.markdown("### 📊 Results")

# Show recognized speech
if st.session_state.last_recognized:
    st.markdown("**Recognized Command:**")
    st.info(st.session_state.last_recognized)

# Show LLM interpretation
if st.session_state.llm_interpretation:
    st.markdown("**Command Interpretation:**")
    st.info(st.session_state.llm_interpretation)

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

# Footer
st.markdown("---")
# Remove old copyright, add new footer with GitHub and LinkedIn
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

# Help section
with st.expander("Available Commands"):
    st.markdown("""
    Here are some example commands you can use:
    
    **Basic Analysis:**
    - "Show summary statistics"
    - "Show top 5 rows"
    - "Show missing values"
    
    **Visualizations:**
    - "Plot histogram for [column name]"
    - "Create boxplot for [column name]"
    - "Display correlation matrix"
    - "Show missing value heatmap"
    - "Scatterplot of [x_column] vs [y_column]"
    
    You can either:
    1. Click the suggested command buttons
    2. Use voice commands
    3. Type your command in the text input field
    """) 