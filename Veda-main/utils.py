import json
import queue
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import re
import pyttsx3
from typing import Optional, Tuple, List, Dict, Any
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import io
from vosk import Model, KaldiRecognizer
from Levenshtein import ratio
import os
import threading
import urllib.request
import zipfile
from llm_processor import LLMProcessor
import whisper
import wave
import tempfile
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

VOSK_MODEL_NAME = "vosk-model-small-en-us-0.15"
VOSK_MODEL_URL = f"https://alphacephei.com/vosk/models/{VOSK_MODEL_NAME}.zip"

class VoiceProcessor:
    def __init__(self):
        self.sample_rate = 16000
        self.channels = 1
        self.recording = False
        self.audio_data = []
        self.is_recording = False
        try:
            self.model = whisper.load_model("base")
            logger.info("Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading Whisper model: {str(e)}")
            self.model = None
        self.engine = pyttsx3.init()
        self.audio_queue = queue.Queue()
        self.continuous_thread = None
        self.continuous_result = ""
        self.command_keywords = {
            "summary": ["summary", "statistics", "stats", "describe"],
            "histogram": ["histogram", "hist", "distribution"],
            "boxplot": ["boxplot", "box", "box plot"],
            "correlation": ["correlation", "correlate", "corr"],
            "missing": ["missing", "null", "na", "nan"],
            "regression": ["regression", "linear", "predict"],
            "top": ["top", "first", "head"],
            "scatter": ["scatter", "scatterplot", "plot"]
        }
        
    def start_recording(self):
        """Start recording audio"""
        try:
            self.recording = True
            self.audio_data = []
            self.is_recording = True
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                callback=self._audio_callback
            )
            self.stream.start()
            logger.info("Recording started")
        except Exception as e:
            logger.error(f"Error starting recording: {str(e)}")
            self.is_recording = False

    def _audio_callback(self, indata, frames, time, status):
        """Callback function for audio recording"""
        if status:
            logger.warning(f"Audio callback status: {status}")
        if self.is_recording:
            self.audio_data.append(indata.copy())

    def stop_recording(self) -> Optional[str]:
        """Stop recording and transcribe audio"""
        try:
            if not self.is_recording:
                return None

            self.is_recording = False
            if hasattr(self, 'stream'):
                self.stream.stop()
                self.stream.close()

            if not self.audio_data:
                logger.warning("No audio recorded")
                return None

            # Combine audio data
            audio_data = np.concatenate(self.audio_data, axis=0)
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_filename = temp_file.name
                with wave.open(temp_filename, 'wb') as wf:
                    wf.setnchannels(self.channels)
                    wf.setsampwidth(2)  # 2 bytes for int16
                    wf.setframerate(self.sample_rate)
                    wf.writeframes((audio_data * 32767).astype(np.int16).tobytes())

            # Transcribe using Whisper
            if self.model:
                try:
                    result = self.model.transcribe(temp_filename)
                    os.unlink(temp_filename)  # Clean up temporary file
                    return result["text"].strip()
                except Exception as e:
                    logger.error(f"Error transcribing audio: {str(e)}")
                    return None
            else:
                logger.error("Whisper model not loaded")
                return None

        except Exception as e:
            logger.error(f"Error stopping recording: {str(e)}")
            return None

    def start_continuous_listening(self, callback_fn):
        """Continuously listen for voice input and call callback_fn with recognized text."""
        self.audio_data = []
        self.continuous_result = ""
        def listen_loop():
            with sd.InputStream(channels=1, samplerate=self.sample_rate) as stream:
                rec = KaldiRecognizer(self.model, self.sample_rate)
                rec.SetWords(True)
                while self.is_recording:
                    data, _ = stream.read(4000)
                    if rec.AcceptWaveform(data):
                        result = json.loads(rec.Result())
                        text = result.get("text", "").strip()
                        if text:
                            self.continuous_result = text
                            callback_fn(text)
        self.continuous_thread = threading.Thread(target=listen_loop, daemon=True)
        self.continuous_thread.start()

    def stop_continuous_listening(self):
        self.is_recording = False
        if self.continuous_thread:
            self.continuous_thread.join(timeout=1)
            self.continuous_thread = None

    def speak(self, text: str):
        """Convert text to speech"""
        self.engine.say(text)
        self.engine.runAndWait()
    
    def find_best_command_match(self, text: str) -> Tuple[str, float]:
        """Find the best matching command using fuzzy matching"""
        text = text.lower()
        best_match = None
        best_score = 0.0
        
        for command_type, keywords in self.command_keywords.items():
            for keyword in keywords:
                score = ratio(text, keyword)
                if score > best_score:
                    best_score = score
                    best_match = command_type
        
        return best_match, best_score

class CommandParser:
    def __init__(self):
        self.llm_processor = LLMProcessor()
        self.command_patterns = {
            "summary": ["summary", "statistics", "stats", "describe"],
            "histogram": ["histogram", "hist", "distribution"],
            "boxplot": ["boxplot", "box", "box plot"],
            "correlation": ["correlation", "correlate", "corr"],
            "missing": ["missing", "null", "na", "nan"],
            "scatter": ["scatter", "scatterplot", "plot"],
            "top": ["top", "first", "head"]
        }
        
        # Command templates for better matching
        self.command_templates = {
            "summary": ["show summary statistics", "show stats", "describe data"],
            "histogram": ["plot histogram of {column}", "show distribution of {column}"],
            "boxplot": ["create boxplot for {column}", "show boxplot of {column}"],
            "correlation": ["display correlation matrix", "show correlations"],
            "missing": ["show missing value heatmap", "display null values"],
            "regression": ["run linear regression with {target} as target and {features} as features"],
            "top": ["show top {n} rows", "display first {n} rows"],
            "scatter": ["scatterplot of {x} vs {y}", "plot {x} against {y}"]
        }
    
    def parse_command(self, command: str, df: pd.DataFrame) -> Tuple[Optional[str], Optional[plt.Figure], Optional[pd.DataFrame]]:
        """Parse voice command and return appropriate analysis"""
        command = command.lower()
        
        # First, try to process with LLM
        llm_result, llm_status = self.llm_processor.process_command(command, df.columns.tolist())
        
        if llm_result:
            # Validate the LLM's command
            is_valid, validation_msg = self.llm_processor.validate_command(llm_result, df.columns.tolist())
            
            if is_valid:
                # Execute the command based on LLM's interpretation
                command_type = llm_result["command_type"]
                params = llm_result["parameters"]
                
                if command_type == "histogram":
                    return self._handle_histogram(df, params["column"])
                elif command_type == "boxplot":
                    return self._handle_boxplot(df, params["column"])
                elif command_type == "correlation":
                    return self._handle_correlation(df)
                elif command_type == "missing":
                    return self._handle_missing(df)
                elif command_type == "regression":
                    return self._handle_regression(df, params["target"], params["features"])
                elif command_type == "scatter":
                    return self._handle_scatter(df, params["x_column"], params["y_column"])
                elif command_type == "summary":
                    return self._handle_summary(df)
            
            return f"LLM interpretation: {llm_result.get('explanation', '')}\nValidation: {validation_msg}", None, None
        
        # Fallback to pattern matching if LLM fails
        for pattern, func in self.command_patterns.items():
            match = re.match(pattern, command)
            if match:
                return func(df, *match.groups())
        
        return "Command not recognized. Please try again with a clearer command.", None, None
    
    def _handle_summary(self, df: pd.DataFrame) -> Tuple[str, Optional[plt.Figure], Optional[pd.DataFrame]]:
        """Handle summary statistics command"""
        try:
            summary = df.describe()
            return "Summary Statistics:", None, summary
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return f"Error generating summary: {str(e)}", None, None
    
    def _handle_histogram(self, df: pd.DataFrame, column: str) -> Tuple[str, Optional[plt.Figure], Optional[pd.DataFrame]]:
        """Handle histogram command"""
        try:
            if column not in df.columns:
                return f"Column '{column}' not found in dataset.", None, None
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(data=df, x=column, ax=ax)
            ax.set_title(f"Histogram of {column}")
            plt.tight_layout()
            
            # Create frequency table
            freq_table = df[column].value_counts().reset_index()
            freq_table.columns = [column, 'Frequency']
            freq_table = freq_table.sort_values(by=column)
            
            return f"Histogram of {column}", fig, freq_table
        except Exception as e:
            logger.error(f"Error generating histogram: {str(e)}")
            return f"Error generating histogram: {str(e)}", None, None
    
    def _handle_boxplot(self, df: pd.DataFrame, column: str) -> Tuple[str, Optional[plt.Figure], Optional[pd.DataFrame]]:
        """Handle boxplot command"""
        try:
            if column not in df.columns:
                return f"Column '{column}' not found in dataset.", None, None
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(data=df, y=column, ax=ax)
            ax.set_title(f"Boxplot of {column}")
            plt.tight_layout()
            
            # Create summary statistics for the boxplot
            stats = df[column].describe()
            stats_df = pd.DataFrame(stats).reset_index()
            stats_df.columns = ['Statistic', 'Value']
            
            return f"Boxplot of {column}", fig, stats_df
        except Exception as e:
            logger.error(f"Error generating boxplot: {str(e)}")
            return f"Error generating boxplot: {str(e)}", None, None
    
    def _handle_correlation(self, df: pd.DataFrame) -> Tuple[str, Optional[plt.Figure], Optional[pd.DataFrame]]:
        """Handle correlation command"""
        try:
            numeric_df = df.select_dtypes(include=[np.number])
            if numeric_df.empty:
                return "No numeric columns found for correlation analysis.", None, None
            
            corr_matrix = numeric_df.corr()
            
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
            ax.set_title("Correlation Matrix")
            plt.tight_layout()
            
            return "Correlation Matrix", fig, corr_matrix
        except Exception as e:
            logger.error(f"Error generating correlation matrix: {str(e)}")
            return f"Error generating correlation matrix: {str(e)}", None, None
    
    def _handle_missing(self, df: pd.DataFrame) -> Tuple[str, Optional[plt.Figure], Optional[pd.DataFrame]]:
        """Handle missing values command"""
        try:
            missing_data = df.isnull().sum()
            missing_percent = (missing_data / len(df)) * 100
            
            missing_df = pd.DataFrame({
                'Missing Values': missing_data,
                'Percentage': missing_percent
            }).sort_values('Missing Values', ascending=False)
            
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis', ax=ax)
            ax.set_title("Missing Value Heatmap")
            plt.tight_layout()
            
            return "Missing Value Analysis", fig, missing_df
        except Exception as e:
            logger.error(f"Error analyzing missing values: {str(e)}")
            return f"Error analyzing missing values: {str(e)}", None, None
    
    def _handle_regression(self, df: pd.DataFrame, target: str, features: str) -> Tuple[str, Optional[plt.Figure], Optional[pd.DataFrame]]:
        """Handle regression command"""
        try:
            if target not in df.columns:
                return f"Target column '{target}' not found in dataset.", None, None
            
            feature_list = [f.strip() for f in features.split(',')]
            missing_features = [f for f in feature_list if f not in df.columns]
            if missing_features:
                return f"Feature columns {missing_features} not found in dataset.", None, None
            
            X = df[feature_list]
            y = df[target]
            
            model = LinearRegression()
            model.fit(X, y)
            
            # Create results DataFrame
            results_df = pd.DataFrame({
                'Feature': feature_list,
                'Coefficient': model.coef_,
                'Importance': np.abs(model.coef_)
            }).sort_values('Importance', ascending=False)
            
            # Plot feature importance
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=results_df, x='Feature', y='Importance', ax=ax)
            ax.set_title("Feature Importance")
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            return f"Linear Regression Results for {target}", fig, results_df
        except Exception as e:
            logger.error(f"Error performing regression: {str(e)}")
            return f"Error performing regression: {str(e)}", None, None
    
    def _handle_scatter(self, df: pd.DataFrame, x_column: str, y_column: str) -> Tuple[str, Optional[plt.Figure], Optional[pd.DataFrame]]:
        """Handle scatter plot command"""
        try:
            if x_column not in df.columns or y_column not in df.columns:
                return f"Columns '{x_column}' or '{y_column}' not found in dataset.", None, None
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(data=df, x=x_column, y=y_column, ax=ax)
            ax.set_title(f"Scatter Plot: {x_column} vs {y_column}")
            plt.tight_layout()
            
            # Create summary statistics for the scatter plot
            stats_df = pd.DataFrame({
                'Statistic': ['Correlation', 'X Mean', 'Y Mean', 'X Std', 'Y Std'],
                'Value': [
                    df[x_column].corr(df[y_column]),
                    df[x_column].mean(),
                    df[y_column].mean(),
                    df[x_column].std(),
                    df[y_column].std()
                ]
            })
            
            return f"Scatter Plot: {x_column} vs {y_column}", fig, stats_df
        except Exception as e:
            logger.error(f"Error generating scatter plot: {str(e)}")
            return f"Error generating scatter plot: {str(e)}", None, None
    
    def _handle_top(self, df: pd.DataFrame, command: str) -> Tuple[str, Optional[plt.Figure], Optional[pd.DataFrame]]:
        """Handle top rows command"""
        try:
            # Extract number from command if present
            n = 5  # default
            for word in command.split():
                if word.isdigit():
                    n = int(word)
                    break
            
            return f"Top {n} rows:", None, df.head(n)
        except Exception as e:
            logger.error(f"Error showing top rows: {str(e)}")
            return f"Error showing top rows: {str(e)}", None, None
    
    def parse_command_with_llm_feedback(self, command: str, df: pd.DataFrame) -> Tuple[str, Optional[plt.Figure], Optional[pd.DataFrame], str]:
        """Parse command and return results with LLM feedback"""
        try:
            command = command.lower()
            result = ""
            fig = None
            result_df = None
            llm_msg = ""

            # Basic command patterns
            if any(pattern in command for pattern in self.command_patterns["summary"]):
                result_df = df.describe()
                result = "Summary statistics generated successfully."
                llm_msg = "Showing summary statistics of the dataset."

            elif any(pattern in command for pattern in self.command_patterns["top"]):
                n = 5  # default
                if "top" in command:
                    try:
                        n = int(command.split("top")[1].split()[0])
                    except:
                        pass
                result_df = df.head(n)
                result = f"Showing top {n} rows of the dataset."
                llm_msg = f"Displaying the first {n} rows of the dataset."

            elif any(pattern in command for pattern in self.command_patterns["missing"]):
                missing = df.isnull().sum()
                result_df = pd.DataFrame({
                    'Column': missing.index,
                    'Missing Values': missing.values,
                    'Percentage': (missing.values / len(df) * 100).round(2)
                })
                result = "Missing value analysis completed."
                llm_msg = "Analyzing missing values in the dataset."

            elif any(pattern in command for pattern in self.command_patterns["histogram"]):
                # Extract column name from command
                for col in df.columns:
                    if col.lower() in command:
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.histplot(data=df, x=col, ax=ax)
                        ax.set_title(f'Distribution of {col}')
                        result = f"Histogram created for {col}."
                        llm_msg = f"Visualizing the distribution of {col}."
                        break

            elif any(pattern in command for pattern in self.command_patterns["boxplot"]):
                # Extract column name from command
                for col in df.columns:
                    if col.lower() in command:
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.boxplot(data=df, y=col, ax=ax)
                        ax.set_title(f'Boxplot of {col}')
                        result = f"Boxplot created for {col}."
                        llm_msg = f"Visualizing the distribution and outliers of {col}."
                        break

            elif any(pattern in command for pattern in self.command_patterns["correlation"]):
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 1:
                    corr = df[numeric_cols].corr()
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
                    ax.set_title('Correlation Matrix')
                    result = "Correlation matrix generated."
                    llm_msg = "Analyzing correlations between numeric columns."
                else:
                    result = "Not enough numeric columns for correlation analysis."
                    llm_msg = "Correlation analysis requires at least two numeric columns."

            elif any(pattern in command for pattern in self.command_patterns["scatter"]):
                # Extract column names from command
                cols = []
                for col in df.columns:
                    if col.lower() in command:
                        cols.append(col)
                        if len(cols) == 2:
                            break
                
                if len(cols) == 2:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.scatterplot(data=df, x=cols[0], y=cols[1], ax=ax)
                    ax.set_title(f'Scatter Plot: {cols[0]} vs {cols[1]}')
                    result = f"Scatter plot created for {cols[0]} vs {cols[1]}."
                    llm_msg = f"Visualizing the relationship between {cols[0]} and {cols[1]}."
                else:
                    result = "Please specify two columns for scatter plot."
                    llm_msg = "Scatter plot requires exactly two columns."

            else:
                result = "Command not recognized. Please try again."
                llm_msg = "I couldn't understand the command. Please use one of the suggested commands."

            return result, fig, result_df, llm_msg

        except Exception as e:
            logger.error(f"Error parsing command: {str(e)}")
            return f"Error: {str(e)}", None, None, "An error occurred while processing the command." 