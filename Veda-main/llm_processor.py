import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import json
import warnings
import logging
import os
import requests
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OllamaLLMProcessor:
    def __init__(self, model_name: str = "phi:2.7b", base_url: str = "http://localhost:11434"):
        """
        Initialize the Ollama LLM processor
        
        Args:
            model_name: The model to use (llama2, llama2:7b, llama2:13b, codellama, etc.)
            base_url: Ollama server URL
        """
        self.model_name = model_name
        self.base_url = base_url
        self.model = None
        self._ensure_model_available()
    
    def _ensure_model_available(self):
        """Ensure the specified model is available, pull if necessary"""
        try:
            # Check if model is available
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                available_models = response.json().get("models", [])
                model_names = [model["name"] for model in available_models]
                
                if self.model_name not in model_names:
                    logger.info(f"Model {self.model_name} not found. Pulling from Ollama...")
                    self._pull_model()
                else:
                    logger.info(f"Model {self.model_name} is available")
            else:
                logger.warning("Could not check available models. Make sure Ollama is running.")
                
        except requests.exceptions.ConnectionError:
            logger.error("Could not connect to Ollama. Make sure Ollama is running on localhost:11434")
            logger.info("To install Ollama, visit: https://ollama.ai")
            logger.info("After installation, run: ollama pull phi:2.7b")
        except Exception as e:
            logger.error(f"Error checking model availability: {str(e)}")
    
    def _pull_model(self):
        """Pull the model from Ollama"""
        try:
            logger.info(f"Pulling {self.model_name}... This may take a few minutes.")
            response = requests.post(f"{self.base_url}/api/pull", 
                                  json={"name": self.model_name})
            if response.status_code == 200:
                logger.info(f"Successfully pulled {self.model_name}")
            else:
                logger.error(f"Failed to pull {self.model_name}")
        except Exception as e:
            logger.error(f"Error pulling model: {str(e)}")
    
    def _generate_response(self, prompt: str, max_tokens: int = 1000) -> str:
        """Generate response from Ollama"""
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": 0.7,
                        "top_p": 0.9
                    }
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "")
            else:
                logger.error(f"Error generating response: {response.status_code}")
                return ""
                
        except requests.exceptions.ConnectionError:
            logger.error("Could not connect to Ollama. Make sure it's running.")
            return ""
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return ""

    def generate_insights(self, dataset_summary: str) -> str:
        """Generate insights about the dataset using local LLM"""
        try:
            prompt = f"""
            Analyze this dataset summary and provide key insights:
            {dataset_summary}
            
            Focus on:
            1. Data quality issues
            2. Potential relationships between columns
            3. Interesting patterns or anomalies
            4. Suggestions for further analysis
            
            Keep the response concise and actionable. Format as a clear list of insights.
            """

            response = self._generate_response(prompt)
            return response if response else "Unable to generate insights. Please check if Ollama is running."

        except Exception as e:
            logger.error(f"Error generating insights: {str(e)}")
            return f"Error generating insights: {str(e)}"

    def analyze_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze the dataset and return insights"""
        try:
            # Basic dataset analysis
            analysis = {
                "data_quality": {
                    "total_rows": len(df),
                    "total_columns": len(df.columns),
                    "duplicate_rows": df.duplicated().sum(),
                    "missing_values": df.isnull().sum().to_dict()
                },
                "column_types": {},
                "insights": []
            }
            
            # Analyze each column
            for col in df.columns:
                col_type = str(df[col].dtype)
                unique_vals = df[col].nunique()
                null_count = df[col].isnull().sum()
                null_percent = (null_count / len(df)) * 100
                
                analysis["column_types"][col] = {
                    "type": col_type,
                    "unique_values": unique_vals,
                    "null_percentage": null_percent
                }
            
            # Generate insights using local LLM
            dataset_summary = f"""
            Dataset Summary:
            - Total Rows: {len(df)}
            - Total Columns: {len(df.columns)}
            - Column Types: {df.dtypes.to_dict()}
            - Missing Values: {df.isnull().sum().to_dict()}
            - Sample Data: {df.head().to_dict()}
            """
            analysis["insights"] = self.generate_insights(dataset_summary)
            
            return analysis

        except Exception as e:
            logger.error(f"Error analyzing dataset: {str(e)}")
            return {
                "error": str(e),
                "data_quality": {},
                "column_types": {},
                "insights": []
            }

    def get_suggestions(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Get AI-powered suggestions for analysis"""
        try:
            # Prepare dataset summary for local LLM
            dataset_summary = f"""
            Dataset Summary:
            - Total Rows: {len(df)}
            - Total Columns: {len(df.columns)}
            - Column Types: {df.dtypes.to_dict()}
            - Missing Values: {df.isnull().sum().to_dict()}
            - Sample Data: {df.head().to_dict()}
            """

            prompt = f"""
            Based on this dataset summary, suggest relevant analysis commands:
            {dataset_summary}
            
            Return the suggestions in this JSON format:
            {{
                "categories": [
                    {{
                        "category": "string",
                        "commands": [
                            {{
                                "text": "string",
                                "command": "string"
                            }}
                        ]
                    }}
                ]
            }}
            
            Focus on:
            1. Data quality and preprocessing
            2. Basic statistical analysis
            3. Visualizations
            4. Advanced analysis
            """

            response = self._generate_response(prompt)
            
            try:
                # Try to parse JSON response
                suggestions = json.loads(response)
                return suggestions.get("categories", [])
            except json.JSONDecodeError:
                # If JSON parsing fails, return default suggestions
                logger.warning("Could not parse LLM response as JSON, using default suggestions")
                return self._get_default_suggestions(df)

        except Exception as e:
            logger.error(f"Error getting suggestions: {str(e)}")
            return self._get_default_suggestions(df)

    def _get_default_suggestions(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Get default suggestions when LLM is not available"""
        return [
            {
                "category": "Data Preprocessing",
                "commands": [
                    {"text": "Show Missing Values", "command": "show missing values"},
                    {"text": "Remove Duplicates", "command": "remove duplicates"},
                    {"text": "Handle Missing Values", "command": "handle missing values"}
                ]
            },
            {
                "category": "Basic Analysis",
                "commands": [
                    {"text": "Show Summary Statistics", "command": "show summary statistics"},
                    {"text": "Show Top 5 Rows", "command": "show top 5 rows"},
                    {"text": "Show Data Types", "command": "show data types"}
                ]
            },
            {
                "category": "Visualizations",
                "commands": [
                    {"text": "Show Correlation Matrix", "command": "show correlation matrix"},
                    {"text": "Plot Histogram", "command": f"plot histogram of {df.columns[0] if len(df.columns) > 0 else 'column'}"},
                    {"text": "Create Boxplot", "command": f"create boxplot for {df.columns[0] if len(df.columns) > 0 else 'column'}"}
                ]
            },
            {
                "category": "Advanced Analysis",
                "commands": [
                    {"text": "Detect Outliers", "command": f"detect outliers in {df.columns[0] if len(df.columns) > 0 else 'column'}"},
                    {"text": "Show Distribution Statistics", "command": "show distribution statistics"},
                    {"text": "Analyze Relationships", "command": "analyze relationships"}
                ]
            }
        ]

    def process_command(self, command: str, available_columns: List[str]) -> Tuple[Optional[Dict[str, Any]], bool]:
        """Process a command using local LLM"""
        try:
            prompt = f"""
            Interpret this data analysis command and return the analysis type and parameters:
            Command: {command}
            Available Columns: {available_columns}
            
            Return the response in this JSON format:
            {{
                "command_type": "string",
                "parameters": {{
                    "column": "string",
                    "target": "string",
                    "features": "string"
                }},
                "explanation": "string"
            }}
            
            Valid command types:
            - summary
            - histogram
            - boxplot
            - correlation
            - missing
            - regression
            - scatter
            """

            response = self._generate_response(prompt)
            
            try:
                result = json.loads(response)
                return result, True
            except json.JSONDecodeError:
                logger.warning("Could not parse LLM response as JSON")
                return None, False

        except Exception as e:
            logger.error(f"Error processing command: {str(e)}")
            return None, False

    def validate_command(self, command_result: Dict[str, Any], available_columns: List[str]) -> Tuple[bool, str]:
        """Validate the command result"""
        try:
            if not command_result:
                return False, "Invalid command result"

            command_type = command_result.get("command_type")
            parameters = command_result.get("parameters", {})

            if not command_type:
                return False, "Missing command type"

            # Validate column names
            for param_name, param_value in parameters.items():
                if param_name in ["column", "target", "features"]:
                    if isinstance(param_value, str):
                        if param_value not in available_columns:
                            return False, f"Column '{param_value}' not found in dataset"
                    elif isinstance(param_value, list):
                        for col in param_value:
                            if col not in available_columns:
                                return False, f"Column '{col}' not found in dataset"

            return True, "Command is valid"

        except Exception as e:
            logger.error(f"Error validating command: {str(e)}")
            return False, f"Error validating command: {str(e)}"

    def get_available_models(self) -> List[str]:
        """Get list of available models"""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                return [model["name"] for model in models]
            return []
        except Exception as e:
            logger.error(f"Error getting available models: {str(e)}")
            return []

    def change_model(self, new_model: str):
        """Change the model being used"""
        self.model_name = new_model
        self._ensure_model_available()

# Backward compatibility - alias for the new class
LLMProcessor = OllamaLLMProcessor 
