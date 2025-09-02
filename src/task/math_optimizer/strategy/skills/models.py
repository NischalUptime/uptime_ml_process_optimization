from .base import Skill
import numpy as np
import torch
import torch.nn as nn
import pickle
import os
import pandas as pd
import sys
import os
import structlog
from storage.minio import get_minio_client

class ANNModel(nn.Module):
    def __init__(self, input_size=5, hidden_size=5, output_size=1, dropout_rate=0.2):
        super(ANNModel, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_size, hidden_size*2),
            nn.BatchNorm1d(hidden_size*2), 
            nn.GELU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_size*2, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.network(x)

class InferenceModel(Skill):
    """
    A neural network inference model that loads trained PyTorch models and handles scaling.
    """
    def __init__(self, name, config, configuration=None):
        super().__init__(name, config)
        self.model_type = config['config'].get('model_type', 'ANN')
        self.model_path = config['config'].get('model_path', None)
        self.scaler_path = config['config'].get('scaler_path', None)
        self.metadata_path = config['config'].get('metadata_path', None)
        self.feature_engineering = config['config'].get('feature_engineering', {})
        self.lag_offset = self.feature_engineering.get('lag_offset', {})
        self.smoothing = self.feature_engineering.get('smoothing', {})
        self.configuration = configuration
        
        # Initialize MinIO client and logger with configuration
        self.minio_client = get_minio_client(configuration)
        self.logger = structlog.get_logger("process_optimization.inference_model")
        
        # Load model and scaler if paths are provided
        self.model = None
        self.scaler = None
        self.metadata = None
        # Note: Models are now cached in memory, no temp files needed
        self._load_model_and_scaler()

    def _load_model_and_scaler(self):
        """Load the trained model and scaler from MinIO."""
        try:
            # Load model from MinIO
            if self.model_path:
                # Remove ../ prefix and add models prefix since we're loading from MinIO
                minio_model_path = f"models/{self.model_path.replace('../', '')}"
                try:
                    # Load model directly from MinIO (now returns model object, not file path)
                    model_state_dict = self.minio_client.get_pytorch_model(minio_model_path)
                    
                    input_size = len(self.inputs)
                    self.model = ANNModel(input_size=input_size, hidden_size=5, output_size=1, dropout_rate=0.2)
                    self.model.load_state_dict(model_state_dict)
                    self.model.eval()
                    self.logger.info(f"Loaded model from MinIO: {minio_model_path}")
                except Exception as e:
                    if "corrupted" in str(e).lower() or "cache" in str(e).lower():
                        self.logger.error(f"Model loading failed, invalidating cache and retrying: {e}")
                        # Invalidate cache and retry once
                        cache = self.minio_client._get_cache()
                        if cache:
                            cache.invalidate_cached_model(minio_model_path)
                        # Retry loading
                        model_state_dict = self.minio_client.get_pytorch_model(minio_model_path)
                        
                        input_size = len(self.inputs)
                        self.model = ANNModel(input_size=input_size, hidden_size=5, output_size=1, dropout_rate=0.2)
                        self.model.load_state_dict(model_state_dict)
                        self.model.eval()
                        self.logger.info(f"Successfully reloaded model after cache invalidation: {minio_model_path}")
                    else:
                        raise
            
            # Load scaler from MinIO
            if self.scaler_path:
                # Remove ../ prefix and add models prefix since we're loading from MinIO
                minio_scaler_path = f"models/{self.scaler_path.replace('../', '')}"
                try:
                    self.scaler = self.minio_client.get_pickle_scaler(minio_scaler_path)
                    self.logger.info(f"Loaded scaler from MinIO: {minio_scaler_path}")
                    self.logger.debug(f"Model inputs: {self.inputs}")
                    self.logger.debug(f"Model outputs: {self.outputs}")
                except Exception as e:
                    if "No such file or directory" in str(e) or "corrupted" in str(e).lower():
                        self.logger.error(f"Scaler issue detected, invalidating cache and retrying: {e}")
                        # Invalidate cache and retry once
                        cache = self.minio_client._get_cache()
                        if cache:
                            cache.invalidate_cached_scaler(minio_scaler_path)
                        # Retry download
                        self.scaler = self.minio_client.get_pickle_scaler(minio_scaler_path)
                        self.logger.info(f"Successfully reloaded scaler after cache invalidation: {minio_scaler_path}")
                    else:
                        raise
            
            # Load metadata from MinIO (optional)
            if self.metadata_path:
                # Remove ../ prefix and add models prefix since we're loading from MinIO
                minio_metadata_path = f"models/{self.metadata_path.replace('../', '')}"
                self.metadata = self.minio_client.get_json_metadata(minio_metadata_path)
                self.logger.debug(f"Loaded metadata from MinIO: {minio_metadata_path}")
                
        except Exception as e:
            self.logger.warning(f"Warning: Could not load model/scaler from MinIO: {e}")
            self.model = None
            self.scaler = None
            self.metadata = None
            
    def __del__(self):
        """Clean up resources when object is destroyed."""
        # Models and scalers are now cached in memory, no temp file cleanup needed
        pass

    def execute(self, context):
        """Execute the model inference, supporting parallel execution."""
        # Get input values - use current values for informative variables, dof values for operative variables
        input_values = []
        for input_id in self.inputs:
            var = context.get_variable(input_id)
            value = 0
            if var.var_type == 'Delta':
                # This is an operative variable, calculate delta
                value = var.dof_value - var.current_value
            elif var.var_type in ['Operative', 'Calculated']:
                variation = self.lag_offset.get(var.var_id, {}).get("variation")
                if variation == 'Increment':
                    value = var.dof_value - var.current_value
                elif variation == 'Absolute':
                    value = var.dof_value
            elif var.var_type == 'Informative':
                variation = self.lag_offset.get(var.var_id, {}).get('variation')
                if variation == 'Increment':
                    value = var.dof_value - var.current_value
                elif variation == 'Absolute':
                    # Get smoothing method
                    smoothing_method = self.smoothing.get("method", "mean")
                    smoothing_alpha = self.smoothing.get("alpha", 0.7)
                    
                    # Get lag/offset info for this variable
                    lag_offset_cfg = self.lag_offset.get(var.var_id, {})
                    lag = lag_offset_cfg.get("lag", 0)
                    offset = lag_offset_cfg.get("offset", 0)
                    
                    # Get the dataframe slice
                    df = context.get_dataframe()
                    series_window = df[var.var_id].iloc[-(lag + offset): -lag]
                    
                    if series_window.empty:
                        value = 0
                    else:
                        if smoothing_method == "ewm":
                            smoothed_series = series_window.ewm(alpha=smoothing_alpha, adjust=False).mean()
                            print(f"EWM Series ({var.var_id}):\n{smoothed_series}")
                            value = smoothed_series.iloc[-1]
                        elif smoothing_method == "mean":
                            mean_value = series_window.mean()
                            print(f"Mean ({var.var_id}): {mean_value}")
                            value = mean_value
            input_values.append(value)

        # Use neural network if available, otherwise return 0
        if self.model is not None and self.scaler is not None:
            result = self._predict_with_nn(input_values, context)
        else:
            result = 0.0  # Default fallback
        # self.logger.debug(f"Input values: {input_values}")
        # self.logger.debug(f"Prediction result for {self.name}: {result}")
        # Set output value
        output_var = context.get_variable(self.outputs[0])
        output_var.dof_value = result

    def _predict_with_nn(self, input_values, context):
        """Make prediction using the neural network model with optimized processing."""
        try:
            if any(v is None for v in input_values):
                return 0.0
            
            # Vectorized scaling of inputs with proper feature names
            scaled_inputs = []
            for i, input_id in enumerate(self.inputs):
                scaler_id = input_id.replace('delta_', '')
                if scaler_id in self.scaler:
                    # Create a properly named DataFrame for scaling
                    input_df = pd.DataFrame({scaler_id: [input_values[i]]})
                    scaled_val = self.scaler[scaler_id].transform(input_df)[0][0]
                else:
                    scaled_val = input_values[i]
                scaled_inputs.append(scaled_val)
            
            # Convert to tensor
            model_inputs = torch.tensor([scaled_inputs], dtype=torch.float32)
            with torch.no_grad():
                diff_prediction = self.model(model_inputs).item()
                
            # Inverse scale the prediction
            output_id = self.outputs[0]
            scaler_id = output_id.replace('delta_', '')
            if scaler_id in self.scaler:
                # Create a properly named DataFrame for inverse scaling
                output_df = pd.DataFrame({scaler_id: [diff_prediction]})
                diff_prediction = self.scaler[scaler_id].inverse_transform(output_df)[0][0]
            else:
                print(f"Scaler not found for {scaler_id}")
            
            # Calculate final prediction
            current_target_var = context.get_variable(output_id)
            current_target_value = current_target_var.current_value if current_target_var.current_value is not None else 0.0
            return current_target_value + diff_prediction

            
        except Exception as e:
            return 0.0  # Default fallback
            