from .base import Skill
import numpy as np
import torch
import torch.nn as nn
import pickle
import os
import pandas as pd
import sys
import os
import logging
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
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
        self.smoothing = config['config'].get('smoothing', 'mean')
        self.configuration = configuration
        
        # Initialize MinIO client and logger with configuration
        self.minio_client = get_minio_client(configuration)
        self.logger = logging.getLogger("process_optimization.inference_model")
        
        # Load model and scaler if paths are provided
        self.model = None
        self.scaler = None
        self.metadata = None
        self._temp_files = []  # Track temporary files for cleanup
        self._load_model_and_scaler()

    def _load_model_and_scaler(self):
        """Load the trained model and scaler from MinIO."""
        try:
            # Load model from MinIO
            if self.model_path:
                # Remove ../ prefix and add models prefix since we're loading from MinIO
                minio_model_path = f"models/{self.model_path.replace('../', '')}"
                try:
                    local_model_path = self.minio_client.get_pytorch_model(minio_model_path)
                    self._temp_files.append(local_model_path)
                    
                    # Validate temp file exists before loading
                    if not Path(local_model_path).exists():
                        raise FileNotFoundError(f"Downloaded temp file missing: {local_model_path}")
                    
                    input_size = len(self.inputs)
                    self.model = ANNModel(input_size=input_size, hidden_size=5, output_size=1, dropout_rate=0.2)
                    self.model.load_state_dict(torch.load(local_model_path, map_location=torch.device('cpu')))
                    self.model.eval()
                    self.logger.info(f"Loaded model from MinIO: {minio_model_path}")
                except FileNotFoundError as e:
                    if "No such file or directory" in str(e) and ".pth" in str(e):
                        self.logger.error(f"Temp file missing, invalidating cache and retrying: {e}")
                        # Invalidate cache and retry once
                        strategy_cache = self.minio_client._get_strategy_cache()
                        if strategy_cache:
                            strategy_cache.invalidate_cached_model(minio_model_path)
                        # Retry download
                        local_model_path = self.minio_client.get_pytorch_model(minio_model_path)
                        self._temp_files.append(local_model_path)
                        
                        input_size = len(self.inputs)
                        self.model = ANNModel(input_size=input_size, hidden_size=5, output_size=1, dropout_rate=0.2)
                        self.model.load_state_dict(torch.load(local_model_path, map_location=torch.device('cpu')))
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
                        strategy_cache = self.minio_client._get_strategy_cache()
                        if strategy_cache:
                            strategy_cache.invalidate_cached_scaler(minio_scaler_path)
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
        """Clean up temporary files when object is destroyed."""
        # Only cleanup temp files if they're not cached or if the cache has expired
        if hasattr(self, '_temp_files') and hasattr(self, 'minio_client'):
            self._cleanup_temp_files_if_safe()
    
    def _cleanup_temp_files_if_safe(self):
        """Clean up temporary files only if they're not needed by other processes."""
        try:
            # Get strategy cache to check if models/scalers are still cached
            strategy_cache = self.minio_client._get_strategy_cache()
            if strategy_cache:
                # Only clean up files that are no longer cached
                files_to_cleanup = []
                for temp_file in self._temp_files:
                    # Check if this model is still cached
                    if hasattr(self, 'model_path'):
                        minio_model_path = f"models/{self.model_path.replace('../', '')}"
                        cached_model = strategy_cache.get_cached_model(minio_model_path)
                        if cached_model is None:
                            # Model not cached anymore, safe to cleanup
                            files_to_cleanup.append(temp_file)
                    else:
                        # No model path info, cleanup to be safe
                        files_to_cleanup.append(temp_file)
                
                if files_to_cleanup:
                    self.minio_client.cleanup_temp_files(files_to_cleanup)
            else:
                # No cache available, perform normal cleanup
                self.minio_client.cleanup_temp_files(self._temp_files)
        except Exception as e:
            # If anything goes wrong, fall back to normal cleanup
            self.logger.warning(f"Error in safe cleanup, falling back to normal: {e}")
            self.minio_client.cleanup_temp_files(self._temp_files)

    def execute(self, context):
        """Execute the model inference, supporting parallel execution."""
        # Get input values - use current values for informative variables, dof values for operative variables
        input_values = []
        
        for input_id in self.inputs:
            var = context.get_variable(input_id)
            
            if var.var_type == 'Delta':
                # This is an operative variable, calculate delta
                dof_val = var.dof_value
                current_val = var.current_value
                delta = dof_val - current_val
                input_values.append(delta)
            else:
                value = 0
                input_values.append(value)

        # Use neural network if available, otherwise return 0
        if self.model is not None and self.scaler is not None:
            result = self._predict_with_nn(input_values, context)
        else:
            result = 0.0  # Default fallback

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
            