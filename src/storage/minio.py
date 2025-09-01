"""
MinIO client utilities for reading configuration and model files from MinIO storage.
"""

import io
import json
import os
import pickle

import time
from pathlib import Path
from typing import Any, Dict, Optional

import structlog
import yaml
from minio import Minio
from minio.error import S3Error
from .in_memory_cache import get_cache

logger = structlog.get_logger(__name__)





class MinIOClient:
    """MinIO client for reading configuration and model files."""
    
    def __init__(self, endpoint: str = "localhost:9002", 
                 access_key: str = "user", 
                 secret_key: str = "password",
                 secure: bool = False):
        """
        Initialize MinIO client.
        
        Args:
            endpoint: MinIO server endpoint
            access_key: MinIO access key
            secret_key: MinIO secret key
            secure: Whether to use HTTPS
        """
        logger.info("Initializing MinIO client", 
                   endpoint=endpoint, 
                   secure=secure,
                   access_key_masked=f"{access_key[:4]}***" if len(access_key) > 4 else "***")
        
        try:
            self.client = Minio(
                endpoint,
                access_key=access_key,
                secret_key=secret_key,
                secure=secure
            )
            self.config_bucket = "process-optimization"
            self.models_bucket = "process-optimization"
            # Lazy import to avoid circular dependency
            self._cache = None
            
            logger.info("MinIO client initialized successfully", 
                       config_bucket=self.config_bucket,
                       models_bucket=self.models_bucket)
        except Exception as e:
            logger.error("Failed to initialize MinIO client", error=str(e), endpoint=endpoint)
            raise
    
    def _get_cache(self):
        """Get the cache instance."""
        if self._cache is None:
            try:
                logger.debug("Getting cache instance")
                self._cache = get_cache()
                logger.debug("Cache instance obtained successfully")
            except Exception as e:
                # If cache not available, disable caching
                logger.warning("Cache not available, caching disabled", error=str(e))
                return None
        return self._cache
    
    def get_config_by_version(self, version: str) -> Dict[str, Any]:
        """
        Read configuration YAML file from MinIO by version with caching.
        
        Args:
            version: Version string (e.g., "1.0.0")
            
        Returns:
            Dictionary containing configuration data
            
        Raises:
            Exception: If config file not found or invalid
        """
        def _load_config_from_minio(version):
            config_filename = f"configs/process-optimization-strategy-config-{version}.yaml"
            logger.info("Loading config from MinIO", 
                       version=version, 
                       filename=config_filename, 
                       bucket=self.config_bucket)
            
            try:
                # Get object from MinIO
                response = self.client.get_object(self.config_bucket, config_filename)
                logger.debug("MinIO object retrieved successfully", filename=config_filename)
                
                # Read and parse YAML
                config_data = yaml.safe_load(response.data.decode('utf-8'))
                response.close()
                response.release_conn()
                
                logger.info("Config loaded and parsed successfully", 
                           version=version, 
                           config_keys=list(config_data.keys()) if isinstance(config_data, dict) else "non-dict")
                return config_data
                
            except S3Error as e:
                if e.code == 'NoSuchKey':
                    logger.error("Config file not found in MinIO", 
                                filename=config_filename, 
                                bucket=self.config_bucket,
                                version=version)
                    raise FileNotFoundError(f"Config file {config_filename} not found in MinIO bucket {self.config_bucket}")
                logger.error("MinIO S3 error while reading config", 
                            error=str(e), 
                            error_code=e.code,
                            filename=config_filename)
                raise Exception(f"MinIO error while reading config: {e}")
            except yaml.YAMLError as e:
                logger.error("YAML parsing error", 
                            error=str(e), 
                            filename=config_filename)
                raise Exception(f"Error parsing YAML config: {e}")
            except Exception as e:
                logger.error("Unexpected error reading config from MinIO", 
                            error=str(e), 
                            filename=config_filename)
                raise Exception(f"Error reading config from MinIO: {e}")
        
        # Check cache first  
        cache = self._get_cache()
        if cache:
            logger.debug("Checking cache for config", version=version)
            cached_config = cache.get_cached_config(version)
            if cached_config is not None:
                logger.info("Config found in cache", version=version)
                return cached_config
            logger.debug("Config not found in cache", version=version)
        
        # Load from MinIO and cache
        config = _load_config_from_minio(version)
        if cache:
            logger.debug("Caching loaded config", version=version)
            cache.set_cached_config(version, config)
        return config
    
    def get_pytorch_model(self, model_path: str) -> Any:
        """
        Load PyTorch model file (.pth) from MinIO directly into memory with caching.
        
        Args:
            model_path: Path to model file in MinIO (e.g., "saved_models/model.pth")
            
        Returns:
            Loaded PyTorch model object
            
        Raises:
            Exception: If model file not found or loading fails
        """
        def _load_model_from_minio(model_path):
            logger.info("Loading model from MinIO", 
                       model_path=model_path, 
                       bucket=self.models_bucket)
            
            try:
                # Get object from MinIO
                response = self.client.get_object(self.models_bucket, model_path)
                logger.debug("MinIO object retrieved successfully", model_path=model_path)
                
                # Load model directly into memory using BytesIO
                import io
                import torch
                model_bytes = io.BytesIO(response.data)
                model_data = torch.load(model_bytes, map_location='cpu')
                response.close()
                response.release_conn()
                
                logger.info("Model loaded directly into memory successfully", 
                           model_path=model_path, 
                           model_type=type(model_data).__name__)
                return model_data
                
            except S3Error as e:
                if e.code == 'NoSuchKey':
                    logger.error("Model file not found in MinIO", 
                                model_path=model_path, 
                                bucket=self.models_bucket)
                    raise FileNotFoundError(f"Model file {model_path} not found in MinIO bucket {self.models_bucket}")
                logger.error("MinIO S3 error while loading model", 
                            error=str(e), 
                            error_code=e.code,
                            model_path=model_path)
                raise Exception(f"MinIO error while loading model: {e}")
            except Exception as e:
                logger.error("Unexpected error loading model from MinIO", 
                            error=str(e), 
                            model_path=model_path)
                raise Exception(f"Error loading model from MinIO: {e}")
        
        # Check cache first
        cache = self._get_cache()
        if cache:
            logger.debug("Checking cache for model", model_path=model_path)
            cached_model = cache.get_cached_model(model_path)
            if cached_model is not None:
                try:
                    # Validate that the cached model is still usable
                    # Try to access a basic attribute to ensure it's not corrupted
                    _ = getattr(cached_model, '__class__', None)
                    logger.info("Model found in cache and validated", model_path=model_path)
                    logger.info(f"Using cached model object for: {model_path}")
                    return cached_model
                except Exception as e:
                    logger.warning("Cached model is corrupted, invalidating cache", 
                                  model_path=model_path, 
                                  error=str(e))
                    # Invalidate this specific cache entry
                    cache.invalidate_cached_model(model_path)
            else:
                logger.debug("Model not found in cache", model_path=model_path)
        
        # Load from MinIO and cache
        model = _load_model_from_minio(model_path)
        if cache:
            logger.info("Caching loaded model", model_path=model_path)
            cache.set_cached_model(model_path, model)
            logger.info(f"Model cached in memory for: {model_path}")
        return model
    
    def get_pickle_scaler(self, scaler_path: str) -> Any:
        """
        Read pickle scaler file (.pkl) from MinIO and deserialize with caching.
        
        Args:
            scaler_path: Path to scaler file in MinIO (e.g., "saved_scalers/scaler.pkl")
            
        Returns:
            Deserialized scaler object
            
        Raises:
            Exception: If scaler file not found or deserialization fails
        """
        def _load_scaler_from_minio(scaler_path):
            logger.info("Loading scaler from MinIO", 
                       scaler_path=scaler_path, 
                       bucket=self.models_bucket)
            
            try:
                # Get object from MinIO
                response = self.client.get_object(self.models_bucket, scaler_path)
                logger.debug("MinIO object retrieved successfully", scaler_path=scaler_path)
                
                # Deserialize pickle data
                scaler_data = pickle.loads(response.data)
                response.close()
                response.release_conn()
                
                logger.info("Scaler loaded and deserialized successfully", 
                           scaler_path=scaler_path, 
                           scaler_type=type(scaler_data).__name__)
                return scaler_data
                
            except S3Error as e:
                if e.code == 'NoSuchKey':
                    logger.error("Scaler file not found in MinIO", 
                                scaler_path=scaler_path, 
                                bucket=self.models_bucket)
                    raise FileNotFoundError(f"Scaler file {scaler_path} not found in MinIO bucket {self.models_bucket}")
                logger.error("MinIO S3 error while reading scaler", 
                            error=str(e), 
                            error_code=e.code,
                            scaler_path=scaler_path)
                raise Exception(f"MinIO error while reading scaler: {e}")
            except pickle.PickleError as e:
                logger.error("Pickle deserialization error", 
                            error=str(e), 
                            scaler_path=scaler_path)
                raise Exception(f"Error deserializing pickle scaler: {e}")
            except Exception as e:
                logger.error("Unexpected error reading scaler from MinIO", 
                            error=str(e), 
                            scaler_path=scaler_path)
                raise Exception(f"Error reading scaler from MinIO: {e}")
        
        # Check cache first
        cache = self._get_cache()
        if cache:
            logger.debug("Checking cache for scaler", scaler_path=scaler_path)
            cached_scaler = cache.get_cached_scaler(scaler_path)
            if cached_scaler is not None:
                try:
                    # Validate that the cached scaler is still usable
                    # Try to access a basic attribute to ensure it's not corrupted
                    _ = getattr(cached_scaler, '__class__', None)
                    logger.info("Scaler found in cache and validated", scaler_path=scaler_path)
                    logger.info(f"Using cached scaler object for: {scaler_path}")
                    return cached_scaler
                except Exception as e:
                    logger.warning("Cached scaler is corrupted, invalidating cache", 
                                  scaler_path=scaler_path, 
                                  error=str(e))
                    # Invalidate this specific cache entry
                    cache.invalidate_cached_scaler(scaler_path)
            else:
                logger.debug("Scaler not found in cache", scaler_path=scaler_path)
        
        # Load from MinIO and cache
        scaler = _load_scaler_from_minio(scaler_path)
        if cache:
            logger.info("Caching loaded scaler", scaler_path=scaler_path)
            cache.set_cached_scaler(scaler_path, scaler)
            logger.info(f"Scaler cached in memory for: {scaler_path}")
        return scaler
    
    def get_json_metadata(self, metadata_path: str) -> Dict[str, Any]:
        """
        Read JSON metadata file from MinIO with caching.
        
        Args:
            metadata_path: Path to metadata file in MinIO (e.g., "saved_metadata/meta.json")
            
        Returns:
            Dictionary containing metadata
            
        Raises:
            Exception: If metadata file not found or invalid JSON
        """
        def _load_metadata_from_minio(metadata_path):
            logger.info("Loading metadata from MinIO", 
                       metadata_path=metadata_path, 
                       bucket=self.models_bucket)
            
            try:
                # Get object from MinIO
                response = self.client.get_object(self.models_bucket, metadata_path)
                logger.debug("MinIO object retrieved successfully", metadata_path=metadata_path)
                
                # Parse JSON
                metadata = json.loads(response.data.decode('utf-8'))
                response.close()
                response.release_conn()
                
                logger.info("Metadata loaded and parsed successfully", 
                           metadata_path=metadata_path, 
                           metadata_keys=list(metadata.keys()) if isinstance(metadata, dict) else "non-dict")
                return metadata
                
            except S3Error as e:
                if e.code == 'NoSuchKey':
                    logger.error("Metadata file not found in MinIO", 
                                metadata_path=metadata_path, 
                                bucket=self.models_bucket)
                    raise FileNotFoundError(f"Metadata file {metadata_path} not found in MinIO bucket {self.models_bucket}")
                logger.error("MinIO S3 error while reading metadata", 
                            error=str(e), 
                            error_code=e.code,
                            metadata_path=metadata_path)
                raise Exception(f"MinIO error while reading metadata: {e}")
            except json.JSONDecodeError as e:
                logger.error("JSON parsing error", 
                            error=str(e), 
                            metadata_path=metadata_path)
                raise Exception(f"Error parsing JSON metadata: {e}")
            except Exception as e:
                logger.error("Unexpected error reading metadata from MinIO", 
                            error=str(e), 
                            metadata_path=metadata_path)
                raise Exception(f"Error reading metadata from MinIO: {e}")
        
        # Note: Metadata caching not implemented in new cache
        # Load directly from MinIO for now
        logger.debug("Loading metadata directly from MinIO (no caching)", metadata_path=metadata_path)
        return _load_metadata_from_minio(metadata_path)


def get_minio_client(configuration: Dict = None) -> MinIOClient:
    """
    Factory function to create MinIO client with configuration settings.
    
    Args:
        configuration: Configuration dictionary containing storage.minio settings
    
    Returns:
        Configured MinIOClient instance
    """
    logger.debug("Creating MinIO client from configuration")
    
    if configuration and 'storage' in configuration and 'minio' in configuration['storage']:
        minio_config = configuration['storage']['minio']
        logger.debug("Using MinIO configuration from config file", 
                    endpoint=minio_config.get('endpoint', 'localhost:9002'))
        return MinIOClient(
            endpoint=minio_config.get('endpoint', 'localhost:9002'),
            access_key=minio_config.get('access_key', 'user'),
            secret_key=minio_config.get('secret_key', 'password'),
            secure=minio_config.get('secure', False)
        )
    
    # Fallback to default settings
    logger.debug("Using default MinIO configuration (fallback)")
    return MinIOClient(endpoint="localhost:9002", access_key="user", secret_key="password")