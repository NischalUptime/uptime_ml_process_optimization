import yaml
import structlog
import sys
import os
from datetime import datetime
from typing import Optional, Dict

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
from storage.minio import get_minio_client
from storage.in_memory_cache import get_cache

class StrategyManager:
    def __init__(self, configuration: Dict = None, 
                 timestamp_file: str = 'src/task/math_optimizer/strategy_manager/last_run_timestamp.yaml', 
                 deployed_config_file: str = 'src/task/math_optimizer/strategy_manager/strategy_version.yaml'):
        """Initialize StrategyManager with configuration from config.yaml.
        
        Args:
            configuration: Configuration dictionary from config.yaml
            timestamp_file: Path to timestamp file
            deployed_config_file: Path to strategy version file
        """
        self.configuration = configuration or {}
        self.timestamp_file = timestamp_file
        self.deployed_config_file = deployed_config_file
        self.minio_client = get_minio_client(configuration)
        self.cache = get_cache()
        self.logger = structlog.get_logger("process_optimization.strategy_manager")

    def get_last_run_timestamp(self) -> Optional[datetime]:
        """Get the last run timestamp from cache or file"""
        def _load_timestamp_from_file():
            try:
                with open(self.timestamp_file, 'r') as f:
                    config = yaml.safe_load(f)
                    if config and 'last_run_timestamp' in config:
                        return datetime.fromisoformat(config['last_run_timestamp'])
            except Exception as e:
                print(f"Warning: Could not read last_run_timestamp from file: {e}")
            return None
        
        return self.cache.get_last_run_timestamp_with_cache(_load_timestamp_from_file)

    def update_last_run_timestamp(self, timestamp: datetime):
        """Write the last run timestamp to config file and update cache"""
        # Write to file
        config = {'last_run_timestamp': timestamp.isoformat()}
        with open(self.timestamp_file, 'w') as f:
            yaml.dump(config, f)
        
        # Update cache
        self.cache.set_cached_last_run_timestamp(timestamp)

    def get_deployed_config_version(self) -> str:
        """Read the deployed strategy version from strategy file"""
        try:
            with open(self.deployed_config_file, 'r') as f:
                config = yaml.safe_load(f)
                if config and 'process-optimization-strategy-config.yaml' in config:
                    return config['process-optimization-strategy-config.yaml']
                else:
                    raise Exception("process-optimization-strategy-config.yaml version not found in strategy_version.yaml")
        except Exception as e:
            raise Exception(f"Failed to read deployed config version: {str(e)}")

    def load_strategy_config_from_minio(self) -> dict:
        """Load strategy configuration from MinIO using deployed version"""
        try:
            # Get the deployed config version
            version = self.get_deployed_config_version()
            self.logger.info(f"Loading config version {version} from MinIO...")
            
            # Check if version changed and invalidate cache if needed
            cache_invalidated = self.cache.check_version_and_invalidate_if_needed(version)
            if cache_invalidated:
                self.logger.info("Cache was invalidated due to version change")
                # Cache cleanup is handled automatically in memory
            
            # Load config from MinIO (will use cache if available)
            config = self.minio_client.get_config_by_version(version)
            self.logger.info("Successfully loaded config from MinIO")
            return config
        except Exception as e:
            raise Exception(f"Failed to load strategy config from MinIO: {str(e)}")

    def load_strategy_config(self, config_file: str = None) -> dict:
        """Load strategy configuration from file (fallback method)"""
        try:
            # Use config file from main configuration if not specified
            if config_file is None:
                config_file = self.configuration.get('optimization', {}).get('config_file', 'process-optimization-config.yaml')
            
            with open(config_file, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            raise Exception(f"Failed to load strategy config: {str(e)}")