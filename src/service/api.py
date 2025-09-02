"""
API Service - Handles REST API endpoints for on-demand optimization.
"""

import structlog
import threading
import tempfile
import yaml
import os
import sys
import logging
from typing import Dict
import traceback
from datetime import datetime
from typing import Dict, Any, Optional
from flask import Flask, request, jsonify
from flask_cors import CORS

from task.math_optimizer.strategy.strategy import OptimizationStrategy
from storage.in_memory_cache import get_cache


class APIService:
    """Service for handling REST API requests."""
    
    def __init__(self, host: str = "0.0.0.0", port: int = 5000, debug: bool = False, configuration: Dict = None):
        """
        Initialize the API service.
        
        Args:
            host: Host to bind to
            port: Port to bind to
            debug: Enable debug mode
            configuration: Configuration dictionary from config.yaml
        """
        self.host = host
        self.port = port
        self.debug = debug
        self.configuration = configuration or {}
        self.logger = structlog.get_logger("process_optimization.api")
        
        # Create Flask app
        self.app = Flask(__name__)
        CORS(self.app)  # Enable CORS for all routes
        
        # Configure Flask logging
        if not debug:
            log = logging.getLogger('werkzeug')
            log.setLevel(logging.WARNING)
        
        # Register routes
        self._register_routes()
        
        # Server instance
        self.server_thread: Optional[threading.Thread] = None
        
    def _register_routes(self):
        """Register all API routes."""
        self.app.add_url_rule('/health', 'health_check', self._health_check, methods=['GET'])
        self.app.add_url_rule('/optimize', 'run_optimization', self._run_optimization, methods=['POST'])
        self.app.add_url_rule('/cache/stats', 'get_cache_stats', self._get_cache_stats, methods=['GET'])
        self.app.add_url_rule('/cache/clear', 'clear_cache', self._clear_cache, methods=['POST'])
        
        # Error handlers
        self.app.errorhandler(404)(self._not_found)
        self.app.errorhandler(500)(self._internal_error)
    
    def _validate_request_data(self, data: Dict[str, Any]) -> tuple[bool, str]:
        """
        Validate the request data structure.
        
        Args:
            data: Request JSON data
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not isinstance(data, dict):
            return False, "Request body must be a JSON object"
        
        # Check required fields
        required_fields = ['input_data', 'config']
        for field in required_fields:
            if field not in data:
                return False, f"Missing required field: {field}"
        
        # Validate input_data structure
        input_data = data.get('input_data')
        if not isinstance(input_data, dict):
            return False, "input_data must be a JSON object"
        
        # Validate config structure
        config = data.get('config')
        if not isinstance(config, dict):
            return False, "config must be a JSON object"
        
        # Check if config has required sections
        required_config_sections = ['variables', 'skills', 'tasks']
        for section in required_config_sections:
            if section not in config:
                return False, f"Missing required config section: {section}"
        
        return True, ""
    
    def _run_single_optimization_cycle(self, input_data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a single optimization cycle with provided input data and config.
        
        Args:
            input_data: Dictionary containing variable values
            config: Configuration dictionary with variables, skills, and tasks
            
        Returns:
            Dictionary containing optimization results
        """
        try:
            self.logger.info("Starting single optimization cycle via API")
            
            # Create temporary config file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as temp_config:
                yaml.dump(config, temp_config)
                temp_config_path = temp_config.name
            
            try:
                # Create strategy with the provided config and configuration
                strategy = OptimizationStrategy(config_path=temp_config_path, use_minio=False, configuration=self.configuration)
                self.logger.info(f"Strategy loaded with {len(strategy.get_operative_variable_ids())} operative variables")
                
                # Validate input data contains required variables
                operative_vars = strategy.get_operative_variable_ids()
                informative_vars = strategy.get_informative_variable_ids()
                calculated_vars = strategy.get_calculated_variable_ids()
                required_vars = operative_vars + informative_vars
                
                missing_vars = []
                for var in required_vars:
                    if var not in input_data:
                        missing_vars.append(var)
                
                if missing_vars:
                    raise ValueError(f"Missing required variables in input_data: {missing_vars}")
                
                # Run the optimization cycle
                self.logger.info("Running optimization cycle...")
                final_context = strategy.run_cycle(input_data)
                
                # Extract results (no post-processing for API calls)
                results = self._extract_optimization_results(final_context, strategy)
                
                self.logger.info(f"Optimization cycle completed successfully. Optimizer changed {results['summary']['total_optimizable_variables_changed']} optimizable variables out of {results['summary']['total_variables']} total variables.")
                return results
                
            finally:
                # Clean up temporary config file
                try:
                    os.unlink(temp_config_path)
                except Exception as e:
                    self.logger.warning(f"Failed to clean up temporary config file: {e}")
            
        except Exception as e:
            self.logger.error(f"Optimization cycle failed: {str(e)}")
            self.logger.error(traceback.format_exc())
            
            return {
                'status': 'error',
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'error_type': type(e).__name__
            }
    
    def _extract_optimization_results(self, final_context, strategy) -> Dict[str, Any]:
        """Extract and format comprehensive optimization results with before/after/delta for all variables."""
        
        # Comprehensive variable results
        all_variables = {}
        
        # Get all variable IDs from strategy
        all_var_ids = set()
        all_var_ids.update(strategy.get_operative_variable_ids())
        all_var_ids.update(strategy.get_informative_variable_ids())
        all_var_ids.update(strategy.get_predicted_variable_ids())
        all_var_ids.update(strategy.get_constraint_variable_ids())
        all_var_ids.update(strategy.get_calculated_variable_ids())
        
        # Process each variable
        for var_id in all_var_ids:
            try:
                var = final_context.get_variable(var_id)
                if var is None:
                    continue
                    
                # Get variable configuration
                var_config = strategy.variables_config.get(var_id, {})
                units = var_config.get('units', '')
                var_type = var_config.get('type', '')
                
                # Determine variable category
                category = self._get_variable_category(var_id, strategy)
                
                # Get user input value and optimizer suggested value
                user_input_value = var.current_value  # This is what user provided
                optimizer_suggested_value = var.dof_value if var.dof_value is not None else var.current_value
                
                # Calculate delta (optimizer suggestion - user input)
                delta = None
                if user_input_value is not None and optimizer_suggested_value is not None:
                    try:
                        delta = float(optimizer_suggested_value) - float(user_input_value)
                    except (ValueError, TypeError):
                        delta = None
                
                # Build variable info
                var_info = {
                    'category': category,
                    'user_input_value': user_input_value,
                    'optimizer_suggested_value': optimizer_suggested_value,
                    'delta': delta,
                    'units': units,
                    'type': var_type
                }
                
                # Add category-specific information
                if category == 'optimizable':
                    var_info['threshold'] = var_config.get('threshold', None)
                    var_info['bounds'] = var_config.get('bounds', None)
                elif category == 'predicted':
                    var_info['model_output'] = True
                elif category == 'constraint':
                    var_info['constraint_type'] = var_config.get('constraint_type', None)
                    var_info['constraint_bounds'] = var_config.get('constraint_bounds', None)
                
                all_variables[var_id] = var_info
                
            except Exception as e:
                self.logger.warning(f"Error processing variable {var_id}: {e}")
                continue
        
        # Get cost function value
        cost_function_value = None
        cost_var = final_context.get_variable('cost_function_total')
        if cost_var and cost_var.dof_value is not None:
            cost_function_value = cost_var.dof_value
        
        # Create categorized views for backward compatibility
        optimized_vars = {var_id: info for var_id, info in all_variables.items() 
                         if info['category'] == 'optimizable'}
        predicted_vars = {var_id: info for var_id, info in all_variables.items() 
                         if info['category'] == 'predicted'}
        constraint_vars = {var_id: info for var_id, info in all_variables.items() 
                          if info['category'] == 'constraint'}
        
        # Calculate summary statistics
        total_with_optimizer_changes = sum(1 for info in all_variables.values() 
                                          if info['delta'] is not None and abs(info['delta']) > 1e-6)
        total_optimizable_changed = len([info for info in all_variables.values() 
                                        if info['category'] == 'optimizable' and 
                                        info['delta'] is not None and abs(info['delta']) > 1e-6])
        
        return {
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'variables': all_variables,  # NEW: Comprehensive view of all variables
            'optimized_variables': optimized_vars,  # Backward compatibility
            'predicted_variables': predicted_vars,  # Backward compatibility
            'constraint_variables': constraint_vars,  # Backward compatibility
            'cost_function_value': cost_function_value,
            'summary': {
                'total_variables': len(all_variables),
                'total_variables_with_optimizer_changes': total_with_optimizer_changes,
                'total_optimizable_variables_changed': total_optimizable_changed,
                'total_predicted_variables': len(predicted_vars),
                'total_constraints': len(constraint_vars),
                'optimization_impact': f"{total_with_optimizer_changes}/{len(all_variables)} variables have optimizer suggestions different from user input"
            }
        }
    
    def _get_variable_category(self, var_id: str, strategy) -> str:
        """Determine the category of a variable."""
        if var_id in strategy.get_optimizable_variable_ids():
            return 'optimizable'
        elif var_id in strategy.get_predicted_variable_ids():
            return 'predicted'
        elif var_id in strategy.get_constraint_variable_ids():
            return 'constraint'
        elif var_id in strategy.get_operative_variable_ids():
            return 'operative'
        elif var_id in strategy.get_informative_variable_ids():
            return 'informative'
        elif var_id in strategy.get_calculated_variable_ids():
            return 'calculated'
        else:
            return 'unknown'
    
    def _health_check(self):
        """Health check endpoint."""
        cache = get_cache()
        cache_stats = cache.get_cache_stats()
        
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'cache_stats': {
                'config_version': cache_stats.get('current_config_version'),
                'cached_timestamp': cache_stats.get('cached_last_run_timestamp'),
                'total_cached_items': sum(
                    stats['active_items'] for stats in cache_stats.values() 
                    if isinstance(stats, dict) and 'active_items' in stats
                )
            }
        })
    
    def _run_optimization(self):
        """Run optimization endpoint."""
        try:
            # Validate request
            if not request.is_json:
                return jsonify({
                    'status': 'error',
                    'timestamp': datetime.now().isoformat(),
                    'error': 'Request must be JSON'
                }), 400
            
            data = request.get_json()
            
            # Validate request data
            is_valid, error_message = self._validate_request_data(data)
            if not is_valid:
                return jsonify({
                    'status': 'error',
                    'timestamp': datetime.now().isoformat(),
                    'error': error_message
                }), 400
            
            # Extract input data and config
            input_data = data['input_data']
            config = data['config']
            
            self.logger.info(f"Received optimization request with {len(input_data)} input variables")
            
            # Run optimization
            result = self._run_single_optimization_cycle(input_data, config)
            
            # Return appropriate status code
            status_code = 200 if result['status'] == 'success' else 500
            
            return jsonify(result), status_code
            
        except Exception as e:
            self.logger.error(f"API endpoint error: {str(e)}")
            self.logger.error(traceback.format_exc())
            
            return jsonify({
                'status': 'error',
                'timestamp': datetime.now().isoformat(),
                'error': 'Internal server error',
                'error_details': str(e)
            }), 500
    
    def _get_cache_stats(self):
        """Get current cache statistics."""
        try:
            cache = get_cache()
            stats = cache.get_cache_stats()
            
            # Format stats for API response
            formatted_stats = {
                'status': 'success',
                'timestamp': datetime.now().isoformat(),
                'current_config_version': stats.get('current_config_version'),
                'cached_last_run_timestamp': stats.get('cached_last_run_timestamp'),
                'cache_details': {}
            }
            
            # Add cache details
            for cache_type, cache_stats in stats.items():
                if isinstance(cache_stats, dict) and 'active_items' in cache_stats:
                    formatted_stats['cache_details'][cache_type] = {
                        'active_items': cache_stats['active_items'],
                        'expired_items': cache_stats['expired_items'],
                        'total_items': cache_stats['total_items']
                    }
            
            return jsonify(formatted_stats)
            
        except Exception as e:
            self.logger.error(f"Cache stats endpoint error: {str(e)}")
            return jsonify({
                'status': 'error',
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }), 500
    
    def _clear_cache(self):
        """Clear all caches."""
        try:
            cache = get_cache()
            cache.clear_all_caches()
            
            return jsonify({
                'status': 'success',
                'timestamp': datetime.now().isoformat(),
                'message': 'All caches cleared successfully'
            })
            
        except Exception as e:
            self.logger.error(f"Cache clear endpoint error: {str(e)}")
            return jsonify({
                'status': 'error',
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }), 500
    
    def _not_found(self, error):
        """Handle 404 errors."""
        return jsonify({
            'status': 'error',
            'timestamp': datetime.now().isoformat(),
            'error': 'Endpoint not found'
        }), 404
    
    def _internal_error(self, error):
        """Handle 500 errors."""
        return jsonify({
            'status': 'error',
            'timestamp': datetime.now().isoformat(),
            'error': 'Internal server error'
        }), 500
    
    def start(self):
        """Start the API service."""
        self.logger.info(f"Starting API service on {self.host}:{self.port}")
        self.logger.info("Available endpoints:")
        self.logger.info("   GET  /health          - Health check")
        self.logger.info("   POST /optimize        - Run optimization cycle")
        self.logger.info("   GET  /cache/stats     - Get cache statistics")
        self.logger.info("   POST /cache/clear     - Clear all caches")
        
        try:
            # Use threaded=True for better shutdown handling
            self.app.run(
                host=self.host, 
                port=self.port, 
                debug=self.debug, 
                use_reloader=False,
                threaded=True
            )
        except KeyboardInterrupt:
            self.logger.info("API service interrupted by user")
        except Exception as e:
            self.logger.error(f"Failed to start API service: {e}")
            raise
    
    def stop(self):
        """Stop the API service."""
        # Flask's development server doesn't have a clean shutdown method
        # In production, this would be handled by the WSGI server
        self.logger.info("API service stop requested")
