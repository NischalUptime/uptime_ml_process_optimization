import yaml
from .data_context import DataContext
from .skills.models import InferenceModel
from .skills.functions import MathFunction
from .skills.constraints import Constraint
from .skills.composition import CompositionSkill
from .skills.optimizer import OptimizationSkill
from task.math_optimizer.strategy_manager.strategy_manager import StrategyManager

class OptimizationStrategy:
    """
    The main orchestrator. Loads strategy, builds skills, and runs the cycle.
    """
    SKILL_CLASS_MAP = {
        'InferenceModel': InferenceModel,
        'MathFunction': MathFunction,
        'Constraint': Constraint,
        'CompositionSkill': CompositionSkill,
        'OptimizationSkill': OptimizationSkill,
    }

    def __init__(self, config_path=None, use_minio=True, configuration=None):
        self.configuration = configuration
        
        if use_minio:
            # Load config from MinIO using StrategyManager with configuration
            strategy_manager = StrategyManager(configuration=configuration)
            self.config = strategy_manager.load_strategy_config_from_minio()
        else:
            # Fallback to local file loading
            if not config_path:
                config_path = 'config.yaml'
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        
        self.variables_config = self.config['variables']
        self.skills_config = self.config['skills']
        self.tasks_config = self.config['tasks']

        self._skills = self._build_skills()

    def _build_skills(self):
        """Instantiates all skill objects from the configuration."""
        skills = {}
        # First pass: instantiate all skills
        for name, config in self.skills_config.items():
            skill_class = self.SKILL_CLASS_MAP.get(config['class'])
            if not skill_class:
                raise ValueError(f"Unknown skill class: {config['class']}")
            
            # Pass configuration to skills that need MinIO access (like InferenceModel)
            if config['class'] == 'InferenceModel':
                skills[name] = skill_class(name, config, configuration=self.configuration)
            else:
                skills[name] = skill_class(name, config)
            
            # Set strategy reference for optimizer skills
            if isinstance(skills[name], OptimizationSkill):
                skills[name].set_strategy(self)
        
        # Second pass: resolve CompositionSkill dependencies
        for skill in skills.values():
            if isinstance(skill, CompositionSkill):
                skill.resolve_skills(skills)
        
        return skills

    def get_operative_variable_ids(self):
        """Returns a list of operative variable IDs."""
        return [
            var_id for var_id, config in self.variables_config.items()
            if config['type'] == 'Operative'
        ]

    def get_calculated_variable_ids(self):
        """Returns a list of calculated variable IDs."""
        return [
            var_id for var_id, config in self.variables_config.items()
            if config['type'] == 'Calculated'
        ]

    def get_informative_variable_ids(self):
        """Returns a list of informative variable IDs."""
        return [
            var_id for var_id, config in self.variables_config.items()
            if config['type'] == 'Informative'
        ]

    def get_delta_variable_ids(self):
        """Returns a list of delta variable IDs."""
        return [
            var_id for var_id, config in self.variables_config.items()
            if config['type'] == 'Delta'
        ]

    def get_predicted_variable_ids(self):
        """Returns a list of predicted variable IDs."""
        return [
            var_id for var_id, config in self.variables_config.items()
            if config['type'] == 'Predicted'
        ]

    def get_constraint_variable_ids(self):
        """Returns a list of constraint variable IDs."""
        return [
            var_id for var_id, config in self.variables_config.items()
            if config['type'] == 'Constraint'
        ]

    def get_optimizable_variable_ids(self):
        """Returns a list of variables that can be optimized (Calculated variables + operative variables that remain operative)."""
        # After pre-calculation, both calculated variables and operative variables that are not inputs to calculated variables are optimizable
        calculated_ids = self.get_calculated_variable_ids()
        fixed_input_ids = set(self.get_fixed_input_variable_ids())
        all_operative_ids = set(self.get_operative_variable_ids())
        
        # Operative variables that are NOT inputs to calculated variables remain operative
        remaining_operative_ids = all_operative_ids - fixed_input_ids
        
        return list(calculated_ids) + list(remaining_operative_ids)

    def get_fixed_input_variable_ids(self):
        """Returns a list of variables that become informative after pre-calculation."""
        # These are the operative variables that are inputs to calculated variables
        # They become informative (read-only) after pre-calculation
        
        # Identify operative variables that are inputs to calculated variables
        calculated_input_vars = set()
        
        # Find the PreCalculateVariables task and get its skills
        precalc_task = None
        for task in self.tasks_config:
            if task['name'] == 'PreCalculateVariables':
                precalc_task = task
                break
        
        if precalc_task:
            # Check each skill in the pre-calculation task to find inputs to calculated variables
            for skill_name in precalc_task.get('skill_sequence', []):
                skill_config = self.skills_config.get(skill_name)
                if skill_config:
                    inputs = skill_config.get('inputs', [])
                    calculated_input_vars.update(inputs)
        
        return list(calculated_input_vars)

    def get_raw_vars_from_calculated_vars(self):
        """
        Returns a list of variable IDs that are required as inputs
        for calculated variables (e.g., derived from MathFunction skills).
        """
        calculated_vars = self.get_calculated_variable_ids()

        # Map each calculated output to its input dependencies
        calculated_var_to_raw_vars_map = {}
        for details in self.skills_config.values():
            if details["class"] == "MathFunction":
                for output_var in details.get("outputs", []):
                    if output_var in calculated_vars:
                        calculated_var_to_raw_vars_map[output_var] = details.get("inputs", [])

        # Collect all inputs needed for the calculated variables
        dependent_inputs = []
        required_raw_vars = []
        seen = set()
        for var in calculated_vars:
            if var in calculated_var_to_raw_vars_map:
                for raw_var in calculated_var_to_raw_vars_map[var]:
                    if raw_var not in seen:
                        seen.add(raw_var)
                        required_raw_vars.append(raw_var)
        return required_raw_vars

    def get_lag_offset_bounds(self):
        """
        Returns:
            (min_lag, max_window_size)
            - min_lag: minimum lag across all features
            - max_window_size: maximum value of (lag + offset) across all features
        """
        min_lag = None
        max_window = None

        for details in self.skills_config.values():
            if details.get("class") != "InferenceModel":
                continue
            feature_engineering = details.get("config", {}).get("feature_engineering")
            if feature_engineering is None:
                continue
            
            lag_offset_block = feature_engineering.get("lag_offset", {})
            if not isinstance(lag_offset_block, dict):
                continue
            
            for feature, params in lag_offset_block.items():
                if not isinstance(params, dict):
                    continue
                lag = params.get("lag", 0)
                offset = params.get("offset", 0)
                window = lag + offset
                if min_lag is None or lag < min_lag:
                    min_lag = lag
                if max_window is None or window > max_window:
                    max_window = window

        if min_lag is None or max_window is None:
            return None, None

        return min_lag, max_window

    def run_cycle(self, initial_data):
        """
        Executes a full optimization cycle.

        Parameters
        ----------
        initial_data : dict or list
            - dict of {var: value}  (backward-compatible single row)
            - OR list of {"timestamp": ..., "data": {...}} dicts (time window)
        """
        # 1. Create and populate the data context for this cycle
        data_context = DataContext(self.variables_config)
        data_context.populate_initial_data(initial_data)

        # 2. Execute tasks in the configured sequence
        for task in self.tasks_config:
            task_name = task['name']
            # print(f"Executing task: {task_name}")
            # If this is the pre-calculation task, mark calculated variables as operative
            if task_name == "PreCalculateVariables":
                self._mark_calculated_as_operative(data_context)
            
            for skill_name in task['skill_sequence']:
                if task_name == "PreCalculateVariables":
                    config = self.skills_config.get(skill_name)
                    if config and config['class'] == 'MathFunction':
                        math_function = MathFunction(skill_name, config)
                        math_function.resolve_dataframe_formula(skill_name, data_context)
                skill = self._skills.get(skill_name)
                if not skill:
                    raise ValueError(f"Skill '{skill_name}' in task '{task_name}' not found.")
                skill.execute(data_context)
        
        return data_context

    def _mark_calculated_as_operative(self, data_context):
        """
        Marks calculated variables as operative for optimization after they have been computed.
        Input variables to calculated variables become informative (read-only).
        """
        calculated_vars = self.get_calculated_variable_ids()
        
        # Mark calculated variables as operative
        for var_id in calculated_vars:
            var = data_context.get_variable(var_id)
            if var.dof_value is not None:
                # Set the calculated value as both current and DOF value
                var.current_value = var.dof_value
                # Ensure both values are set for delta calculations
                var.dof_value = var.dof_value  # Keep the calculated value as DOF value initially
                # print(f"Marked {var_id} as operative with value: {var.dof_value}")
            else:
                print(f"Warning: {var_id} has None dof_value after pre-calculation")
