from .base import Skill
from scipy.optimize import minimize

class OptimizationSkill(Skill):
    """
    A skill that performs numerical optimization using scipy's minimize function.
    """
    def __init__(self, name, config):
        super().__init__(name, config)
        self.cost_skill_name = config['config']['cost_skill_name']
        self.cost_feature_name = config['config']['cost_feature_name']
        self.algorithm = config['config'].get('algorithm', 'SLSQP')
        self._strategy = None  # Will be set by strategy.py

    def set_strategy(self, strategy):
        """Set the reference to the parent strategy object."""
        self._strategy = strategy

    def execute(self, context):
        if not self._strategy:
            raise RuntimeError("Strategy reference not set in OptimizationSkill")

        # Get the cost calculation skill
        cost_skill = self._strategy._skills[self.cost_skill_name]
        
        # Get optimizable variables (calculated variables only after pre-calculation)
        optimizable_vars = self._strategy.get_optimizable_variable_ids()
        
        # Filter inputs to only include optimizable variables
        optimizable_inputs = [var_id for var_id in self.inputs if var_id in optimizable_vars]
        
        print(f"Optimizable inputs: {optimizable_inputs}")
        cost_iterations = []
        
        # Define the objective function for scipy.minimize
        def objective(x):
            # Update the DOF values in the context for optimizable variables
            for var_id, value in zip(optimizable_inputs, x):
                context.get_variable(var_id).dof_value = value
            
            # Run the cost calculation
            cost_skill.execute(context)
            
            # Get the cost value
            cost_var = context.get_variable(self.cost_feature_name)
            cost = cost_var.dof_value if cost_var.dof_value is not None else 0.0
            cost_iterations.append(round(cost, 2))
            return cost

        # Get initial values and bounds for optimizable variables
        x0 = []
        bounds = []
        eps_values = []
        dynamic_bounds_map = {}
        try:
            dynamic_bounds_map = context.get_dynamic_bounds()
        except Exception:
            dynamic_bounds_map = {}
        for var_id in optimizable_inputs:
            var = context.get_variable(var_id)
            # Ensure we have a valid current_value
            if var.current_value is None:
                print(f"Warning: {var_id} has None current_value, using 0.0")
                var.current_value = 0.0
                var.dof_value = 0.0
            
            # Check if threshold exists
            if not hasattr(var, 'threshold') or var.threshold is None:
                print(f"Warning: {var_id} has no threshold, using 1.0")
                threshold = 1.0
            else:
                threshold = var.threshold
            
            print(f"Variable {var_id}: current_value={var.current_value}, threshold={threshold}, min_hard_limit={var.min_hard_limit}, max_hard_limit={var.max_hard_limit}")
            
            x0.append(var.current_value)
            # Bounds must be provided by BoundsBuilderSkill; do not modify here
            dyn = dynamic_bounds_map.get(var_id) or {}
            if 'min' not in dyn or 'max' not in dyn:
                raise RuntimeError(f"Missing dynamic bounds for '{var_id}'. Ensure BoundsBuilderSkill provides bounds for all optimizer inputs.")
            min_bound = float(dyn['min'])
            max_bound = float(dyn['max'])
            
            bounds.append((min_bound, max_bound))
            eps_values.append(0.01 * var.current_value)

        print("bounds: ", bounds)
        
        # Solver constraints built by the skill
        constraints_to_pass = []
        try:
            constraints_to_pass = context.get_solver_constraints() or []
        except Exception:
            constraints_to_pass = []

        # Run optimization based on method
        method = (self.algorithm or '').strip()
        result = None

        if method in {'SLSQP', 'COBYLA', 'trust-constr', 'COBYQA'}:
            # Methods with native constraint support
            minimize_kwargs = {
                'fun': objective,
                'x0': x0,
                'method': method,
                'bounds': bounds,
                'options': {
                    'maxiter': 1000,
                    'disp': False,
                }
            }
            if constraints_to_pass:
                minimize_kwargs['constraints'] = constraints_to_pass
            result = minimize(**minimize_kwargs)
        elif method == 'differential_evolution':
            # Global optimization with differential evolution
            from scipy.optimize import differential_evolution
            
            result = differential_evolution(
                func=objective,
                bounds=bounds,
                maxiter=100,
                popsize=15,
                tol=1e-6,
                polish=True,
                constraints=constraints_to_pass,
                seed=42
            )
        elif method == 'shgo':
            # Global optimization with SHGO
            from scipy.optimize import shgo
            
            result = shgo(
                func=objective,
                bounds=bounds,
                n=100,
                iters=3,
                sampling_method='sobol',
                constraints=constraints_to_pass
            )
        else:
            # Methods without constraint support - try with constraints first, fallback to bounds-only
            try:
                # First try with constraints (might fail for some methods)
                minimize_kwargs = {
                    'fun': objective,
                    'x0': x0,
                    'method': method,
                    'bounds': bounds,
                    'options': {
                        'maxiter': 1000,
                        'disp': False,
                    }
                }
                if constraints_to_pass:
                    minimize_kwargs['constraints'] = constraints_to_pass
                result = minimize(**minimize_kwargs)
            except Exception as e:
                print(f"Method {method} failed with constraints: {e}")
                print("Falling back to bounds-only optimization...")
                # Fallback: run without constraints, then refine with constraints
                result = minimize(fun=objective, x0=x0, method=method, bounds=bounds, 
                               options={'maxiter': 1000, 'disp': False})
        # Store the optimal values
        if result.success:
            try:
                print("Number of iterations: ", result.nit)
            except Exception:
                pass
            for var_id, value in zip(optimizable_inputs, result.x):
                var = context.get_variable(var_id)
                var.recommended_value = value
                var.dof_value = value  # Update DOF value to optimal
            print("cost_iterations: ", cost_iterations)
        else:
            raise RuntimeError(f"Optimization failed: {result.message}")