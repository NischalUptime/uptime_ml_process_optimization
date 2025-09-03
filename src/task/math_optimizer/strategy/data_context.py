from.variable import Variable

class DataContext:
    """
    A transient, in-memory container holding the state of all Variable
    objects for a single execution cycle.
    """
    def __init__(self, variables_config, weights_config=None):
        self._variables = {
            var_id: Variable(var_id, config)
            for var_id, config in variables_config.items()
        }
        self._df = None  # DataFrame to hold time series data if needed
        self.weights = weights_config or {}
        self._dynamic_bounds = {}
        self._solver_constraints = []
    def get_variable(self, var_id):
        if var_id not in self._variables:
            raise KeyError(f"Variable '{var_id}' not found in DataContext.")
        return self._variables[var_id]

    def has_variable(self, var_id):
        return var_id in self._variables

    def populate_initial_data(self, records):
        """
        Populates the context with live data at the start of a cycle.

        Args:
        records (list | dict):
            - A list of {"timestamp": ..., "data": {...}} dictionaries
            - OR a single {"var1": ..., "var2": ...} dict (backward compatibility)
        """
        import pandas as pd

        if not records:
            raise ValueError("No data provided to populate_initial_data")

        # Normalize input into a list of {"timestamp": ..., "data": {...}}
        if isinstance(records, dict) and "data" not in records:
            records = [{"timestamp": None, "data": records}]
        elif isinstance(records, dict):
            records = [records]

        # Store full dataset in a DataFrame
        self._df = pd.DataFrame(
            [{"timestamp": row.get("timestamp"), **row.get("data", {})} for row in records]
        )

        # Use the last row’s data for initializing variables
        last_row = records[-1].get("data", {})

        # 1. Assign provided values to known variables
        for var_id, value in last_row.items():
            if self.has_variable(var_id):
                self.get_variable(var_id).set_initial_value(value)
        
        # Initialize all other variables with default values to prevent None errors
        for var_id, variable in self._variables.items():
            if variable.current_value is None:
                # Set default values based on variable type
                if variable.var_type in ['Delta', 'Predicted', 'Constraint', 'CalculatedKPI']:
                    variable.current_value = 0.0
                    variable.dof_value = 0.0
                    variable.recommended_value = 0.0
                elif variable.var_type == 'Calculated':
                    # For calculated variables, we'll set them after pre-calculation
                    variable.current_value = 0.0
                    variable.dof_value = 0.0
                    variable.recommended_value = 0.0

    def get_all_variables(self):
        return self._variables

    def set_dataframe(self, df):
        """Set the DataFrame for the context."""
        self._df = df

    def get_dataframe(self):
        """Return the full dataframe stored in context."""
        return self._df

    def set_dynamic_bounds(self, bounds_map):
        """
        Set dynamic bounds for variables.
        bounds_map format: { var_id: { 'min': float, 'max': float }, ... }
        """
        self._dynamic_bounds = bounds_map or {}

    def get_dynamic_bounds(self):
        """Get dynamic bounds map if set, else empty dict."""
        return self._dynamic_bounds

    def set_solver_constraints(self, constraints):
        """
        Store pre-built solver constraints (e.g., SciPy NonlinearConstraint objects).
        """
        self._solver_constraints = constraints or []

    def get_solver_constraints(self):
        """Retrieve solver constraints list if any."""
        return self._solver_constraints