class Variable:
    """
    Represents a process variable with its current and recommended values.
    """
    def __init__(self, var_id, config):
        self.var_id = var_id
        self.var_type = config.get('type', 'Unknown')
        self.units = config.get('units', '')
        
        # Handle physical limits with defaults
        self.threshold = config.get('threshold', 0.0)
        self.min_hard_limit = config.get('min_hard_limit', 0.0)
        self.max_hard_limit = config.get('max_hard_limit', 1000.0)

        # Initialize values
        self.current_value = None  # Set by populate_initial_data
        self.dof_value = None      # Used during optimization
        self.recommended_value = None  # Final recommendation

    def __repr__(self):
        current_str = f"{self.current_value:.2f}" if self.current_value is not None else "None"
        dof_str = f"{self.dof_value:.2f}" if self.dof_value is not None else "None"
        rec_str = f"{self.recommended_value:.2f}" if self.recommended_value is not None else "None"
        return (f"Variable(id={self.var_id}, "
                f"current={current_str}, "
                f"dof={dof_str}, "
                f"rec={rec_str}, "
                f"min_limit={self.min_hard_limit:.2f}, "
                f"max_limit={self.max_hard_limit:.2f})")

    def set_initial_value(self, value):
        """Sets the initial state for the optimization cycle."""
        self.current_value = float(value)
        self.dof_value = float(value)
        self.recommended_value = float(value)