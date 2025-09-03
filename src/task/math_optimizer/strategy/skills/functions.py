from .base import Skill
import asteval

class MathFunction(Skill):
    """A Skill for evaluating safe mathematical and logical expressions."""
    def __init__(self, name, config):
        super().__init__(name, config)
        self.formula = config['config']['formula']
        self.df_formula = config['config']['dataframe_formula'] if 'dataframe_formula' in config['config'] else None
        self.aeval = asteval.Interpreter()

    def resolve_dataframe_formula(self, data_context):
        """
        Resolves a dataframe formula and adds the calculated variable(s) to the context.

        Parameters
        ----------
        data_context : object
            The context object holding the dataframe (with get_dataframe / set_dataframe methods).

        Notes
        -----
        - `self.inputs` is expected to be a list of input column names (raw variables of calculated variables).
        - `self.outputs` is expected to be a list with the output column name(s) (calculated variables).
        - `self.df_formula` should define the method to apply and its parameters.
        """
        df = data_context.get_dataframe()
        method = self.df_formula.get('method')
        axis = self.df_formula.get("axis")
        df[self.outputs[0]] = getattr(df[self.inputs], method)(axis=axis)
        data_context.set_dataframe(df)

    def execute(self, data_context) -> None:
        # Populate the symbol table for the expression evaluator
        for var_id in self.inputs:
            variable = data_context.get_variable(var_id)
            # Make both dof and current values available in the formula
            # e.g., 'kiln_feed' refers to 'kiln_feed_dof'
            # and 'kiln_feed_current' refers to 'kiln_feed_current'
            dof_value = variable.dof_value if variable.dof_value is not None else 0.0
            current_value = variable.current_value if variable.current_value is not None else 0.0
            
            # Debug: Print if we encounter None values
            if variable.dof_value is None or variable.current_value is None:
                print(f"Warning: {var_id} has None values - dof: {variable.dof_value}, current: {variable.current_value}")
            
            self.aeval.symtable[f"{var_id}_dof"] = dof_value
            self.aeval.symtable[f"{var_id}_current"] = current_value
            if hasattr(variable, 'threshold') and variable.threshold != 0.0:
                self.aeval.symtable[f"{var_id}_threshold"] = variable.threshold
            # Also make the base variable name available (for backward compatibility)
            self.aeval.symtable[var_id] = dof_value
        

        
        # Safely evaluate the expression
        try:
            result = self.aeval.eval(self.formula, show_errors=False)
        except Exception as e:
            print(f"Error in MathFunction {self.name}: {e}")
            print(f"Formula: {self.formula}")
            print(f"Symbol table keys: {list(self.aeval.symtable.keys())}")
            print(f"Symbol table values: {[(k, v) for k, v in self.aeval.symtable.items() if k in self.formula]}")
            result = 0.0
        
        # Handle None result
        if result is None:
            print(f"Warning: MathFunction {self.name} returned None for formula: {self.formula}")
            result = 0.0

        # Write the result to the output variable
        if self.outputs:
            output_variable_name = self.outputs[0]  # Get first output variable
            data_context.get_variable(output_variable_name).dof_value = result