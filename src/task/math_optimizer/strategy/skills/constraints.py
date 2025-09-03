from .base import Skill
from scipy.optimize import NonlinearConstraint as SciPyNonlinearConstraint
import numpy as np


class Constraints(Skill):
    """
    Builds dynamic hard constraints and stores them in the DataContext.

    Config options inside config:
      - predicted_var: str or list[str]  # variable ids for predicted constraints
      - op_min: float or dict[var->float]  # lower operational bound(s)
      - op_max: float or dict[var->float]  # upper operational bound(s)
      - template: list[dict]              # alternatively, a full list of constraint dicts

    Inputs can be used to compute bounds via MathFunction-style formulas upstream.
    This skill only assembles and writes constraints to the context.
    """

    def __init__(self, name, config):
        super().__init__(name, config)
        self._strategy = None  # optional
        # Strict: only template supported
        self.template = self.config.get('template')

    def execute(self, data_context) -> None:
        # Strict: require non-empty template list
        if not isinstance(self.template, list) or not self.template:
            data_context.set_solver_constraints([])
            return
        # Validate entries
        for idx, c in enumerate(self.template):
            if 'predicted_var' not in c or 'op_min' not in c or 'op_max' not in c:
                raise ValueError(f"Constraints: template[{idx}] missing required keys: 'predicted_var', 'op_min', 'op_max'")

        # Build solver constraints directly from template
        solver_constraints = []
        if self._strategy is not None:
            # We need the optimization inputs and cost skill to evaluate the predicted variable
            # Find the first OptimizationSkill to get its inputs and cost skill name
            optimizer_skills = [s for s in self._strategy._skills.values() if s.__class__.__name__ == 'OptimizationSkill']
            if optimizer_skills:
                opt = optimizer_skills[0]
                cost_skill = self._strategy._skills.get(opt.cost_skill_name)
                optimizable_inputs = self._strategy.get_optimizable_variable_ids()
                opt_inputs = [var_id for var_id in opt.inputs if var_id in optimizable_inputs]

                def make_constraint_fun(op_min, op_max, pred_var_id):
                    def fun(x):
                        # Update context with current x values
                        for var_id, value in zip(opt_inputs, x):
                            data_context.get_variable(var_id).dof_value = value
                        # Run cost skill to produce prediction
                        cost_skill.execute(data_context)
                        # Get predicted value
                        pred_var = data_context.get_variable(pred_var_id)
                        v = pred_var.dof_value if pred_var and pred_var.dof_value is not None else 0.0
                        return np.array([
                            v - op_min,  # >= 0
                            op_max - v   # >= 0
                        ])
                    return fun

                for idx, c in enumerate(self.template):
                    pred_var_id = c['predicted_var']
                    op_min = float(c['op_min'])
                    op_max = float(c['op_max'])
                    solver_constraints.append(
                        SciPyNonlinearConstraint(
                            fun=make_constraint_fun(op_min, op_max, pred_var_id),
                            lb=np.array([0.0, 0.0]),
                            ub=np.array([np.inf, np.inf])
                        )
                    )

        data_context.set_solver_constraints(solver_constraints)

    # Optional hook to attach strategy without changing base class
    def set_strategy(self, strategy):
        self._strategy = strategy

