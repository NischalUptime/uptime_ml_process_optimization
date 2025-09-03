from .base import Skill


class BoundsBuilderSkill(Skill):
    """
    Builds dynamic bounds and stores them in the DataContext.
    Always uses 'window' behavior: for each input variable, sets
    [current - threshold, current + threshold] clamped to hard limits.
    """

    def __init__(self, name, config):
        super().__init__(name, config)

    def execute(self, data_context) -> None:
        bounds_map = {}
        candidate_vars = self.inputs or []

        for v in candidate_vars:
            try:
                var = data_context.get_variable(v)
            except Exception:
                continue
            threshold = getattr(var, 'threshold', 0.0) or 0.0
            current = var.current_value if var.current_value is not None else 0.0
            mn = max(var.min_hard_limit, current - threshold)
            mx = min(var.max_hard_limit, current + threshold)
            # Ensure ordering; if invalid, fallback to hard limits
            if mn > mx:
                mn, mx = min(var.min_hard_limit, var.max_hard_limit), max(var.min_hard_limit, var.max_hard_limit)
            bounds_map[v] = {
                'min': float(mn),
                'max': float(mx),
            }

        data_context.set_dynamic_bounds(bounds_map)


