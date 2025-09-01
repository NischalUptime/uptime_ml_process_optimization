# Mock model_trainer module to satisfy import requirements for saved models

class ANNModel:
    """
    Mock ANNModel class to satisfy import requirements.
    This is used when loading models that were saved with references to components.model_trainer.ANNModel
    """
    def __init__(self, *args, **kwargs):
        pass
    
    def __getattr__(self, name):
        # Return a dummy method for any attribute access
        return lambda *args, **kwargs: None
    
    def state_dict(self):
        # Return empty state dict if called
        return {}
    
    def eval(self):
        # Mock eval method
        pass
    
    def parameters(self):
        # Return empty parameters iterator
        return iter([])
