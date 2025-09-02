from .base import Skill
from .models import InferenceModel
import structlog
import concurrent.futures

class CompositionSkill(Skill):
    """
    A skill that executes a sequence of other skills in order.
    """
    def __init__(self, name, config):
        super().__init__(name, config)
        self.skill_sequence_names = config['config'].get('skill_sequence', [])
        self.skill_sequence = []  # Will be populated by resolve_skills
        self.logger = structlog.get_logger("process_optimization.composition")
        self.max_workers = 4  # Number of parallel workers for inference models
        self.logger.debug(f"CompositionSkill initialized with name: {name}")

    def resolve_skills(self, skill_registry):
        """
        Resolves skill names to actual skill objects.
        Called after all skills are instantiated.
        """
        self.skill_sequence = []
        for skill_name in self.skill_sequence_names:
            if skill_name not in skill_registry:
                raise ValueError(f"Skill '{skill_name}' not found in registry")
            self.skill_sequence.append(skill_registry[skill_name])

    def execute(self, context):
        """
        Executes each skill in the sequence in order, with parallel execution for inference models.
        """
        self.logger.debug(f"\nExecuting composition skill: {self.name}")
        
        i = 0
        while i < len(self.skill_sequence):
            skill = self.skill_sequence[i]
            
            # For non-inference models, execute sequentially
            if not isinstance(skill, InferenceModel):
                self.logger.debug(f"Executing sequential skill: {skill.name}")
                skill.execute(context)
                i += 1
                continue
            
            # For inference models, collect consecutive ones
            inference_group = []
            while i < len(self.skill_sequence) and isinstance(self.skill_sequence[i], InferenceModel):
                inference_group.append(self.skill_sequence[i])
                i += 1
            
            # Execute inference models in parallel
            if inference_group:
                self.logger.debug(f"Executing inference models in parallel: {[s.name for s in inference_group]}")
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    futures = []
                    for inf_model in inference_group:
                        future = executor.submit(inf_model.execute, context)
                        futures.append((inf_model.name, future))
                    
                    # Wait for all to complete
                    for name, future in futures:
                        try:
                            future.result()
                        except Exception as e:
                            self.logger.error(f"Error in inference model {name}: {e}")
                            raise