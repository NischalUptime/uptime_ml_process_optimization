import ray
import threading
import structlog
from service.optimization import OptimizationService  # from your code

@ray.remote
class OptimizationActor:
    def __init__(self, configuration: dict):
        self.shutdown_event = threading.Event()
        self.configuration = configuration
        self.log = structlog.get_logger()
        self.optimization_service = None

    def start(self):
        try:
            self.log.info("Starting continuous optimization service (Ray actor)...")
            self.optimization_service = OptimizationService(self.shutdown_event, self.configuration)
            self.optimization_service.run_continuous()  # blocks until shutdown_event is set
        except Exception as e:
            import traceback
            self.log.error(f"Error in continuous optimization: {e}")
            self.log.error(traceback.format_exc())

    def stop(self):
        self.log.info("Stopping optimization service...")
        self.shutdown_event.set()
        return "Stopped"
