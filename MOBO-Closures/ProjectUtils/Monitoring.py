import wandb, os
class WandbLogger:
    def __init__(self, project_name = None, run_name = None, config = None, reinit = False):
        self.project_name = project_name
        self.run_name = run_name
        self.config = config
        self.reinit = reinit
        self.wandb_run = self.get_WandB()
        self.metrics = {}
    def checklogin(self):
        status = False
        if (not os.environ["WANDB_API_KEY"]):
            return status
        else:
            status = wandb.login(anonymous='never', key = os.environ['WANDB_API_KEY'])
            return status
    def setProjectName(self, project_name: str):
        self.project_name = project_name
    def setRunName(self, run_name: str):
        self.run_name = run_name
    def setConfig(self, config: dict):
        self.config = config
    def setReInit(self, reinit: bool):
        self.reinit = reinit
    def addMetric(self, metric_name, metric_value):
        if(self.metrics.get(metric_name)):
            self.metrics[metric_name].append(metric_value)
        self.metrics[metric_name] = metric_value
    def exit(self):
        wandb.finish()
    def log(self, metrics):
        self.wandb_run.log(metrics)