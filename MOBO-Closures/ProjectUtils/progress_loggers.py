import wandb
import os
import torch
import datetime
import sys
import pandas as pd
from abc import ABC
from textwrap import dedent

from . import config_editor as ce


class Tracker(ABC):
    def __init__(self, conf):
        self.save_every_n = conf["save_every_n_call"]

    def write_problem_summary(self, hv_pareto, hv_npoints, ref_point):
        pass

    def log_iter_results(self, res):
        if not res['last_call'][-1] % self.save_every_n == 0:
            return
        self._log_iter_results(res)

    def _log_iter_results(self, res):
        pass

    def finalize(self):
        pass


class LocalTracker(Tracker):
    def __init__(self, conf):
        super().__init__(conf)
        self.out_dir = conf["OUTPUT_DIR"]
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)


class CsvTracker(LocalTracker):
    def __init__(self, conf):
        super().__init__(conf)
        self.full_path = os.path.join(self.out_dir, "profile_data.csv")

    def _log_iter_results(self, res):
        pd.DataFrame(res).to_csv(self.full_path)


class TxtTracker(LocalTracker):
    def __init__(self, conf):
        super().__init__(conf)
        self.full_path = os.path.join(self.out_dir, "optimInfo.txt")
        optimization_info = dedent(f"""
            Optimization Info with name: {conf["name"]}
            Optimization has {conf["n_objectives"]} objectives
            Optimization has {conf["n_design_params"]} design parameters
            Optimization Info with description: {conf["description"]}
            Starting optimization at {datetime.datetime.now()}
            Optimization is running on {os.uname().nodename}
            Optimization description: {conf["description"]}
        """)
        if torch.cuda.is_available():
            optimization_info += f"Optimization is running on GPU: {torch.cuda.get_device_name()}\n"
        with open(self.full_path, "w") as f:
            f.write(optimization_info)

    def write_problem_summary(self, hv_pareto, hv_npoints, ref_point):
        with open(self.full_path, "a") as f:
            f.write(dedent(f"""\
            Problem Reference points: {ref_point}
            Problem Pareto Front Hypervolume: {hv_pareto}
            Problem Random Points Hypervolume: {hv_npoints}"""))

    def _log_iter_results(self, res):
        with open(self.full_path, "a") as f:
            f.write(dedent(f"""\
            Optimization call: {res['last_call'][-1]}
            Optimization HV: {res['hv'][-1]}
            Optimization Pareto HV - HV / Pareto HV: {res['converged'][-1]:.4f}
            Optimization converged: {res['is_converged'][-1]}"""))


class WandBTracker(Tracker):
    def __init__(self, conf):
        super().__init__(conf)
        wandb.login(anonymous='never', key=os.environ['WANDB_API_KEY'], relogin=True)
        track_conf = {k: conf[k] for k in ["n_design_params", "n_objectives"]}
        self._tracker = wandb.init(config=track_conf, **conf["WandB_params"])
        num_samples = 64 if (not conf.get("MOBO_params")) else conf["MOBO_params"]["num_samples"]
        warmup_steps = 128 if (not conf.get("MOBO_params")) else conf["MOBO_params"]["warmup_steps"]
        self._tracker.config.update({
            "BATCH_SIZE": conf["n_batch"],
            "N_BATCH": conf["n_calls"],
            "num_samples": num_samples,
            "warmup_steps": warmup_steps
        })
        self._tracker.define_metric("iterations")
        log_metrics = ["MCMC Training [s]",
                       f"Gen Acq func (q = {conf['n_batch']}) [s]",
                       f"Trail Exec (q = {conf['n_batch']}) [s]",
                       "HV",
                       "Increase in HV w.r.t true pareto",
                       "HV Calculation [s]",
                       "Total time [s]"]
        for lm in log_metrics:
            self._tracker.define_metric(lm, step_metric="iterations")

    def write_problem_summary(self, hv_pareto, hv_npoints, ref_point):
        self._tracker.summary.update({
            "HV": hv_pareto,
            "HV_RandomPoints": hv_npoints,
            "ref_point": str(ref_point.tolist())
        })

    def _log_iter_results(self, res):
        bs = self._tracker.config['BATCH_SIZE']
        self._tracker.log({
            "MCMC Training [s]": res['time_mcmc'][-1],
            f"Gen Acq func (q = {bs}) [s]": res['time_gen'][-1],
            f"Trail Exec (q = {bs}) [s]": res['time_trail'][-1],
            "HV": res['hv'][-1],
            "Increase in HV w.r.t true pareto": res['converged'][-1],
            "HV Calculation [s]": res['time_hv'][-1],
            "Total time [s]": res['time_tot'][-1],
            "iterations": res['last_call'][-1]
        })

    def finalize(self):
        self._tracker.finish()
