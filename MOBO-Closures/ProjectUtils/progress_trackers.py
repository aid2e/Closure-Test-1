import wandb
import os
import torch
import datetime
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

    def log_iter_results(self, res):
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

    def log_iter_results(self, res):
        with open(self.full_path, "a") as f:
            f.write(dedent(f"""\
            Optimization call: {res['last_call'][-1]}
            Optimization HV: {res['hv'][-1]}
            Optimization Pareto HV - HV / Pareto HV: {res['converged'][-1]:.4f}
            Optimization converged: {res['is_converged'][-1]}"""))


class WandBTracker(Tracker):
    def __init__(self, conf):
        super().__init__(conf)
        if not os.getenv("WANDB_API_KEY") and not os.path.exists(conf['secret_file']):
            print("Please set WANDB_API_KEY in your environment variables or include a file named secrets.key in the "
                  "same directory as this script.")
            exit(1)
        api_key = os.environ.get('WANDB_API_KEY', ce.read_json_file(conf['secret_file'])['WANDB_API_KEY'])
        wandb.login(anonymous='never', key=api_key, relogin=True)
        num_samples = 64 if (not conf.get("MOBO_params")) else conf["MOBO_params"]["num_samples"]
        warmup_steps = 128 if (not conf.get("MOBO_params")) else conf["MOBO_params"]["warmup_steps"]
        tracker_conf = {
            "n_design_params": conf["n_design_params"],
            "n_objectives": conf["n_objectives"],
            "BATCH_SIZE": conf["n_batch"],
            "N_BATCH": conf["n_calls"],
            "num_samples": num_samples,
            "warmup_steps": warmup_steps
        }
        self._tracker = wandb.init(config=tracker_conf, **conf["WandB_params"])
        self._tracker.define_metric("iterations")
        self.metrics_dict = {
            "MCMC Training [s]": 'time_mcmc',
            f"Gen Acq func (q = {conf['n_batch']}) [s]": 'time_gen',
            f"Trail Exec (q = {conf['n_batch']}) [s]": 'time_trial',
            "HV": 'hv',
            "Increase in HV w.r.t true pareto": 'converged',
            "HV Calculation [s]": 'time_hv',
            "Total time [s]": 'time_tot'
        }
        for m in self.metrics_dict.keys():
            self._tracker.define_metric(m, step_metric="iterations")

    def write_problem_summary(self, hv_pareto, hv_npoints, ref_point):
        self._tracker.summary.update({
            "HV": hv_pareto,
            "HV_RandomPoints": hv_npoints,
            "ref_point": str(ref_point.tolist())
        })

    def log_iter_results(self, res):
        log_dict = {k: res[v][-1] for k, v in self.metrics_dict.items()}
        log_dict['iterations'] = res['last_call'][-1]
        self._tracker.log(log_dict)

    def finalize(self):
        self._tracker.finish()


class TrackerGroup:
    def __init__(self, trackers=None):
        self.trackers = trackers

    def append(self, item):
        self.trackers.append(item)

    def __iter__(self):
        return iter(self.trackers)

    def write_problem_summary(self, hv_pareto, hv_npoints, ref_point):
        for t in self.trackers:
            t.write_problem_summary(hv_pareto, hv_npoints, ref_point)

    def log_iter_results(self, res):
        for t in self.trackers:
            if res['last_call'][-1] % t.save_every_n == 0:
                t.log_iter_results(res)

    def finalize(self):
        for t in self.trackers:
            t.finalize()
