from ProjectUtils.config_editor import *
from ProjectUtils.mobo_utilities import *

import os, pickle, torch, argparse, datetime
import time

import pandas as pd
from ax import *

from ax.metrics.noisy_function import GenericNoisyFunctionMetric
from ax.service.utils.report_utils import exp_to_df

# Model registry for creating multi-objective optimization models.
from ax.modelbridge.registry import Models

from botorch.utils.multi_objective.box_decompositions.dominated import (
    DominatedPartitioning,
)

from botorch.test_functions.multi_objective import DTLZ2

import wandb

from collections import defaultdict
from copy import deepcopy


def RunProblem(problem, x, kwargs):
    return problem(torch.tensor(x, **kwargs).clamp(0.0, 1.0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimization Closure Test-1")
    parser.add_argument('-c', '--config',
                        help='Optimization configuration file',
                        type=str, required=True)
    parser.add_argument('-j', '--json_file',
                        help="The json file to load and continue optimization",
                        type=str, required=False)
    parser.add_argument('-s', '--secret_file',
                        help="The file containing the secret key for weights and biases",
                        type=str, required=False,
                        default="secrets.key")
    parser.add_argument('-p', '--profile',
                        help="Profile the code",
                        type=bool, required=False,
                        default=False)
    args = parser.parse_args()

    # READ SOME INFO 
    config = read_json_file(args.config)
    jsonFile = args.json_file
    profiler = args.profile
    outdir = config["OUTPUT_DIR"]
    save_every_n = config["save_every_n_call"]
    doMonitor = (True if config.get("WandB_params") else False) and profiler
    MLTracker = None
    if doMonitor:
        if not os.getenv("WANDB_API_KEY") and not os.path.exists(args.secret_file):
            print("Please set WANDB_API_KEY in your environment variables or include a file named secrets.key in the "
                  "same directory as this script.")
            sys.exit()
        else:
            os.environ["WANDB_API_KEY"] = read_json_file(args.secret_file)["WANDB_API_KEY"] if not os.getenv(
                "WANDB_API_KEY") else os.environ["WANDB_API_KEY"]
            wandb.login(anonymous='never', key=os.environ['WANDB_API_KEY'], relogin=True)
            track_config = {"n_design_params": config["n_design_params"], "n_objectives": config["n_objectives"]}
            MLTracker = wandb.init(config=track_config, **config["WandB_params"])

    optimInfo = "optimInfo.txt" if not jsonFile else "optimInfo_continued.txt"
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    d = config["n_design_params"]
    M = config["n_objectives"]
    isGPU = torch.cuda.is_available()
    tkwargs = {
        "dtype": torch.double,
        "device": torch.device("cuda" if isGPU else "cpu"),
    }
    optimization_info = f"""
        Optimization Info with name: {config["name"]}
        Optimization has {config["n_objectives"]} objectives
        Optimization has {config["n_design_params"]} design parameters
        Optimization Info with description: {config["description"]}
        Starting optimization at {datetime.datetime.now()}
        Optimization is running on {os.uname().nodename}
        Optimization description: {config["description"]}
    """
    if torch.cuda.is_available():
        optimization_info += f"Optimization is running on GPU: {torch.cuda.get_device_name()}\n"
    with open(os.path.join(outdir, optimInfo), "w") as f:
        f.write(optimization_info)

    print("Running on GPU? ", isGPU)

    problem = DTLZ2(dim=d, num_objectives=M, negate=True).to(**tkwargs)

    problem.ref_point = torch.tensor([-max(1.1, d / 10.) for _ in range(M)], **tkwargs)

    print("Problem Reference points : ", problem.ref_point)

    NPoints = 10000
    pareto_fronts = problem.gen_pareto_front(NPoints // 10)
    hv_pareto = DominatedPartitioning(ref_point=problem.ref_point,
                                      Y=pareto_fronts
                                      ).compute_hypervolume().item()
    print(f"Pareto Front Hypervolume: {hv_pareto}")
    n_points = problem(torch.rand(NPoints * d, **tkwargs).reshape(NPoints, d))
    hv_npoints = DominatedPartitioning(ref_point=problem.ref_point,
                                       Y=n_points
                                       ).compute_hypervolume().item()
    print(f"Random Points Hypervolume: {hv_npoints}")

    if doMonitor:
        MLTracker.summary["HV"] = hv_pareto
        MLTracker.summary["HV_RandomPoints"] = hv_npoints
        MLTracker.summary["ref_point"] = str(problem.ref_point.tolist())

    with open(os.path.join(outdir, optimInfo), "a") as f:
        f.write(f"""Problem Reference points: {problem.ref_point}
                    Problem Pareto Front Hypervolume: {hv_pareto}
                    Problem Random Points Hypervolume: {hv_npoints}""")

    @glob_fun
    def ftot(x):
        return RunProblem(problem, x, tkwargs)


    def f1(xdic):
        x = tuple(xdic[k] for k in xdic.keys())
        return float(ftot(x)[0])


    def f2(xdic):
        x = tuple(xdic[k] for k in xdic.keys())
        return float(ftot(x)[1])


    def f3(xdic):
        x = tuple(xdic[k] for k in xdic.keys())
        return float(ftot(x)[2])


    def f4(xdic):
        x = tuple(xdic[k] for k in xdic.keys())
        return float(ftot(x)[3])


    search_space = SearchSpace(
        parameters=[
            RangeParameter(name=f"x{i}",
                           lower=0, upper=1,
                           parameter_type=ParameterType.FLOAT)
            for i in range(d)],
    )
    param_names = [f"x{i}" for i in range(d)]

    names = ["a", "b", "c", "d"]
    functions = [f1, f2, f3, f4]
    metrics = []

    for name, function in zip(names[:M], functions[:M]):
        metrics.append(
            GenericNoisyFunctionMetric(
                name=name, f=function, noise_sd=0.0, lower_is_better=False
            )
        )
    mo = MultiObjective(
        objectives=[Objective(m) for m in metrics],
    )
    objective_thresholds = [
        ObjectiveThreshold(metric=metric, bound=val, relative=False)
        for metric, val in zip(mo.metrics, problem.ref_point.to(tkwargs["device"]))
    ]
    optimization_config = MultiObjectiveOptimizationConfig(objective=mo,
                                                           objective_thresholds=objective_thresholds, )
    N_INIT = max(config["n_initial_points"], M * (d + 1))
    BATCH_SIZE = config["n_batch"]
    N_BATCH = config["n_calls"]
    num_samples = 64 if (not config.get("MOBO_params")) else config["MOBO_params"]["num_samples"]
    warmup_steps = 128 if (not config.get("MOBO_params")) else config["MOBO_params"]["warmup_steps"]
    if doMonitor:
        MLTracker.config["BATCH_SIZE"] = BATCH_SIZE
        MLTracker.config["N_BATCH"] = N_BATCH
        MLTracker.config["num_samples"] = num_samples
        MLTracker.config["warmup_steps"] = warmup_steps
        MLTracker.define_metric("iterations")
        logMetrics = ["MCMC Training [s]",
                      f"Gen Acq func (q = {BATCH_SIZE}) [s]",
                      f"Trail Exec (q = {BATCH_SIZE}) [s]",
                      "HV",
                      "Increase in HV w.r.t true pareto",
                      "HV Calculation [s]",
                      "Total time [s]"]
        for l in logMetrics:
            MLTracker.define_metric(l, step_metric="iterations")

    iter_res = defaultdict(list)
    model = None

    if not jsonFile:
        start_tot = time.time()
        experiment = build_experiment(search_space, optimization_config)
        start_gen = time.time()
        data = initialize_experiment(experiment, N_INIT)
        end_gen = time.time()
        exp_df = exp_to_df(experiment)
        outcomes = torch.tensor(exp_df[names[:M]].values, **tkwargs)
        start_hv = time.time()
        partitioning = DominatedPartitioning(ref_point=problem.ref_point, Y=outcomes)
        try:
            hv = partitioning.compute_hypervolume().item()
        except:
            hv = 0.
        end_hv = time.time()
        end_tot = time.time()
        iter_res['time_tot'].append(end_tot - start_tot)
        iter_res['time_gen'].append(end_gen - start_gen)
        iter_res['time_hv'].append(end_hv - start_hv)
        iter_res['time_mcmc'].append(-1.)
        iter_res['time_trail'].append(-1.)
        iter_res['hv'].append(hv)
        iter_res['last_call'].append(0)
        iter_res['converged'].append((hv_pareto - hv) / hv_pareto)

        print(f"Initialized points, HV: {hv}")
        list_dump = {
            "experiment": experiment,
            "HV_PARETO": hv_pareto,
            "data": data,
            "outcomes": outcomes,
        }
        list_dump.update(iter_res)
        with open(os.path.join(outdir, "ax_state_init.json"), 'wb') as handle:
            pickle.dump(list_dump, handle, pickle.HIGHEST_PROTOCOL)
            print("saved initial generation file")
    if jsonFile:
        print("\n\n WARNING::YOU ARE LOADING AN EXISTING FILE: ", jsonFile, "\n\n")
        tmp_list = pickle.load(open(jsonFile, "rb"))
        experiment = tmp_list.pop('experiment')
        hv_pareto = tmp_list.pop('HV_PARETO')
        data = tmp_list.pop('data')
        outcomes = tmp_list.pop('outcomes')
        iter_res.update(tmp_list)

    tol = config["hv_tolerance"]
    max_calls = config["max_calls"]
    check_imp = True
    roll = 30
    roll2 = min(len(iter_res['hv']) - 1, 2 * roll)
    if len(iter_res['hv']) > roll:
        tmp_tol = 1. if iter_res['hv'][-roll] == 0. else\
            abs((iter_res['hv'][-1] - iter_res['hv'][-roll]) / iter_res['hv'][-roll])

        # atleast 5% improvement w.r.t. last 5 calls and last call is better than first call
        check_imp = (tmp_tol > 0.0001) or (
                iter_res['hv'][-roll2] >= iter_res['hv'][-1])  # or (abs((hv_list[-1] - hv_list[1])/hv_list[1]) < 0.01)

    if profiler:
        pd.DataFrame(iter_res).to_csv(os.path.join(outdir, "profile_data.csv"))
    if doMonitor and jsonFile:
        logMetrics = {f"Trail Exec (q = {BATCH_SIZE}) [s]": iter_res['time_trail'][-1],
                      "HV": iter_res['hv'][-1],
                      "Increase in HV w.r.t true pareto": iter_res['converged'][-1],
                      "HV Calculation [s]": iter_res['time_hv'][-1],
                      "Total time [s]": iter_res['time_tot'][-1],
                      "iterations": iter_res['last_call'][-1]
                      }
        MLTracker.log(logMetrics)
    while iter_res['converged'][-1] > tol and iter_res['last_call'][-1] <= max_calls and check_imp:
        start_tot = time.time()
        start_mcmc = time.time()
        model = Models.FULLYBAYESIANMOO(
            experiment=experiment,
            data=data,
            num_samples=num_samples,
            warmup_steps=warmup_steps,
            torch_device=tkwargs["device"],
            torch_dtype=tkwargs["dtype"],
            verbose=False,  # Set to True to print stats from MCMC
            disable_progbar=False,  # Set to False to print a progress bar from MCMC
        )
        end_mcmc = time.time()
        start_gen = time.time()
        generator_run = model.gen(BATCH_SIZE)
        end_gen = time.time()
        start_trail = time.time()
        trial = experiment.new_batch_trial(generator_run=generator_run)
        trial.run()
        end_trail = time.time()
        data = Data.from_multiple_data([data, trial.fetch_data()])
        exp_df = exp_to_df(experiment)
        outcomes = torch.tensor(exp_df[names[:M]].values, **tkwargs)
        start_hv = time.time()
        partitioning = DominatedPartitioning(ref_point=problem.ref_point, Y=outcomes)
        try:
            hv = partitioning.compute_hypervolume().item()
        except:
            hv = 0.

        end_hv = time.time()
        end_tot = time.time()

        iter_res['last_call'].append(iter_res['last_call'][-1] + 1)
        iter_res['converged'].append((hv_pareto - hv) / hv_pareto)
        iter_res['hv'].append(hv)
        if len(iter_res['hv']) > roll:
            tmp_tol = 1. if (iter_res['hv'][-roll] == 0.) else\
                abs((iter_res['hv'][-1] - iter_res['hv'][-roll]) / iter_res['hv'][-roll])
            # atleast 5% improvement w.r.t. last #roll calls and last call is better than first call
            check_imp = (tmp_tol > 0.0001) or (
                    iter_res['hv'][-roll2] >= iter_res['hv'][-1])  # or (abs((hv_list[-1] - hv_list[1])/hv_list[-1]) < 0.01)
        iter_res['time_tot'].append(end_tot - start_tot)
        iter_res['time_mcmc'].append(end_mcmc - start_mcmc)
        iter_res['time_gen'].append(end_gen - start_gen)
        iter_res['time_trail'].append(end_trail - start_trail)
        iter_res['time_hv'].append(end_hv - start_hv)
        roll2 += 1

        with open(os.path.join(outdir, optimInfo), "a") as f:
            f.write(f"""Optimization call: {iter_res['last_call'][-1]}
                        Optimization HV: {iter_res['hv'][-1]}
                        Optimization Pareto HV - HV / Pareto HV: {iter_res['converged'][-1]:.4f}
                        Optimization converged: {iter_res['converged'][-1] < tol}""")

        if iter_res['last_call'][-1] % save_every_n == 0:
            list_dump = {
                "experiment": experiment,
                "HV_PARETO": hv_pareto,
                "data": data,
                "outcomes": outcomes,
            }
            list_dump.update(iter_res)
            with open(os.path.join(outdir, f"optim_iteration_{iter_res['last_call'][-1]}.json"), 'wb') as handle:
                pickle.dump(list_dump, handle)
                print(f"saved the file for {iter_res['last_call'][-1]} iteration")
            if profiler:
                pd.DataFrame(iter_res).to_csv(os.path.join(outdir, "profile_data.csv"))
            if doMonitor:
                logMetrics = {"MCMC Training [s]": iter_res['time_mcmc'][-1],
                              f"Gen Acq func (q = {BATCH_SIZE}) [s]": iter_res['time_gen'][-1],
                              f"Trail Exec (q = {BATCH_SIZE}) [s]": iter_res['time_trail'][-1],
                              "HV": iter_res['hv'][-1],
                              "Increase in HV w.r.t true pareto": iter_res['converged'][-1],
                              "HV Calculation [s]": iter_res['time_hv'][-1],
                              "Total time [s]": iter_res['time_tot'][-1],
                              "iterations": iter_res['last_call'][-1]
                              }
                MLTracker.log(logMetrics)
    if doMonitor:
        MLTracker.finish()
