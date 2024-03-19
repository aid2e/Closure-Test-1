from ProjectUtils.config_editor import *
from ProjectUtils.mobo_utilities import *
from ProjectUtils.progress_trackers import *

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

from collections import defaultdict


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
    # Dynamically add stuff to config
    config['is_gpu'] = torch.cuda.is_available()

    if not os.getenv("WANDB_API_KEY") and not os.path.exists(args.secret_file):
        print("Please set WANDB_API_KEY in your environment variables or include a file named secrets.key in the "
              "same directory as this script.")
        sys.exit()
    os.environ["WANDB_API_KEY"] = read_json_file(args.secret_file)["WANDB_API_KEY"] if not os.getenv(
        "WANDB_API_KEY") else os.environ["WANDB_API_KEY"]

    tracker_group = TrackerGroup([TxtTracker(config)])
    if args.profile:
        tracker_group.append(CsvTracker(config))

    if config.get("WandB_params") and args.profile:
        tracker_group.append(WandBTracker(config))

    d = config["n_design_params"]
    M = config["n_objectives"]
    isGPU = torch.cuda.is_available()
    tkwargs = {
        "dtype": torch.double,
        "device": torch.device("cuda" if isGPU else "cpu"),
    }

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

    tracker_group.write_problem_summary(hv_pareto, hv_npoints, problem.ref_point)

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

    iter_res = defaultdict(list)
    model = None

    if not args.json_file:
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
        iter_res['is_converged'].append(((hv_pareto - hv) / hv_pareto) < config["hv_tolerance"])

        print(f"Initialized points, HV: {hv}")
        list_dump = {
            "experiment": experiment,
            "HV_PARETO": hv_pareto,
            "data": data,
            "outcomes": outcomes,
        }
        list_dump.update(iter_res)
        with open(os.path.join(config["OUTPUT_DIR"], "ax_state_init.json"), 'wb') as handle:
            pickle.dump(list_dump, handle, pickle.HIGHEST_PROTOCOL)
            print("saved initial generation file")
    if args.json_file:
        print("\n\n WARNING::YOU ARE LOADING AN EXISTING FILE: ", args.json_file, "\n\n")
        tmp_list = pickle.load(open(args.json_file, "rb"))
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
        tmp_tol = 1. if iter_res['hv'][-roll] == 0. else \
            abs((iter_res['hv'][-1] - iter_res['hv'][-roll]) / iter_res['hv'][-roll])

        # atleast 5% improvement w.r.t. last 5 calls and last call is better than first call
        check_imp = (tmp_tol > 0.0001) or (
                iter_res['hv'][-roll2] >= iter_res['hv'][-1])  # or (abs((hv_list[-1] - hv_list[1])/hv_list[1]) < 0.01)

    tracker_group.log_iter_results(iter_res)

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
        iter_res['is_converged'].append(((hv_pareto - hv) / hv_pareto) < tol)
        iter_res['hv'].append(hv)
        if len(iter_res['hv']) > roll:
            tmp_tol = 1. if (iter_res['hv'][-roll] == 0.) else \
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

        tracker_group.log_iter_results(iter_res)

    tracker_group.finalize()
