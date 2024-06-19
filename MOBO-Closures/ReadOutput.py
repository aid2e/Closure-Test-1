from ProjectUtils.config_editor import *
from ProjectUtils.config_editor import *
from ProjectUtils.mobo_utilities import *

import os, pickle, torch, argparse, datetime, glob
import time

import pandas as pd
from ax import *

import numpy as np

from ax.metrics.noisy_function import GenericNoisyFunctionMetric
from ax.service.utils.report_utils import exp_to_df

# Model registry for creating multi-objective optimization models.
from ax.modelbridge.registry import Models

from ax.core.metric import Metric
from botorch.utils.multi_objective.box_decompositions.dominated import (
    DominatedPartitioning,
)

from botorch.test_functions.multi_objective import DTLZ2

import matplotlib.pyplot as plt

def RunProblem(problem, x, kwargs):
    return problem(torch.tensor(x, **kwargs).clamp(0.0, 1.0))

def ReadAndWriteOutputs(jsonFile: str, outFileName: str, M: int, d: int):
    isGPU = torch.cuda.is_available()
    tkwargs = {
        "dtype": torch.double, 
        "device": torch.device("cuda" if isGPU else "cpu"),
    }
    problem = DTLZ2(dim=d, num_objectives=M, negate=True).to(**tkwargs)
    problem.ref_point = torch.tensor([-max(1.1, d/10.) for _ in range(M)], **tkwargs)
    
    
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
    def f5(xdic):
        x = tuple(xdic[k] for k in xdic.keys())
        return float(ftot(x)[4])
    
    search_space = SearchSpace(
        parameters=[
            RangeParameter(name=f"x{i}", 
                           lower=0, upper=1, 
                           parameter_type=ParameterType.FLOAT)
            for i in range(d)],
        )
    param_names = [f"x{i}" for i in range(d)]
    
    names = ["a", "b", "c", "d", "e"]
    functions = [f1, f2, f3, f4, f5]
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
                                                           objective_thresholds=objective_thresholds,)
    
    tmp_list = pickle.load(open(jsonFile, "rb" ))
    # THIS BELOW IS NOT WORKING. NOT ABLE TO DESERIALIZE THESE OBJECTS FROM CUDA TO CPU
    #tmp_list = torch.load(jsonFile, 
    #                      pickle_module=pickle, 
    #                      map_location={torch.device("cuda:0"): torch.device("cpu")}
    #                      )
    tmp_list.pop("experiment")
    tmp_list.pop("data")
    tmp_list.pop("outcomes")
    tmp_list["iteration"] = [i for i in range(tmp_list["last_call"] + 1)]
    tmp_list["HV_PARETO"] = [tmp_list["HV_PARETO"] for _ in range(tmp_list["last_call"] + 1)]
    tmp_list.pop("last_call")
    for k, v in tmp_list.items():
        print (k, len(v))
    pd.DataFrame(tmp_list).to_csv(outFileName, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-j", "--jsonFile", 
                        help="json file corresponding to the last iteration of optimization", 
                        required=True
                        )
    parser.add_argument("-m", "--M", 
                        help="number of objectives", 
                        required = True, type = int
                        )
    parser.add_argument("-d", "--D", 
                        help="number of dimensions", 
                        required = True, type = int
                        )
    parser.add_argument("-o", "--output", 
                        help="output file name in csv format", 
                        default = "output.csv"
                        )
    args = parser.parse_args()
    if (not os.path.exists(args.jsonFile)):
        print ("Error: file " + args.jsonFile + " does not exist")
        exit()
    ReadAndWriteOutputs(args.jsonFile, args.output, args.M, args.D)
    
    print("Done")