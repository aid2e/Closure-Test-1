import torch  # pytorch package, allows using GPUs

from ax.metrics.noisy_function import GenericNoisyFunctionMetric
from ax.service.utils.report_utils import \
    exp_to_df  # https://ax.dev/api/service.html#ax.service.utils.report_utils.exp_to_df
from ax.runners.synthetic import SyntheticRunner

# Plotting imports and initialization
# from ax.utils.notebook.plotting import render, init_notebook_plotting
from ax.plot.contour import plot_contour
from ax.plot.pareto_utils import compute_posterior_pareto_frontier
from ax.plot.pareto_frontier import plot_pareto_frontier
# init_notebook_plotting()

# Model registry for creating multi-objective optimization models.
from ax.modelbridge.registry import Models

# Analysis utilities, including a method to evaluate hypervolumes
from ax.modelbridge.modelbridge_utils import observed_hypervolume

from ax import SumConstraint
from ax import OrderConstraint
from ax import ParameterConstraint
from ax.core.search_space import SearchSpace
from ax.core.parameter import RangeParameter, ParameterType

from ax.core.objective import MultiObjective, Objective, ScalarizedObjective
from ax.core.optimization_config import ObjectiveThreshold, MultiObjectiveOptimizationConfig

from ax.core.experiment import Experiment

from botorch.utils.multi_objective.box_decompositions.dominated import DominatedPartitioning
from ax.core.data import Data

from ax.core.types import ComparisonOp

from sklearn.utils import shuffle
from functools import wraps, lru_cache

from matplotlib import pyplot as plt

from matplotlib.cm import ScalarMappable


# ---------------------- TOY FUNCTIONS ------------------------#

def glob_dummy(loc_fun):
    @wraps(loc_fun)
    @lru_cache(maxsize=None)
    def inner(xdic):
        x_sorted = [xdic[p_name] for p_name in xdic.keys()]  # it assumes x will be given as, e.g., dictionary
        res = loc_fun(x_sorted)
        return res

    return inner


# Define the objectives
@glob_dummy
def f1(x):
    res = np.sum(x * x)
    return res  # obj1


@glob_dummy
def f2(x):
    res = 1. / (np.sum(x * x) + 0.01)
    return res  # obj2


"""
def glob_fun(loc_fun):
  @lru_cache(maxsize=None)
  def inner(xdic):
    x_sorted = [xdic[p_name] for p_name in xdic.keys()]
    res = list(loc_fun(x_sorted))
    return res
  return inner
"""


def glob_fun(loc_fun):
    @wraps(loc_fun)
    @lru_cache(maxsize=None)
    def inner(xsorted):
        res = list(loc_fun(xsorted))
        return res

    return inner


# ---------------------- BOTORCH FUNCTIONS ------------------------#


def build_experiment(search_space, optimization_config):
    experiment = Experiment(
        name="pareto_experiment",
        search_space=search_space,
        optimization_config=optimization_config,
        runner=SyntheticRunner(),
    )
    return experiment


def initialize_experiment(experiment, N_INIT):
    sobol = Models.SOBOL(search_space=experiment.search_space)

    experiment.new_batch_trial(sobol.gen(N_INIT)).run()

    return experiment.fetch_data()


def get_hypervolume(experiment, ref_point, tkwargs):
    exp_df = exp_to_df(experiment)
    metric_names = experiment.optimization_config.metrics.keys()
    outcomes = torch.tensor(exp_df[metric_names].values, **tkwargs)
    partitioning = DominatedPartitioning(ref_point=ref_point, Y=outcomes)
    try:
        return partitioning.compute_hypervolume().item()
    except:
        return 0.
