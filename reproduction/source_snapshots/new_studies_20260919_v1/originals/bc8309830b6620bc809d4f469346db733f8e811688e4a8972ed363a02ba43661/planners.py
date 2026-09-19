"""Official Random/LHS and an explicitly new sklearn GP-EI scalarization.

All learning uses caller-supplied history only. The product utility is fixed by
the extension protocol; this is not EHVI, TSEMO, or an official Summit SOBO run.
"""
import numpy as np
from scipy.special import ndtr
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from oracle.official import BOUNDS, load_official, parameters

NAMES = tuple(BOUNDS)
LOW = np.asarray([BOUNDS[k][0] for k in NAMES])
SPAN = np.asarray([BOUNDS[k][1] - BOUNDS[k][0] for k in NAMES])


def product_utility(objectives):
    if set(objectives) != {"sty", "e_factor"}:
        raise ValueError("Exactly sty/e_factor required")
    sty, ef = (float(objectives[k]) for k in ("sty", "e_factor"))
    if not np.isfinite([sty, ef]).all():
        raise ValueError("Nonfinite historical objective")
    return (sty / 13000.0) * ((1000.0 - ef) / 1000.0)


def official_design(kind, seed, count):
    if kind not in {"random", "lhs"} or type(count) is not int or count < 1:
        raise ValueError("Invalid design request")
    cls, _, Random, LHS, _ = load_official()
    # _setup_domain is an unchanged official method; no ODE is executed here.
    domain = cls._setup_domain(None)
    planner = {"random": Random, "lhs": LHS}[kind](domain, random_state=np.random.RandomState(seed))
    design = planner.suggest_experiments(count)
    return [parameters({k: float(design[k].iloc[i]) for k in NAMES}) for i in range(count)]


def propose_with_metadata(history, seed, query_index, *, strategy="gp_ei", budget=50):
    if type(seed) is not int or not 0 <= seed < 2**32 or type(query_index) is not int or not 0 <= query_index < budget:
        raise ValueError("Invalid seed/query index")
    metadata = {"strategy": strategy, "seed": seed, "query_index": query_index, "history_count": len(history)}
    if strategy in {"random", "lhs"}:
        point = official_design(strategy, seed, budget)[query_index]
        return {"parameters": point, "planner": dict(metadata, phase="official_" + strategy)}
    if strategy != "gp_ei":
        raise ValueError("Unknown planner")
    if len(history) < 2:
        return {"parameters": official_design("lhs", seed, budget)[query_index],
                "planner": dict(metadata, phase="initial_design_insufficient_history")}
    X = np.asarray([[parameters(row["parameters"])[k] for k in NAMES] for row in history])
    X = (X - LOW) / SPAN
    y = np.asarray([product_utility(row["objectives"]) for row in history])
    local_seed = int(np.random.SeedSequence([seed, query_index, 1729]).generate_state(1)[0])
    candidates = qmc.Sobol(4, scramble=True, seed=local_seed).random_base2(10)
    candidates = candidates[np.min(np.max(np.abs(candidates[:, None] - X[None]), axis=2), axis=1) > 1e-12]
    if not len(candidates):
        raise ValueError("Candidate pool exhausted")
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(np.ones(4), (1e-2, 1e2), nu=2.5) + WhiteKernel(1e-6, (1e-9, 1e-2))
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, alpha=1e-10,
                                  n_restarts_optimizer=1, random_state=local_seed)
    gp.fit(X, y)
    mean, sd = gp.predict(candidates, return_std=True)
    improvement = mean - y.max() - 0.01
    z = improvement / np.maximum(sd, 1e-15)
    ei = improvement * ndtr(z) + sd * np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
    ei[sd < 1e-15] = 0.0
    index = int(np.argmax(ei))
    point = parameters(dict(zip(NAMES, map(float, LOW + candidates[index] * SPAN))))
    return {"parameters": point, "planner": dict(metadata, phase="gp_expected_improvement",
            official_summit_sobo=False, scalarization="(sty/13000)*((1000-e_factor)/1000)",
            sty_upper_clipped=False, gp_kernel=str(gp.kernel_), candidate_pool_size=len(candidates),
            acquisition="EI", xi=0.01, selected_ei=float(ei[index]),
            selected_predictive_mean=float(mean[index]), selected_predictive_sd=float(sd[index]))}


def propose(history, seed, query_index, *, strategy="gp_ei", budget=50):
    return propose_with_metadata(history, seed, query_index, strategy=strategy, budget=budget)["parameters"]
