"""Small-CPU-NN diagnostic; original optimizer/preprocessor/model are imported unchanged."""
from dataclasses import asdict
import importlib.util
from pathlib import Path
import sys
import math
import statistics
import numpy as np
import torch


def require(ok, message):
    if not ok: raise ValueError(message)


def original_module(path):
    name = '_snar_original_risk_feature_diagnostic'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    exec(compile(Path(path).read_bytes(), str(path), 'exec'), module.__dict__)
    return module


def names_for(names, family):
    require(family in ('sampling', 'hidden', 'graph', 'all'), 'Unknown feature family')
    return list(names) if family == 'all' else [k for k in names if k.startswith(family+'.')]


def matrix(rows, names):
    return np.array([[r['features'][k] for k in names] for r in rows], dtype=np.float64)


def fit(module, training, development, names, config, seed):
    # No test argument exists. Subset happens before every learned preprocessing step.
    cfg = module.RiskTrainingConfig(label_kind=config['label'], **{
        k: seed if k == 'seed' else config[k] for k in
        ('seed', 'hidden_width', 'epochs', 'batch_size', 'learning_rate', 'weight_decay', 'patience', 'include_error_similarity')})
    model = module.CalibratedRiskModel().fit(matrix(training, names), np.array([r['label'] for r in training]),
        matrix(development, names), np.array([r['label'] for r in development]), feature_names=names,
        train_groups=[r['episode'] for r in training], dev_groups=[r['episode'] for r in development],
        train_episode_ids=[r['episode'] for r in training], config=cfg)
    require(model.features.names == names and model.features.mean.shape == (len(names),), 'Feature subsetting leaked')
    require(all(t.device.type == 'cpu' for t in model.model.state_dict().values()), 'Unexpected model device')
    return model


def raw_and_calibrated(model, rows, names, *, individually=False):
    x = matrix(rows, names)
    batches = [x[i:i+1] for i in range(len(x))] if individually else [x]
    logits, probabilities = [], []
    for batch in batches:
        values = model.features.transform(batch, names, model.config.include_error_similarity)
        model.model.eval()
        with torch.no_grad(): z = model.model(torch.from_numpy(values)).numpy()
        logits.extend(float(v) for v in z)
        probabilities.extend(float(v) for v in model.predict_proba(batch, names))
    # Match original expit float32 behavior, including before calibration.
    from scipy.special import expit
    raw = expit(np.asarray(logits, dtype=np.float32))
    return {'logits': logits, 'raw': [float(v) for v in raw], 'calibrated': probabilities}


def assert_replication(retrained, original, old_action_probabilities, new_action_probabilities, *, atol):
    for name in ('features', 'config', 'provenance', 'history', 'selected_epoch', 'temperature'):
        a, b = getattr(retrained, name), getattr(original, name)
        if name == 'features': a, b = a.state(), b.state()
        elif name == 'config': a, b = asdict(a), asdict(b)
        require(a == b, 'Original replication differs: '+name)
    left, right = retrained.model.state_dict(), original.model.state_dict()
    require(left.keys() == right.keys(), 'Original model state keys differ')
    require(all(torch.equal(left[k], right[k]) for k in left), 'Original replication parameter tensors differ')
    require(len(old_action_probabilities) == len(new_action_probabilities) and len(old_action_probabilities), 'Replication score cohort differs')
    differences = [abs(float(a)-float(b)) for a, b in zip(old_action_probabilities, new_action_probabilities)]
    require(all(math.isfinite(v) and v <= atol for v in differences), 'Original saved action score replication differs')
    return {'passed': True, 'exact_model_tensors': True, 'exact_training_metadata': True,
        'saved_action_scores': len(differences), 'maximum_absolute_score_difference': max(differences),
        'score_absolute_tolerance': atol, 'test_outcome_labels_used': False}


def metrics(module, labels, predictions, threshold):
    value = module.risk_metrics(labels, predictions)
    value['threshold'] = threshold
    value['at_or_above_fixed_threshold'] = sum(p >= threshold for p in predictions)
    return value


def score(module, model, rows, names, threshold):
    values = raw_and_calibrated(model, rows, names, individually=True)
    labels = [r['label'] for r in rows]
    return {'raw': metrics(module, labels, values['raw'], threshold),
        'calibrated': metrics(module, labels, values['calibrated'], threshold)}, values


def summarize_initializations(reports):
    """No ranking/winner: fixed-family summaries across exactly 3 NN initializations."""
    out = {}
    for family in ('sampling', 'hidden', 'graph', 'all'):
        rows = [v for v in reports if v['family'] == family]
        require(len(rows) == 3, 'Incomplete initialization population')
        families = {}
        for calibration in ('raw', 'calibrated'):
            metrics_ = {}
            for key in ('auroc', 'error_auprc', 'brier', 'nll', 'ece', 'risk_coverage_auc'):
                values = [r['test'][calibration][key] for r in rows]
                valid = [x for x in values if x is not None]
                metrics_[key] = {'n_defined': len(valid), 'values': values,
                    'mean': statistics.mean(valid) if valid else None,
                    'sample_variance': statistics.variance(valid) if len(valid) > 1 else None}
            families[calibration] = metrics_
        out[family] = families
    return out
