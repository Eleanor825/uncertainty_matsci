"""Diagnostic CPU training only; no production controller checkpoint format."""
from dataclasses import asdict
import math
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit
import torch
from torch import nn

STEPS = (0, 2, 4, 8, 16, 32, 64, 128, 200)


def require(ok, message):
    if not ok: raise ValueError(message)


def score(y, logits, metrics, temperature=1.):
    y = np.asarray(y); z = np.asarray(logits, dtype=np.float64)
    require(np.isfinite(z).all(), 'Nonfinite diagnostic logits')
    p = expit(z/temperature)
    result = metrics(y, p)
    # Selection uses raw BCE, not log_loss after clipping saturated probabilities.
    result['probability_metric_nll'] = result['nll']
    result['nll'] = float(np.mean(np.logaddexp(0,z/temperature)-y*z/temperature))
    result.update(raw_logit_min=float(z.min()), raw_logit_max=float(z.max()),
                  probability_min=float(p.min()), probability_max=float(p.max()),
                  n_at_or_above_fixed_point6=int((p >= .6).sum()))
    return result


def fit_duration(x, y, groups, names, config, updates, library, *, output, heldout=None,
                 candidates=STEPS, publish):
    """Original preprocessing/MLP/AdamW/BCE, explicit optimizer-step duration.

    heldout is a training episode only. It never affects gradients or fitted
    preprocessing. Every update is recorded; candidate states include step0.
    """
    require(updates >= 0 and updates <= 200 and not torch.cuda.is_initialized(), 'CPU bounded updates only')
    x = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=np.int64)
    require(len(x) == len(y) == len(groups) and set(y) == {0, 1}, 'Training rows/classes/groups differ')
    if heldout is not None:
        require(not set(groups) & set(heldout['groups']), 'A training episode leaked across its diagnostic fold')
    torch.manual_seed(config.seed)
    preprocessing = library.TrainOnlyFeatures().fit(x, y, names, groups)
    transformed = torch.from_numpy(preprocessing.transform(x, names, config.include_error_similarity))
    targets = torch.as_tensor(y, dtype=torch.float32)
    weights = torch.as_tensor(library._episode_weights(groups), dtype=torch.float32)
    model = library.RiskMLP(transformed.shape[1], config.hidden_width)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    generator = torch.Generator().manual_seed(config.seed)
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    training = {'rows': len(x), 'groups': sorted(set(groups)), 'config': asdict(config),
                'feature_names': list(names), 'preprocessing': preprocessing.state(),
                'stopping_rule': {'unit':'optimizer_updates','requested_updates':updates,
                                  'original_config_epochs_and_patience_are_reference_only':True},
                'loss': 'original inverse-episode-length weighted BCEWithLogits',
                'diagnostic_group_unit': 'original episode identity; original chemical metadata preserved separately'}
    publish(output/'training_inputs.json', training)
    history = []; snapshots = {}; epoch = 0; batches = []; index = 0
    prior = {k: v.detach().clone() for k, v in model.state_dict().items()}
    initial_hash = library.tensor_state_hash(prior)

    def evaluate(step):
        model.eval()
        with torch.no_grad():
            logits = model(transformed).numpy().astype(np.float64)
            held = None
            if heldout is not None:
                hx = torch.from_numpy(preprocessing.transform(heldout['x'], names, config.include_error_similarity))
                held = model(hx).numpy().astype(np.float64)
        state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        value = {'schema': 'standalone_scalar_diagnostic_state_v1', 'optimizer_updates': step,
                 'model': state, 'config': asdict(config), 'features': preprocessing.state(),
                 'stopping_rule':training['stopping_rule'],'actual_epochs_started':epoch,
                 'model_state_hash': library.tensor_state_hash(state), 'initial_state_hash': initial_hash,
                 'production_controller_checkpoint': False}
        path = output/f'step_{step:04d}.pt'
        with path.open('xb') as stream: torch.save(value, stream)
        record = {'updates': step, 'checkpoint': str(path), 'model_state_hash': value['model_state_hash'],
                  'train': score(y, logits, library.risk_metrics)}
        if held is not None:
            record['heldout_training_episode'] = score(heldout['y'], held, library.risk_metrics)
        publish(output/f'step_{step:04d}.json', record)
        snapshots[step] = record

    evaluate(0)
    for step in range(1, updates+1):
        if index >= len(batches):
            epoch += 1; batches = list(torch.randperm(len(x), generator=generator).split(config.batch_size)); index = 0
        batch = batches[index]; index += 1
        model.train()
        logits = model(transformed[batch])
        loss = (nn.functional.binary_cross_entropy_with_logits(logits, targets[batch], reduction='none')*weights[batch]).mean()
        require(bool(torch.isfinite(loss)), 'Nonfinite diagnostic training loss')
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        state = model.state_dict()
        delta = {k: float(torch.linalg.vector_norm((state[k]-prior[k]).double())) for k in state}
        require(all(math.isfinite(v) for v in delta.values()), 'Nonfinite parameter delta')
        history.append({'optimizer_update': step, 'epoch': epoch, 'batch_in_epoch': index,
                        'batch_rows': len(batch), 'weighted_BCE': float(loss.detach()),
                        'state_hash': library.tensor_state_hash(dict(state)), 'parameter_delta_l2': delta})
        prior = {k: v.detach().clone() for k, v in state.items()}
        if step in candidates or step == updates: evaluate(step)
    publish(output/'updates.json', history)
    model.eval()
    return model, preprocessing, snapshots


def choose_duration(folds, candidates=STEPS):
    require(len(folds) == 3, 'Exactly three original training episodes required')
    curve = []
    for step in candidates:
        values = [f[step]['heldout_training_episode']['nll'] for f in folds]
        require(all(math.isfinite(v) for v in values), 'Nonfinite training-fold selection metric')
        curve.append({'optimizer_updates': step, 'macro_training_holdout_BCE': float(np.mean(values)), 'fold_values': values})
    selected = min(curve, key=lambda x: (x['macro_training_holdout_BCE'], x['optimizer_updates']))
    return selected['optimizer_updates'], curve


def evaluate_development(model, features, x, y, names, library):
    """The original bounded temperature objective, after train-only selection."""
    with torch.no_grad():
        z = model(torch.from_numpy(features.transform(x, names, True))).numpy().astype(np.float64)
    y = np.asarray(y, dtype=np.int64)
    optimum = minimize_scalar(lambda log_t: float(np.mean(np.logaddexp(0,z/np.exp(log_t))-y*z/np.exp(log_t))),
                              bounds=(-4,4), method='bounded')
    require(bool(optimum.success) and math.isfinite(optimum.fun), 'Calibration solver failed; no success receipt')
    temperature = float(np.exp(optimum.x))
    return {'raw': score(y,z,library.risk_metrics), 'calibrated': score(y,z,library.risk_metrics,temperature),
            'temperature': temperature, 'calibration_log_bounds': [-4,4], 'fixed_controller_threshold': .6,
            'dev_calibration_metrics_are_in_sample': True,
            'dev_used_for_weight_or_duration_selection': False,
            'dNLL_d_inverse_temperature_at_zero': float(np.mean(z*(.5-y))),
            'raw_logits': z.tolist(), 'labels': y.tolist(), 'calibrated_probabilities': expit(z/temperature).tolist()}


def verify_saved_state(path, x, y, groups, names, config, library, heldout=None):
    """Recompute predictions/preprocessing from saved tensors without retraining."""
    value = torch.load(path,map_location='cpu',weights_only=True)
    require(value['schema']=='standalone_scalar_diagnostic_state_v1' and value['production_controller_checkpoint'] is False
            and value['config']==asdict(config) and value['stopping_rule']['unit']=='optimizer_updates'
            and value['stopping_rule']['original_config_epochs_and_patience_are_reference_only'] is True,
            'Wrong diagnostic checkpoint schema/config')
    features = library.TrainOnlyFeatures().fit(x,y,names,groups)
    require(features.state()==value['features'], 'Fold normalization/prototype used different rows')
    model = library.RiskMLP(value['model']['layers.0.weight'].shape[1],config.hidden_width)
    model.load_state_dict(value['model']); model.eval()
    require(library.tensor_state_hash(dict(model.state_dict()))==value['model_state_hash'], 'Saved tensor hash differs')
    if value['optimizer_updates']==0:
        torch.manual_seed(config.seed)
        initial = library.RiskMLP(value['model']['layers.0.weight'].shape[1],config.hidden_width)
        require(library.tensor_state_hash(dict(initial.state_dict()))==value['initial_state_hash']==value['model_state_hash'],
                'Step0 is not original seeded initialization')
    with torch.no_grad():
        z = model(torch.from_numpy(features.transform(x,names,config.include_error_similarity))).numpy()
        hz = None if heldout is None else model(torch.from_numpy(features.transform(heldout['x'],names,config.include_error_similarity))).numpy()
    record = {'updates':value['optimizer_updates'],'checkpoint':str(path),'model_state_hash':value['model_state_hash'],
              'train':score(y,z,library.risk_metrics)}
    if hz is not None:record['heldout_training_episode']=score(heldout['y'],hz,library.risk_metrics)
    return model,features,record
