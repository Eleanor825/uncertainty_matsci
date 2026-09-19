# Scientific source excerpt: uncertainty.py

Logical source: `src/matdiscovery/uncertainty.py`.

SHA256 of the **complete original source**: `62b81414c2c1cc2df467a5d20cf44f76fd326ce9ee7562de5475529953a08656`. This is a quoted excerpt for review, not a portable executable or the complete source file. Line numbers below refer to the original file; the export manifest separately hashes this excerpt.

## Lines 116–174

```text
116:     def fit(self, train_x, train_y, dev_x, dev_y, *, feature_names, train_groups, dev_groups, train_episode_ids, config: RiskTrainingConfig):
117:         assert_group_disjoint(train_groups, dev_groups)
118:         train_y, dev_y = np.asarray(train_y), np.asarray(dev_y)
119:         if len(np.unique(train_y)) != 2 or len(dev_y) < 2 or not np.isin(dev_y, [0, 1]).all():
120:             raise ValueError("Training needs both observed classes and development needs valid labels")
121:         if len(train_episode_ids) != len(train_y):
122:             raise ValueError("Episode identifiers are required to avoid overweighting long trajectories")
123:         if len(dev_groups) != len(dev_y):
124:             raise ValueError("Development groups missing")
125:         torch.manual_seed(config.seed)
126:         self.config = config
127:         self.features = TrainOnlyFeatures().fit(train_x, train_y, feature_names, train_groups)
128:         x = torch.from_numpy(self.features.transform(train_x, feature_names, config.include_error_similarity))
129:         dx = torch.from_numpy(self.features.transform(dev_x, feature_names, config.include_error_similarity))
130:         y = torch.as_tensor(train_y, dtype=torch.float32)
131:         dy = torch.as_tensor(dev_y, dtype=torch.float32)
132:         weights = torch.as_tensor(_episode_weights(train_episode_ids), dtype=torch.float32)
133:         self.model = RiskMLP(x.shape[1], config.hidden_width)
134:         optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
135:         generator = torch.Generator().manual_seed(config.seed)
136:         best_loss, best_state, stale = float("inf"), None, 0
137:         self.history = []
138:         for epoch in range(config.epochs):
139:             self.model.train()
140:             order = torch.randperm(len(x), generator=generator)
141:             for batch in order.split(config.batch_size):
142:                 logits = self.model(x[batch])
143:                 loss = (nn.functional.binary_cross_entropy_with_logits(logits, y[batch], reduction="none") * weights[batch]).mean()
144:                 optimizer.zero_grad(set_to_none=True)
145:                 loss.backward()
146:                 optimizer.step()
147:             self.model.eval()
148:             with torch.no_grad():
149:                 dev_loss = float(nn.functional.binary_cross_entropy_with_logits(self.model(dx), dy))
150:             self.history.append({"epoch": epoch + 1, "development_bce": dev_loss})
151:             if not np.isfinite(dev_loss):
152:                 raise RuntimeError("Nonfinite uncertainty training loss")
153:             if dev_loss < best_loss - 1e-8:
154:                 best_loss, best_state, stale = dev_loss, {k: v.detach().clone() for k, v in self.model.state_dict().items()}, 0
155:                 self.selected_epoch = epoch + 1
156:             else:
157:                 stale += 1
158:             if stale >= config.patience:
159:                 break
160:         self.model.load_state_dict(best_state)
161:         with torch.no_grad():
162:             dev_logits = self.model(dx).numpy().astype(np.float64)
163:         # Scalar temperature calibration uses development labels only.
164:         fit = minimize_scalar(lambda log_t: float(np.mean(np.logaddexp(0, dev_logits / np.exp(log_t)) - dev_y * dev_logits / np.exp(log_t))), bounds=(-4, 4), method="bounded")
165:         self.temperature = float(np.exp(fit.x))
166:         self.provenance = {"training_group_hash": _fingerprint(train_groups), "development_group_hash": _fingerprint(dev_groups), "train_rows": len(train_y), "development_rows": len(dev_y), "label_kind": config.label_kind, "test_used_for_fit": False}
167:         return self
168:
169:     def predict_proba(self, x, feature_names):
170:         self.model.eval()
171:         values = self.features.transform(x, feature_names, self.config.include_error_similarity)
172:         with torch.no_grad():
173:             logits = self.model(torch.from_numpy(values)).numpy()
174:         return expit(logits / self.temperature)
```

## Lines 194–209

```text
194:
195:
196: def risk_metrics(y, probability, *, bins=10):
197:     y, p = np.asarray(y, dtype=int), np.asarray(probability, dtype=float)
198:     if y.shape != p.shape or not len(y) or not np.isin(y, [0, 1]).all() or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
199:         raise ValueError("Metrics require complete valid binary labels and probabilities")
200:     ece = 0.0
201:     assignments = np.minimum((p * bins).astype(int), bins - 1)
202:     for b in range(bins):
203:         mask = assignments == b
204:         if mask.any():
205:             ece += mask.mean() * abs(p[mask].mean() - y[mask].mean())
206:     order = np.argsort(p, kind="stable")
207:     risks = np.cumsum(y[order]) / np.arange(1, len(y) + 1)
208:     return {"n": len(y), "error_rate": float(y.mean()), "auroc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None, "error_auprc": float(average_precision_score(y, p)) if y.sum() else None, "brier": float(brier_score_loss(y, p)), "nll": float(log_loss(y, np.clip(p, 1e-8, 1 - 1e-8), labels=[0, 1])), "ece": float(ece), "risk_coverage_auc": float(risks.mean()), "overconfident_error_rate_p_le_0_1": float(np.mean((y == 1) & (p <= .1)))}
```
