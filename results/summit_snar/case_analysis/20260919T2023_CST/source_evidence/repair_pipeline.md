# Scientific source excerpt: repair_pipeline.py

Logical source: `benchmark_extensions/summit_snar_main_repair_20260919_v2/repair_pipeline.py`.

SHA256 of the **complete original source**: `b9179c9058572cca44db6d22b9776edfa2f20a6b401505b86f66d57693f90903`. This is a quoted excerpt for review, not a portable executable or the complete source file. Line numbers below refer to the original file; the export manifest separately hashes this excerpt.

## Lines 82–89

```text
82: def eligible(reg):
83:     out=Path(reg['output']);diag=c.read(reg['original_diagnostic']['path']);items=[]
84:     for source in diag['prefixes']:
85:         original=Path(source['generation']['path']);generation=out/original.relative_to(reg['original_output'])
86:         items.append(dict(index=source['index'],split=source['split'],episode=source['episode'],label=source['label'],
87:             generation=generation,query=out/Path(source['query']['path']).relative_to(reg['original_output']),
88:             graph=generation.parent/'risk_graph.json',prefix_hash=source['original_generation_prefix_hash']))
89:     c.require(len(items)==125 and [x['index'] for x in items]==list(range(125)),'Incomplete125-prefix graph population')
```

## Lines 131–159

```text
131: def fit_risk(reg):
132:     # Exact original fit tail: no policy/Oracle construction and no outcome features.
133:     c.modules(reg)
134:     import numpy as np
135:     import torch
136:     from matdiscovery.uncertainty import CalibratedRiskModel,RiskTrainingConfig,risk_metrics
137:     torch.set_num_threads(2);out=Path(reg['output']);protocol=c.read(reg['original_protocol']['path'])
138:     rows={'train':[],'dev':[]};okay=0;total=125;dev_total=50
139:     for item in eligible(reg):
140:         graph=c.read(item['graph'])
141:         if not graph['available']:continue
142:         okay+=1;rows[item['split']].append(dict(features=graph['features'],label=item['label'],episode=item['episode'],
143:             query_path=str(item['query']),graph_path=str(item['graph']),prefix_hash=item['prefix_hash']))
144:     c.require(okay/total>=.9,'SnAr native training graph availability below registered90%')
145:     names=sorted({k for row in rows['train'] for k in row['features']})
146:     arrays={s:np.array([[row['features'].get(k,float('nan')) for k in names] for row in rs]) for s,rs in rows.items()}
147:     ys={s:np.array([row['label'] for row in rs]) for s,rs in rows.items()};cfg=protocol['risk']
148:     config=RiskTrainingConfig(label_kind=cfg['label'],**{k:cfg[k] for k in
149:         ('seed','hidden_width','epochs','batch_size','learning_rate','weight_decay','patience','include_error_similarity')})
150:     risk=CalibratedRiskModel().fit(arrays['train'],ys['train'],arrays['dev'],ys['dev'],feature_names=names,
151:         train_groups=[x['episode'] for x in rows['train']],dev_groups=[x['episode'] for x in rows['dev']],
152:         train_episode_ids=[x['episode'] for x in rows['train']],config=config)
153:     risk.save(out/'snar_risk.pt');probabilities=risk.predict_proba(arrays['dev'],names)
154:     report=dict(feature_names=names,rows=rows,graph_fraction=okay/total,provenance=risk.provenance,
155:         development_metrics=risk_metrics(ys['dev'],probabilities),
156:         random_controller_retry_probability=float((np.sum(probabilities>=.5)+dev_total-len(rows['dev']))/dev_total),
157:         random_controller_rate_population=dict(valid_development_actions_after_prefix_filter=dev_total,
158:             available_graphs=len(rows['dev']),unavailable_graphs_counted_as_retry=dev_total-len(rows['dev'])),test_used=False)
159:     c.publish(out/'risk_fit.json',report)
```

## Lines 185–191

```text
185:         elif kind=='freeze':
186:             p.freeze();paths=[out/'shared_prior_archive.json',out/'adaptation_complete.json']
187:         elif kind=='evaluate':
188:             prior=c.read(out/'shared_prior_archive.json')
189:             if job['arm'] in ('uq_esopt','uq_only'):p.load_method()
190:             if job['arm'] in ('uq_esopt','es_only'):selected_generation(p,'full' if job['arm']=='uq_esopt' else 'es_only')
191:             p.episode(job['job_id'],arm=job['arm'],seed=job['seed'],budget=50,prior=prior)
```
