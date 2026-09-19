# Scientific source excerpt: acceptance_repair.py

Logical source: `benchmark_extensions/summit_snar_main_repair_20260919_v2/acceptance_repair.py`.

SHA256 of the **complete original source**: `57567104060e594b5e7713faec62ed36f75bb060064b4b38d58ada7c3e7a26c9`. This is a quoted excerpt for review, not a portable executable or the complete source file. Line numbers below refer to the original file; the export manifest separately hashes this excerpt.

## Lines 202–213

```text
202:             scores=[]
203:             for seed in cfg['checkpoint_dev_seeds']:
204:                 s=episodes[f'es_{branch}_dev_G{g}_{seed}'];require(s['weight_hash']==states[g],'Development evaluated wrong generation')
205:                 scores.append(fitness(s) if branch=='full' else s['mean_querywise_hv']-.1*s['invalid_proposal_rate'])
206:             record=ev.read(root/f'{branch}_es_development_G{g}.json')
207:             expected={'generation':g,'fitness':sum(scores)/len(scores),'scores':scores,'weight_hash':states[g]}
208:             require(_same(record,expected),'Development fitness record differs');dev.append(record)
209:         selected=max(dev,key=lambda x:(x['fitness'],-x['generation']))
210:         selection=ev.read(root/f'{branch}_checkpoint_selection.json')
211:         require(selection['test_used'] is False and _same(selection['development'],dev) and _same(selection['selected'],selected),'Test-informed or different checkpoint selection')
212:         selected_history=ev.read(root/f'{branch}_selected_es_history.json')
213:         require(selected_history['history']==full['history'][:selected['generation']] and selected_history['current_state_hash']==selected['weight_hash']
```

## Lines 265–272

```text
265:     prior=ev.read(root/'shared_prior_archive.json');adaptation=ev.read(root/'adaptation_complete.json')
266:     require(len(prior)==550 and adaptation['queries']==550 and adaptation['archive_fingerprint']==digest(prior),'Common adaptation prior differs')
267:     base_record=ev.read(root/'base_policy.json');base=base_record['base_state_hash'];require(_sha(base),'Invalid base weight hash')
268:     summaries={};all_rows={};adapt_rows=[]
269:     for name,spec in sorted(expected.items()):
270:         folder=folders[name];header=ev.read(folder/'episode.json');p=prior if spec.get('role')=='test' else []
271:         require(all(header[k]==spec[k] for k in ('name','arm','seed','budget')) and header['protocol_fingerprint']==digest(protocol)
272:                 and header['prior_fingerprint']==digest(p),'Episode scope/prior identity differs')
```

## Lines 281–310

```text
281:                     and (receipt['cache_hit'],receipt['physical_oracle_calls_this_request']) in ((False,1),(True,0)),'Query result does not match its immutable official terminal')
282:             require(row['parameters']==original['parameters'] and row['objectives']==original['objectives'],'Published objective/parameters differ from official result')
283:             require(set(row['parameters'])==set(BOUNDS) and all(type(row['parameters'][k]) in (int,float) and math.isfinite(row['parameters'][k]) and lo<=row['parameters'][k]<=hi for k,(lo,hi) in BOUNDS.items()),'Invalid executed parameter')
284:             if i<5:require(row['parameters']==design[i] and row['kind']=='paired_lhs_initial_design','Paired initialization differs')
285:             before=hypervolume([*p,*rows]);after=hypervolume([*p,*rows,row])
286:             require(_same(row['hv_increment'],after-before) and row['no_hvi']==int(after-before<=protocol['objective']['non_improvement_tolerance']),'Incorrect HV increment or observed risk label')
287:             risk=row.get('risk');require(risk is None or type(risk) in (int,float) and math.isfinite(risk) and 0<=risk<=1,'Invalid uncertainty value')
288:             require(type(row['proposal_count']) is int and 0<=row['invalid_proposals']<=row['proposal_count']<=2,'Proposal accounting differs')
289:             if spec['arm'] not in ('uq_esopt','uq_only'):require(risk is None,'Ablation/base silently used risk predictions')
290:             intent=ev.read(folder/f'query{i:03d}_intent.json')
291:             require(intent['query_id']==qid and intent['parameters']==row['parameters'] and _same(intent['prior_and_history_hv'],before)
292:                     and intent['chosen_generation']==row['chosen_generation'],'Published query intent differs')
293:             candidates=[]
294:             for directory in sorted(folder.glob(f'query{i:03d}_proposal*')):
295:                 require(directory.name in {f'query{i:03d}_proposal0',f'query{i:03d}_proposal1'},'Unregistered extra proposal')
296:                 candidate=ev.read(directory/'candidate.json');saved=ev.read(directory/'generation.json');generation=saved['generation']
297:                 require(Path(candidate['generation_path']).resolve()==(directory/'generation.json').resolve()
298:                         and saved['weight_hash']==header['weight_hash'] and generation['model_stamp']['checkpoint_hash']==base_record['checkpoint_hash']
299:                         and generation['configuration_fingerprint']==base_record['configuration_fingerprint']
300:                         and generation['policy_runtime']==base_record['runtime'],'Candidate policy/configuration differs')
301:                 require(_prefix_hash(generation['input_ids_with_completion'])==generation['prefix_hash'],'Stored generated token bytes differ')
302:                 start=generation['prompt_token_count'];length=len(generation['input_ids_with_completion'][0])
303:                 require(type(start) is int and 1<=start<=length and generation['completion_count']==length-start<=128
304:                         and (generation['success'] is False or start<length),'Original token boundary differs')
305:                 payload=saved['prompt']['prompt_payload']
306:                 require(payload['query_count']==i and payload['budget']==spec['budget'] and payload['remaining_budget']==spec['budget']-i
307:                         and payload['history']==[observed(r) for r in rows][-protocol['policy']['history_window']:]
308:                         and payload['history_omitted_count']==max(0,i-protocol['policy']['history_window'])
309:                         and payload['shared_prior']==prior_summary(p,4),'Candidate prompt includes wrong/future data')
310:                 require(candidate['success']==generation['success'] and 'no_hvi' not in candidate,'Candidate observation/label leakage')
```

## Lines 327–346

```text
327:         if spec['arm'] in ('uq_esopt','uq_only'):
328:             valid=[c for c in all_candidates if c['success']];fraction=sum(c['risk'] is not None for c in valid)/len(valid) if valid else 0.
329:             require(_same(actual.get('valid_proposal_graph_fraction'),fraction) and fraction>=protocol['risk']['minimum_valid_graph_fraction'],'Native graph coverage failed')
330:         summaries[name]=actual
331:         if spec.get('role')!='test':adapt_rows.extend(rows)
332:     require(set(all_rows)==set(physical) and len(physical)==2300,'Extra/omitted physical calls')
333:     require(prior==adapt_rows,'Shared prior is not the exact ordered complete adaptation archive')
334:     selections=_evolution(root,protocol,summaries,ev,base)
335:     for name,spec in expected.items():
336:         if spec.get('role')=='test':
337:             expected_hash=selections['full']['weight_hash'] if spec['arm']=='uq_esopt' else selections['es_only']['weight_hash'] if spec['arm']=='es_only' else None if spec['arm'] in ('random','gp_ei_scalarized') else base
338:             require(summaries[name]['weight_hash']==expected_hash,'Evaluation uses the wrong selected/base policy')
339:         elif name.startswith('collection_'):require(summaries[name]['weight_hash']==base,'Risk collection not original policy')
340:     risk=ev.read(root/'risk_fit.json');require(risk['test_used'] is False and risk['provenance']['test_used_for_fit'] is False,'Risk fit used test data')
341:     tr=risk['rows']['train'];dv=risk['rows']['dev'];require(tr and dv,'Empty risk split')
342:     require(not ({r['episode'] for r in tr}&{r['episode'] for r in dv}) and not ({r['prefix_hash'] for r in tr}&{r['prefix_hash'] for r in dv}),'Risk train/dev leakage')
343:     for split,rows in [('train',tr),('dev',dv)]:
344:         for row in rows:
345:             require(row['episode'] in expected and expected[row['episode']].get('split')==split,'Risk includes another scientific split')
346:             original=ev.read(row['query_path']);require(row['label']==original['no_hvi'],'Risk label differs from executed observation')
```
