from pathlib import Path
import json,hashlib,math
R=Path(__file__).resolve().parent
d=json.loads((R/'archive_actual.json').read_text());rows=d['archive_counterfactuals'];assert len(rows)==5
counts={'train':[0,0,0,0],'dev':[0,0,0,0]}
for row in rows:
 split='train' if row['episode'].startswith('collection_train_') else 'dev';assert row['prior_rows']==(60 if split=='train' else 90)
 events=row['events'];assert row['eligible_rows']==len(events)==25
 for x in events:
  assert x['original_label']==int(x['original_hv_increment']<=1e-12)
  assert x['alternate_archive_label']==int(x['alternate_hv_increment']<=1e-12)
 assert row['original_no_hvi']==sum(x['original_label'] for x in events)
 assert row['alternate_no_hvi']==sum(x['alternate_archive_label'] for x in events)
 assert row['label_flips']==[x for x in events if x['original_label']!=x['alternate_archive_label']]
 counts[split]=[a+b for a,b in zip(counts[split],[len(events),row['original_no_hvi'],row['alternate_no_hvi'],len(row['label_flips'])])]
assert counts=={'train':[75,70,73,3],'dev':[50,45,48,3]}
c=json.loads((R/'constant_train_prior_controls.json').read_text());p=70/75
for row in c.values():
 f=row['observed_no_hvi_fraction'];assert abs(row['brier']-(f*(1-p)**2+(1-f)*p*p))<1e-12
 assert abs(row['nll']-(-f*math.log(p)-(1-f)*math.log(1-p)))<1e-12
e=d['additional_full_5201'];q=[x for x in e['queries'] if x['risk'] is not None]
assert len(q)==45 and len(e['queries'])==50 and sum(x['no_hvi'] for x in q)==45
assert abs(sum((x['risk']-x['no_hvi'])**2 for x in q)/45-e['recomputed_brier'])<1e-12
assert d['all_inputs_unchanged'] and d['fits']==d['model_loads']==d['graph_calls']==d['oracle_calls']==0
for item in json.loads((R/'manifest.json').read_text())['files']:
 p=R/item['path'];assert p.stat().st_size==item['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256']
print(json.dumps({'passed':True,'archive_rows':125,'train_label_flips':3,'dev_label_flips':3,'new_scientific_calls':0,'counterfactual_labels_not_used_for_fitting':True}))
