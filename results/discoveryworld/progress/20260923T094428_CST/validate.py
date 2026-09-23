"""Checks this additive bundle and saved predictions only; no science/history scan."""
from pathlib import Path
import csv,hashlib,json,math
from collections import Counter,defaultdict
def weighted_mse(ids,predictions,rows):
 rs=[rows[k]for k in ids];counts=Counter((r['source_episode'],r['input_hash'])for r in rs);groups=defaultdict(set)
 for r in rs:groups[r['source_episode']].add(r['input_hash'])
 return sum((pred-r['target'])**2/(len(groups)*len(groups[r['source_episode']])*counts[r['source_episode'],r['input_hash']])for pred,r in zip(predictions,rs))
P=Path(__file__).resolve().parent
def load(n):return json.loads((P/n).read_bytes())
for n,v in load('manifest.json')['files'].items():
 b=(P/n).read_bytes();assert len(b)==v['bytes']and hashlib.sha256(b).hexdigest()==v['sha256']
s=load('snapshot.json');p=s['p335'];assert [r['task_score']for r in p['rows']]==[0,0,.125,0,.25]
assert [r['official_action_failures']for r in p['rows']]==[4,3,4,3,4]and all(not r['task_success']for r in p['rows'])
assert [r['task_score_difference']for r in p['contrasts']]==[.125,.125,.25]and p['single_seed_variance']is None
assert [r['official_action_failures']for r in s['p332']['rows']]==[29,9,20]and s['p332']['NN_minus_common_task_score']==0
assert len(list(csv.DictReader((P/'p335_actions_all150.csv').open())))==150
u=load('utility_all_predictions_and_seals.json');targets={r['window_id']:r['target']for r in u['rows']};rowmap={r['window_id']:r for r in u['rows']};assert len(targets)==78 and sum(y>0 for y in targets.values())==12 and sum(y==0 for y in targets.values())==66
fold_count=0
for choices in u['selection']['all_candidates'].values():
 assert len(choices)==6
 for c in choices:
  scores=[]
  for f in c['folds']:
   error=weighted_mse(f['validation_window_ids'],f['predictions'],rowmap);assert math.isclose(error,f['validation']['MSE'],rel_tol=1e-10,abs_tol=1e-14);scores.append(error);fold_count+=1
  assert math.isclose(sum(scores)/10,c['macro_source_MSE'],rel_tol=1e-10)
assert fold_count==120 and all(not r['beats_constant_on_selection_CV']for r in u['selection']['selection'].values())
assert u['counts']['fits']==42 and u['counts']['updates']==8250 and not u['online_admitted']
assert sum(r['attempts']for r in s['native7']['rows'])==700 and s['publication_scientific_calls']==0
print(json.dumps({'passed':True,'p335_all_arms':5,'p335_actions':150,'p332_all_arms':3,'native_TRAIN_episodes':7,'TRAIN_windows':78,'grid_candidates':12,'CV_fold_records':120,'utility_fits':42,'utility_updates':8250,'utility_beats_constant':False,'new_scientific_calls':0},sort_keys=True))
