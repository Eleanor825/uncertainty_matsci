"""Recompute published audit summaries and the exact dev-only selector; stdlib only."""
from pathlib import Path
import argparse,collections,copy,csv,hashlib,importlib.util,json,math,re,statistics
HERE=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def need(ok,msg):
 if not ok:raise ValueError(msg)
def validate(repo):
 provenance=json.loads((HERE/'code/source_provenance.json').read_text())
 need(sha(HERE/'code/selector.py')==provenance['selector_source_sha256'],'Selector source changed')
 need(sha(HERE/'code/protocol.json')==provenance['protocol_sha256'],'Protocol changed')
 need(sha(HERE/'selection.json')==provenance['original_selection_output_sha256'],'Original selection output changed')
 projection=provenance['original_dev_projection'];p=repo/projection['path'];need(sha(p)==projection['sha256'],'Historical dev projection SHA changed')
 value=json.loads(p.read_text());spec=importlib.util.spec_from_file_location('_public_incumbent_selector',HERE/'code/selector.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 rows=m.rows_from_historical_projection(value);chosen=m.select_incumbent(rows,expected_seeds=[value['environment_seed']]);saved=json.loads((HERE/'selection.json').read_text())
 need(all(saved[k]==v for k,v in chosen.items()),'Original selector replay differs')
 need((chosen['original_selected_generation'],chosen['selected_generation'])==(2,0),'Wrong original/new ranking')
 need(saved['test_outcomes_read_by_this_audit']is False and saved['eligible_for_live_model_activation']is False,'Wrong audit/deployment claim')
 changed=copy.deepcopy(rows)
 for row in changed:row['AUDC']=.5
 tie=m.select_incumbent(changed,expected_seeds=[value['environment_seed']]);need((tie['original_selected_generation'],tie['selected_generation'])==(1,0),'Tie rule differs')
 changed[1].update(status='failed',AUDC=None);incomplete=m.select_incumbent(changed[:2],expected_seeds=[value['environment_seed']]);need(incomplete['selected_generation']is None and incomplete['deployment_default_generation']==0,'Unknown became winner/zero')
 for replacement in ('test','non_AUDC','duplicate'):
  changed=copy.deepcopy(rows)
  if replacement=='test':changed[0]['split']='test'
  elif replacement=='non_AUDC':changed[0]['metric_name']='MACE'
  else:changed.append(changed[0])
  try:m.select_incumbent(changed,expected_seeds=[value['environment_seed']])
  except ValueError:pass
  else:raise ValueError('Bad input accepted: '+replacement)
 summary=json.loads((HERE/'evidence_summary.json').read_text())
 for item in summary['inputs'].values():need(sha(repo/item['path'])==item['sha256'],'Historical public source changed: '+item['path'])
 with(repo/summary['inputs']['latest_results']['path']).open(newline='')as f:results=list(csv.DictReader(f))
 need(dict(collections.Counter(r['status']for r in results))=={'accepted':1079,'failed':1},'Original result count differs')
 with(repo/summary['inputs']['latest_pairs']['path']).open(newline='')as f:pairs=list(csv.DictReader(f))
 for reported in summary['paired_recomputed_from_published_CSV']:
  group=[r for r in pairs if r['arm']==reported['arm']and int(r['budget'])==reported['budget']]
  need(len(group)==reported['n']==(89 if reported['budget']==50 else 90),'Wrong matched denominator')
  for metric in ('SUN','AUDC'):need(math.isclose(statistics.mean(float(r[metric+'_delta'])for r in group),reported['mean_'+metric+'_delta'],abs_tol=1e-12),'Paired mean differs')
 critic=json.loads((HERE/'critic_metrics.json').read_text());scalar=critic['original_scalar'];dev=critic['original_fit']['dev']
 need(scalar['selected_epoch']==1 and scalar['optimization_steps_retained']==2 and len(scalar['history'])==16,'Wrong original training history')
 need(math.isclose(dev['calibrated']['auroc'],.516260162601626,abs_tol=1e-14),'Wrong original development AUC')
 need(math.isclose(scalar['temperature'],54.5978037363953,abs_tol=1e-12),'Wrong original temperature')
 manifest=json.loads((HERE/'manifest.json').read_text())
 for name,item in manifest['files'].items():
  p=HERE/name;need(p.is_file()and not p.is_symlink()and sha(p)==item['sha256']and p.stat().st_size==item['bytes'],'New export file differs: '+name)
 for p in HERE.rglob('*'):
  if p.is_file()and p.suffix in ('.json','.py','.md'):
   text=p.read_text()
   # Construct patterns without embedding private root strings in the public file.
   forbidden=['/'+ 'Users'+'/', '/'+ 'mnt'+'/',r'tj-\d+-t-\d+',r'GPU-[0-9a-f]{8}-[0-9a-f-]{20,}',r'Bearer\s+[A-Za-z0-9._-]{12,}']
   need(not any(re.search(pattern,text)for pattern in forbidden),'Unredacted private material: '+p.name)
 return {'passed':True,'original_and_public_projection_SHA_identical':True,'original_selected_generation':2,'incumbent_selected_generation':0,'original_scalar_dev_AUROC':dev['calibrated']['auroc'],'accepted_original_results':1079,'retained_technical_failure':1,'pair_summaries_recomputed':9,'selector_edge_case_checks':5,'new_scientific_calls':0,'new_heldout_experiments':0,'production_model_activated':False,'manifest_files_checked':len(manifest['files'])}

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,default=HERE.parents[2]);a=p.parse_args();print(json.dumps(validate(a.repo),sort_keys=True))
