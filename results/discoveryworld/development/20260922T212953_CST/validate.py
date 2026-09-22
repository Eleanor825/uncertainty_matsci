"""Validate exported arithmetic and provenance inventory, without scientific calls."""
import csv,hashlib,json,math
from pathlib import Path
HERE=Path(__file__).resolve().parent
def need(ok,message):
    if not ok:raise ValueError(message)
def near(a,b):return math.isclose(float(a),float(b),abs_tol=1e-12,rel_tol=1e-12)
def main():
    s=json.loads((HERE/'summary.json').read_text());p=json.loads((HERE/'provenance.json').read_text())
    rows=list(csv.DictReader((HERE/'metrics.csv').open()));episodes=list(csv.DictReader((HERE/'episode_metrics.csv').open()))
    models=list(csv.DictReader((HERE/'models_and_calibration.csv').open()))
    need(len(rows)==len(models)==8 and len(episodes)==16,'Wrong exported eight-model inventory')
    need(s['counts']['refresh_fits_completed']==8 and s['counts']['optimizer_updates']==500 and s['actual_optimizer_return_rows']==500,'Optimizer ledger differs')
    need(sum(int(r['actual_optimizer_updates'])for r in models)==500,'Model update counts differ')
    need(s['data_counts']=={'base84':84,'with_ES':96,'with_milestones':87,'combined':99,'calibration':60,'added_ES_eligible':12,'added_milestones':3},'Data counts differ')
    need(s['raw_ES_rows_retained']==20 and s['primary_variant']=='combined','Wrong raw/primary scope')
    need(all(v['passed']and all(v['checks'].values())and v['actual_final_tensor_hash']==v['expected_final_tensor_hash']for v in s['base_reproduction'].values()),'Exact reproduction failed')
    for row in rows:
        key=row['variant']+'/'+row['family'];m=s['metrics'][key]
        for name in m:
            if name in row and isinstance(m[name],(int,float)):need(near(row[name],m[name]),'CSV/source projection mismatch: '+key+'/'+name)
        ep=[e for e in episodes if e['variant']==row['variant']and e['family']==row['family']]
        need(len(ep)==2 and sum(int(e['rows'])for e in ep)==40,'Episode denominator differs')
        for field in ('brier','nll','mean_predicted_failure_probability','predicted_high_risk_fraction_at_05'):
            need(near(row[field],sum(float(e[field])*int(e['rows'])for e in ep)/40),'Episode aggregation mismatch: '+field)
        need(near(row['observed_failure_prevalence'],10/40)and near(row['FNR_at_05'],sum(int(e['false_negative_count_at_05'])for e in ep)/10),'Failure/FNR denominator differs')
    for family,variants in s['variant_minus_base'].items():
        base=s['metrics']['base84/'+family]
        for variant,deltas in variants.items():
            for name,v in deltas.items():need(near(v,s['metrics'][variant+'/'+family][name]-base[name]),'Paired model delta differs')
        for name,v in s['factorial_interaction_descriptive'][family].items():
            expected=s['metrics']['combined/'+family][name]-s['metrics']['with_ES/'+family][name]-s['metrics']['with_milestones/'+family][name]+base[name]
            need(near(v,expected),'Factorial arithmetic differs')
    for item in json.loads((HERE/'manifest.json').read_text())['files']:
        b=(HERE/item['path']).read_bytes();need(len(b)==item['bytes']and hashlib.sha256(b).hexdigest()==item['sha256'],'Export hash changed')
    need(p['raw_files_published']is False and p['publication_scientific_calls']==0,'Publication scope changed')
    print(json.dumps({'passed':True,'models':8,'optimizer_updates':500,'metric_rows':8,'episode_metric_rows':16,'exact_base_reproductions':2,'new_scientific_calls':0,'raw_prediction_recomputation_claimed':False}))
if __name__=='__main__':main()
