"""All-60 original B50 posthoc accounting and CPU numeric hull reconstruction."""
from pathlib import Path
from collections import Counter,defaultdict
import csv,gzip,hashlib,json,math,re,sys
import numpy as np
import scipy
from scipy.optimize import linprog

L=Path(__file__).resolve().parent
INPUT=L/'scientific_payload.json.gz'
INPUT_SHA='05a838b3636a4bcc2787c108c718e05b368b7d420b0afd9ab26c3bc6bad45a62'
OUT=L/'recomputed'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def dump(n,v):
    with (OUT/n).open('x')as f:json.dump(v,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
def table(n,values):
    with (OUT/n).open('x',newline='')as f:
        w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)
def composition(s):
    found=re.findall(r'([A-Z][a-z]?)([0-9]+(?:\.[0-9]+)?)',s)
    assert re.sub(r'([A-Z][a-z]?)([0-9]+(?:\.[0-9]+)?)','',s).strip()==''
    assert len({x[0]for x in found})==len(found)
    return {e:float(n)for e,n in found}
def point(c,energy,elements):
    n=sum(c.values());assert n>0 and set(c)<=set(elements)and all(v>0 for v in c.values())
    return {'x':[c.get(e,0)/n for e in elements],'y':energy/n,'composition':c}
def hull(points,x):
    r=linprog([p['y']for p in points],A_eq=np.asarray([p['x']for p in points]).T,b_eq=x,bounds=(0,None),method='highs')
    assert r.success,r.message
    return float(r.fun)
def fraction(n,d):return n/d if d else None

def stats(rows):
    c=Counter();forms=Counter();hashes=set()
    for s in rows:
        c['attempts']+=1;c['stable']+=s['stable'];c['novel']+=s['novel'];c['stable_new']+=s['stable_new']
        c['non_novel']+=not s['novel'];c['stable_non_novel']+=s['stable']and not s['novel']
        c['unstable_novel']+=not s['stable']and s['novel'];c['unstable_non_novel']+=not s['stable']and not s['novel']
        c['net_SUN_gain']+=s['SUN_delta'];c['reclassified_loss']+=s['reclassified_loss']
        c['SUN_negative_steps']+=s['SUN_delta']<0;c['SUN_downward_magnitude']+=max(0,-s['SUN_delta'])
        c['stable_new_zero_delta']+=s['stable_new']and s['SUN_delta']==0
        c['exact_hash_repeated_in_episode']+=s['candidate_hash_repeated_in_episode']
        c['non_novel_with_previous_exact_hash']+=(not s['novel'])and s['candidate_hash_repeated_in_episode']
        forms[s['reduced_formula']]+=1;hashes.add(s['candidate_hash'])
    return {k:int(c[k])for k in ('attempts','stable','novel','stable_new','non_novel','stable_non_novel',
        'unstable_novel','unstable_non_novel','net_SUN_gain','reclassified_loss','SUN_negative_steps','SUN_downward_magnitude',
        'stable_new_zero_delta','exact_hash_repeated_in_episode','non_novel_with_previous_exact_hash')} | {
        'stable_rate':fraction(c['stable'],len(rows)),'novel_rate':fraction(c['novel'],len(rows)),
        'stable_new_rate':fraction(c['stable_new'],len(rows)),'stable_given_novel':fraction(c['stable_new'],c['novel']),
        'unique_formulas':len(forms),'formula_top1_fraction':max(forms.values())/len(rows),
        'formula_hhi':sum((n/len(rows))**2 for n in forms.values())}

def main():
    assert sha(INPUT)==INPUT_SHA
    data=json.loads(gzip.decompress(INPUT.read_bytes()));assert len(data['trajectories'])==60 and data['new_oracle_calls']==data['fits']==0
    OUT.mkdir(exist_ok=False)
    allrows=[];episodes=[];losses=[];native_sun_match=0;maxerror=0.;lp_calls=0
    for index,t in enumerate(data['trajectories']):
        j=t['job'];elements=j['task_id'].split('-');native=t['initialization']['numeric_hull'];assert native is not None
        assert digest(native['entries'])==native['numeric_projection_hash']
        initial=[point(x['composition'],x['energy_eV'],elements)for x in native['entries']]
        points=list(initial);novel=[];seen=set();rows=[];cache_initial={};last_sun=0
        for s in t['steps']:
            assert s['ok']is True and s['error']is None
            o=s['observation'];r=s['oracle_result'];c=composition(r['formula']);n=sum(c.values());assert n==r['natoms']
            p=point(c,r['energy'],elements);assert abs(p['y']-r['energy_per_atom'])<1e-12
            p.update(step=s['attempt'],candidate_hash=s['candidate_hash'],formula=o['reduced_formula'])
            key=tuple(p['x'])
            if key not in cache_initial:cache_initial[key]=hull(initial,p['x']);lp_calls+=1
            previous_hull=hull(points,p['x']);lp_calls+=1
            current=max(0,p['y']-previous_hull)
            error=abs(current-o['e_above_hull']);maxerror=max(maxerror,error);assert error<1e-10
            assert o['is_stable']==(o['e_above_hull']<=.1)
            points.append(p);lowered=previous_hull-p['y']>1e-12;lost=[]
            if lowered:
                for q in novel:
                    before=q['distance'];after=max(0,q['y']-hull(points,q['x']));lp_calls+=1
                    assert after+1e-10>=before
                    if before<=.1<after:
                        entry={'job_id':j['job_id'],'method':j['method'],'task_id':j['task_id'],
                            'new_step':s['attempt'],'new_rpc_id':s['rpc_id'],'new_candidate_hash':s['candidate_hash'],
                            'new_formula':o['reduced_formula'],'new_energy_eV_per_atom':p['y'],
                            'new_local_hull_lowering_eV_per_atom':previous_hull-p['y'],
                            'lost_step':q['step'],'lost_candidate_hash':q['candidate_hash'],'lost_formula':q['formula'],
                            'lost_e_above_hull_before':before,'lost_e_above_hull_after':after,
                            'official_threshold_eV_per_atom':.1}
                        lost.append(entry);losses.append(entry)
                    q['distance']=after
            if o['is_newly_discovered']:
                p['distance']=current;novel.append(p)
            sun=sum(q['distance']<=.1 for q in novel)
            assert sun==s['SUN_after']and len(novel)==s['official_metrics']['num_newly_discovered_structures']
            assert s['SUN_before']==last_sun and s['SUN_delta']==sun-last_sun
            assert s['event_stable_new']==int(o['is_stable']and o['is_newly_discovered'])
            assert len(lost)==s['inferred_prior_stable_net_loss'];native_sun_match+=1;last_sun=sun
            assert isinstance(s['candidate_hash'],str)and len(s['candidate_hash'])==64
            row={'job_id':j['job_id'],'method':j['method'],'task_id':j['task_id'],'seed':1,'attempt':s['attempt'],
                'ten_step_band':(s['attempt']-1)//10+1,'rpc_id':s['rpc_id'],'candidate_hash':s['candidate_hash'],
                'reduced_formula':o['reduced_formula'],'composition_arity':len(c),'task_arity':len(elements),
                'uses_all_task_elements':len(c)==len(elements),'stable':o['is_stable'],'novel':o['is_newly_discovered'],
                'stable_new':s['event_stable_new'],'SUN_before':s['SUN_before'],'SUN_after':sun,'SUN_delta':s['SUN_delta'],
                'reclassified_loss':len(lost),'candidate_hash_repeated_in_episode':s['candidate_hash']in seen,
                'official_energy_eV_per_atom':p['y'],'official_e_above_hull':o['e_above_hull'],
                'LP_e_above_hull':current,'LP_abs_error':error,
                'signed_energy_minus_initial_hull_eV_per_atom':p['y']-cache_initial[key],
                'local_hull_lowering_eV_per_atom':max(0,previous_hull-p['y'])}
            seen.add(s['candidate_hash']);rows.append(row)
        assert len(rows)==50 and sun==j['SUN'];allrows+=rows
        st=stats(rows);assert st['stable_new']-st['reclassified_loss']==sun
        assert st['stable_new']==t['counts']['event_stable_new']and st['non_novel']==t['counts']['stable_not_new']+t['counts']['unstable_not_new']
        episodes.append({'job_id':j['job_id'],'task_id':j['task_id'],'method':j['method'],'seed':1,
            'official_SUN':j['SUN'],'official_AUDC':j['AUDC'],'official_mSUN':j['SUN']/50,
            'acceptance_kind':j['acceptance_evidence_kind'],**st})
        if(index+1)%10==0:print(json.dumps({'analyzed_trajectories':index+1,'LP_calls':lp_calls}),flush=True)
    methods=('baseline','esopt_graph_risk');summary={};bands=[];episode_bands=[];arity=[]
    for method in methods:
        rows=[x for x in allrows if x['method']==method];summary[method]=stats(rows)
        summary[method].update(episodes=30,official_final_SUN_sum=sum(x['official_SUN']for x in episodes if x['method']==method),
            official_AUDC_mean=sum(x['official_AUDC']for x in episodes if x['method']==method)/30)
        for band in range(1,6):bands.append({'method':method,'band':band,'first_attempt':band*10-9,'last_attempt':band*10,**stats([x for x in rows if x['ten_step_band']==band])})
        for a in sorted({x['composition_arity']for x in rows}):
            g=[x for x in rows if x['composition_arity']==a]
            arity.append({'method':method,'candidate_composition_arity':a,**stats(g)})
    for ep in episodes:
        for band in range(1,6):
            group=[x for x in allrows if x['job_id']==ep['job_id']and x['ten_step_band']==band]
            episode_bands.append({'job_id':ep['job_id'],'method':ep['method'],'task_id':ep['task_id'],'band':band,**stats(group)})
    pairs=[]
    for task in sorted({x['task_id']for x in episodes}):
        b=next(x for x in episodes if x['task_id']==task and x['method']=='baseline')
        f=next(x for x in episodes if x['task_id']==task and x['method']=='esopt_graph_risk')
        row={'task_id':task,'seed':1,'SUN_delta_full_minus_baseline':f['official_SUN']-b['official_SUN'],
            'AUDC_delta_full_minus_baseline':f['official_AUDC']-b['official_AUDC']}
        for name,e in [('baseline',b),('full',f)]:
            for k in ('job_id','official_SUN','official_AUDC','stable_new','stable','novel','non_novel','stable_non_novel',
                'unstable_novel','unstable_non_novel','reclassified_loss','unique_formulas','formula_top1_fraction',
                'exact_hash_repeated_in_episode'):row[name+'_'+k]=e[k]
        pairs.append(row)
    assert summary['baseline']['official_final_SUN_sum']==233 and summary['esopt_graph_risk']['official_final_SUN_sum']==147
    counts=Counter('win'if p['SUN_delta_full_minus_baseline']>0 else 'loss'if p['SUN_delta_full_minus_baseline']<0 else 'tie'for p in pairs)
    final={'schema':'original_seed1_all_B50_posthoc_mechanism_analysis_v1','input_sha256':INPUT_SHA,'original_observation_CST':data['CST'],
        'arms':summary,'SUN_pair_directions':dict(counts),'LP_native_SUN_agreement':native_sun_match,
        'LP_max_candidate_e_above_hull_abs_error':maxerror,'LP_calls':lp_calls,'prior_entries_reclassified':len(losses),
        'SUN_gap_decomposition':{'event_success_delta_full_minus_baseline':148-237,
            'reclassification_correction_full_minus_baseline':-(1-4),'official_SUN_delta_full_minus_baseline':147-233},
        'runtime':{'python':sys.version,'scipy':scipy.__version__,'numpy':np.__version__},
        'new_oracle_calls':0,'new_models_loaded':0,'fits':0,'test_outcomes_used_to_change_training':False,
        'metrics_unchanged':True,'scope_limit':'one original evaluation seed, time-band/chemistry observations not independent seed replicates'}
    table('steps.csv',allrows);table('episodes.csv',episodes);table('paired_systems.csv',pairs)
    table('ten_step_bands.csv',bands);table('per_episode_bands.csv',episode_bands);table('composition_arity.csv',arity)
    table('reclassified_entries.csv',losses);dump('summary.json',final)
    assert sha(INPUT)==INPUT_SHA
    dump('manifest.json',{'source':{'path':'../recompute.py','sha256':sha(__file__)},'input':{'path':str(INPUT.relative_to(L)),'sha256':INPUT_SHA},
        'files':{p.name:sha(p)for p in sorted(OUT.iterdir())if p.is_file()},'no_oracle_model_or_fit':True})
    print(json.dumps({'arms':summary,'directions':dict(counts),'LP_counts_matched':native_sun_match,'LP_max_error':maxerror,'losses':len(losses)},sort_keys=True),flush=True)

if __name__=='__main__':main()
