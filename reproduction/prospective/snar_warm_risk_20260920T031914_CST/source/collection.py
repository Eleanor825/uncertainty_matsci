"""New warm train/dev identities; original G0 actions/oracle and diagnostic graphs."""
from pathlib import Path
from types import SimpleNamespace
from dataclasses import asdict
import inspect,math,os,socket,time,traceback
import warm_common as w

PRIOR_LINE="p=prior if spec.get('role')=='test' else []"
PRIOR_REPLACEMENT="p=prior  # Explicit newly registered warm train/dev scope; not an old test identity."

def warm_auditor(original):
    """Only archive dispatch differs; original token/physical/summary math remains."""
    source=inspect.getsource(original.audit_episodes)
    w.require(source.count(PRIOR_LINE)==1,'Original audit prior branch changed')
    namespace=dict(original.__dict__)
    exec(compile(source.replace(PRIOR_LINE,PRIOR_REPLACEMENT),str(original.__file__)+'::warm_scope','exec'),namespace)
    return namespace['audit_episodes']

def catalog(reg):
    return {j['job_id']:{'name':j['job_id'],**{k:j[k] for k in ('arm','seed','budget','scope','role')}} for j in reg['jobs']}

def audit_collection(reg,job):
    q,c,parent,tc=w.helpers();c.modules(parent)
    audit=c.load_module('_snar_warm_private_acceptance',w.MAIN/'acceptance_repair.py')
    protocol=w.read(w.check(reg['parent_protocol']));expected=catalog(reg)
    def exact_catalog(value):w.require(value==protocol,'Unexpected behavior protocol');return expected
    audit.expected_episodes=exact_catalog
    out=Path(reg['output'])/'jobs'/job['job_id']/'science';ev=audit.Evidence()
    locks=[x for x in parent['sources'] if Path(x['path']).name=='source_lock.json'];w.require(len(locks)==1,'Official source lock ambiguous')
    source=ev.read(w.check(locks[0]));selected={job['job_id']:expected[job['job_id']]}
    physical,counts=audit._oracle_evidence(out,protocol,selected,ev,source,locks[0]['sha256'])
    prior=ev.read(w.check(reg['assets']['shared_prior_archive.json']));base=ev.read(out/'base_policy.json')
    summaries,queries,_=warm_auditor(audit)(out,protocol,selected,physical,ev,prior,base)
    w.require(len(physical)==len(queries)==30 and base['base_state_hash']==reg['base_state_hash'],'Wrong physical count/base policy')
    actual=ev.read(out/'executed_policy_states'/f"{job['job_id']}.json")
    w.require(actual['actual_weight_hash']==base['base_state_hash'] and actual['computed_from_actual_parameter_tensors'] is True and actual['model_stamp']['generation']==0,'Actual G0 execution proof differs')
    for prefix in (out/'episodes'/job['job_id']).glob('query*_proposal*'):
        candidate=ev.read(prefix/'candidate.json');saved=ev.read(prefix/'generation.json')
        w.require(candidate['risk'] is None and candidate['features'] is None,'Diagnostic graph leaked into action scoring')
        if candidate['success']:
            graph=ev.read(prefix/'warm_graph.json')
            w.require(graph['schema']=='snar_warm_diagnostic_graph_v1' and graph['generation']==w.artifact(prefix/'generation.json'),
                      'Graph/generation provenance differs')
            if graph['available']:
                audit._graph(graph['metadata'],graph['features'],ev,base['base_state_hash'],protocol,
                             audit._prefix_hash(saved['generation']['input_ids_with_completion'],True))
    rows=[]
    for query in sorted(queries.values(),key=lambda x:x['query_id']):
        item={'query_id':query['query_id'],'episode':job['job_id'],'role':job['role'],'seed':job['seed'],
              'label':query['no_hvi'],'hv_increment':query['hv_increment'],'features':None,'prefix_hash':None,
              'included':False,'reason':'lhs' if query['kind']=='paired_lhs_initial_design' else 'invalid_fallback',
              'query':w.artifact(out/'episodes'/job['job_id']/('query'+query['query_id'].rsplit('q',1)[1]+'.json'))}
        if query['chosen_generation']:
            generation=Path(query['chosen_generation']);graph=ev.read(generation.parent/'warm_graph.json');saved=ev.read(generation)
            item.update(generation=w.artifact(generation),graph=w.artifact(generation.parent/'warm_graph.json'),
                        prefix_hash=saved['generation']['prefix_hash'],reason='graph_unavailable')
            if graph['available']:
                values=graph['features'];w.require(len(values)==285 and all(math.isfinite(float(x)) for x in values.values()),'Expected complete finite285 features')
                item.update(features=values,included=True,reason='included')
        rows.append(item)
    return {'passed':True,'physical_calls':30,'oracle_counts':counts,'rows':rows,'summary':summaries[job['job_id']],
            'evidence_files':sorted(ev.files.values(),key=lambda x:x['path']),
            'audit_adapter':'one explicit prior dispatch change; original oracle/token/label/HV/graph checks unchanged'}

def runtime(reg,path,gpu):
    old=w.read(w.check(reg['parent_runtime']));actual=w.read(path)
    w.require(set(actual)==set(old) and all(actual[k]==v for k,v in old.items() if k!='gpu_uuid') and actual['gpu_uuid']==gpu,
              'Only GPU UUID placement may change')
    import sys
    w.require(Path(sys.executable).resolve()==Path(actual['policy_python']).resolve(),'Wrong policy interpreter')
    return actual

def execute(reg,jid,runtime_path,gpu,fds):
    q,c,parent,tc=w.helpers();job=next(j for j in reg['jobs'] if j['job_id']==jid);rt=runtime(reg,runtime_path,gpu)
    q.lease_proof(tc,gpu,fds)
    guard=c.load_module('_snar_warm_original_resource_guard',Path(parent['original_source_root'])/'runtime/resource_guard.py')
    w.require(not guard.admission_errors(guard.snapshot(gpu),socket.gethostname()),'Original SnAr resource admission failed')
    directory=Path(reg['output'])/'jobs'/jid;directory.mkdir(parents=True,exist_ok=False)
    w.publish(directory/'claim.json',w.seal({'schema':w.SCHEMA,'registration_fingerprint':reg['fingerprint'],'job':job,
        'runtime':w.artifact(runtime_path),'pid':os.getpid(),'hostname':socket.gethostname(),'gpu_uuid':gpu,
        'leases':q.lease_proof(tc,gpu,fds),'automatic_replay':False,'started_at':time.time()}))
    out=directory/'science';out.mkdir();base=c.modules(parent);p=None;completed=False;started=time.time()
    class WarmPipeline(base.Pipeline):
        def __init__(self):
            self.args=SimpleNamespace(stage='warm_collect',arm='qwen_base',branch=None);self.out=out
            self.protocol=w.read(w.check(reg['parent_protocol']));self.runtime=rt;self.event_path=out/'events.jsonl'
            self.oracle=None;self.policy=self.adapter=self.optimizer=self.bank=self.attributor=self.risk=None
            self.graph_states=set();self.feature_names=[];self.weight_hash=None
            self.load_policy();self.load_bank()
            self.oracle=base.Oracle(rt,out,job['scope'],30)
        def load_bank(self):
            from method.features import load_candidate_bank,make_attributor
            from matdiscovery.representation_training import GraphStageConfig
            selected=w.read(w.check(reg['assets']['representation_selected.json']));report=w.read(w.check(reg['assets']['fresh_snar_transfer_fidelity.json']))
            self.graph_config=GraphStageConfig(**self.protocol['graph']);self.bank=load_candidate_bank(selected['path'],self.policy)
            w.require(self.bank.bank_fingerprint==selected['bank_fingerprint']==report['bank_fingerprint'] and report['passed'] is True
                and report['fingerprint']==selected['fidelity_fingerprint']==w.digest({k:v for k,v in report.items() if k!='fingerprint'}),'Original actual TC transfer proof differs')
            self.bank.transfer_report=report;self.attributor=make_attributor(self.policy,self.bank,config=self.graph_config)
        def candidate(self,path,observation,history,prior,seed,**kwargs):
            w.require(not kwargs.get('guided') and not kwargs.get('capture_split'),'Warm collection cannot use NN controller or old collection identity')
            result=super().candidate(path,observation,history,prior,seed,**kwargs)
            if result['success']:
                path=Path(path);saved=w.read(path/'generation.json')
                try:
                    features,metadata=self.graph(self.hydrate(saved['generation']),path)
                    graph={'available':True,'features':features,'metadata':metadata}
                except Exception as exc:graph={'available':False,'error_type':type(exc).__name__,'error':str(exc)}
                w.publish(path/'warm_graph.json',dict(graph,schema='snar_warm_diagnostic_graph_v1',generation=w.artifact(path/'generation.json')))
            return result # Original candidate/risk remains unchanged; graph never selects/retries.
    try:
        p=WarmPipeline.__new__(WarmPipeline);p.__init__();initial=p.optimizer.model_state_hash();w.require(initial==p.weight_hash==reg['base_state_hash'],'Not originalG0')
        w.publish(out/'executed_policy_states'/f'{jid}.json',{'episode':jid,'actual_weight_hash':initial,'model_stamp':asdict(p.policy.model_stamp),'computed_from_actual_parameter_tensors':True})
        prior=w.read(w.check(reg['assets']['shared_prior_archive.json']));w.require(len(prior)==550 and w.digest(prior)==reg['prior_fingerprint'],'Prior changed')
        p.episode(jid,arm='qwen_base',seed=job['seed'],budget=30,prior=prior)
        w.require(p.optimizer.model_state_hash()==initial,'Collection changed original policy weights')
        p.oracle.close(completed=True);p.oracle=None;completed=True
        audit=audit_collection(reg,job);w.publish(directory/'collection_audit.json',audit)
        receipt=w.seal({'schema':w.SCHEMA,'complete':True,'registration_fingerprint':reg['fingerprint'],'job':job,
            'physical_calls':30,'actual_weight_hash':initial,'policy_generation':0,'online_NN_used':False,
            'audit':w.artifact(directory/'collection_audit.json'),'evidence_files':audit['evidence_files'],
            'elapsed_seconds':time.time()-started,'gpu_closed_claimed':False})
        w.publish(directory/'receipt.json',receipt);return receipt
    except BaseException as exc:
        w.publish(directory/'failure.json',{'registration_fingerprint':reg['fingerprint'],'job_id':jid,'error_type':type(exc).__name__,
            'error':str(exc),'traceback':traceback.format_exc(),'automatic_replay':False});raise
    finally:
        if p and p.oracle:p.oracle.close(completed=completed)
