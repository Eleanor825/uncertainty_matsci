"""CPU contracts only. Tiny synthetic evidence is never a scientific result."""
import ast,copy,hashlib,importlib.util,json,os,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from contextlib import contextmanager
from types import SimpleNamespace
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));os.environ['CUDA_VISIBLE_DEVICES']=''
import warm_common as w
import collection as col
import engine
import runner


def toy():
    import numpy as np
    rng=np.random.default_rng(113);names=[f'hidden.toy_{i}' for i in range(12)];rows=[]
    for role,groups in [('train',3),('dev',2)]:
        for g in range(groups):
            for i in range(6):
                rows.append({'query_id':f'{role}{g}/q{i:03d}','episode':f'{role}{g}','role':role,'seed':g,'label':i%2,
                    'features':dict(zip(names,rng.normal(size=len(names)).tolist())),'prefix_hash':f'{role}-{g}-{i}','included':True,'reason':'included'})
    return {'rows':rows,'feature_names':names},json.loads((HERE/'plan.json').read_text())

class Contracts(unittest.TestCase):
    def test_exact_catalog_new_scope_150(self):
        p=json.loads((HERE/'plan.json').read_text());j=w.job_catalog(p)
        self.assertEqual(len(j),5);self.assertEqual(sum(x['budget'] for x in j),150)
        self.assertEqual([x['role'] for x in j],['train']*3+['dev']*2)
        self.assertEqual({x['prior_rows'] for x in j},{550});self.assertEqual(len({x['seed'] for x in j}),5)
        self.assertTrue(all(not x['job_id'].startswith(('test_','collection_')) for x in j))

    def test_data_insufficient_zero_models(self):
        data,plan=toy()
        for r in data['rows']:r['label']=1
        with tempfile.TemporaryDirectory() as d:
            result=engine.run(data,plan,d)
            self.assertEqual(result['status'],'data_insufficient');self.assertEqual(result['NN_fits'],0)
            self.assertFalse(list(Path(d).glob('*.pt')));self.assertTrue(engine.audit(data,plan,d)['passed'])

    def test_graph_missing_retained_and_blocks(self):
        data,plan=toy()
        for r in data['rows'][:10]:r.update(included=False,features=None,reason='graph_unavailable')
        status,_,_=engine.eligible(data)
        self.assertFalse(status['ready']);self.assertIn('native_graph_fraction_below_0.9',status['reasons'])
        self.assertEqual(status['counts']['train']['reasons']['graph_unavailable'],10)

    def test_pending_does_not_fit_or_claim(self):
        with tempfile.TemporaryDirectory() as d:
            reg={'output':d,'jobs':[{'job_id':'missing'}]}
            with patch.object(engine,'run',side_effect=AssertionError('fit happened')):
                result=runner.fit(reg)
            self.assertEqual(result['status'],'pending');self.assertFalse((Path(d)/'nn').exists())

    def test_error_observation_is_not_missing(self):
        with patch.object(Path,'lstat',side_effect=OSError('I/O error')):
            with self.assertRaises(OSError):w.present('/unknown')

    def test_atomic_publication_no_replay(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'claim.json';w.publish(p,{'v':1})
            with self.assertRaises(FileExistsError):w.publish(p,{'v':2})
            self.assertEqual(w.read(p),{'v':1})

    def test_failed_dispatch_is_not_replayed_by_next_worker(self):
        # Resource and Popen are mocks; actual persistent claim publication is real.
        @contextmanager
        def lock(path):
            path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('a+') as f:yield f
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);runtime=out/'runtime.json';runtime.write_text(json.dumps({'policy_python':sys.executable,'gpu_uuid':'old'}))
            (out/'registration.json').write_text('{}')
            reg={'output':str(out),'fingerprint':'toy','jobs':[{'job_id':'one'}],'parent_runtime':w.artifact(runtime)}
            calls=[]
            q=SimpleNamespace(lease_proof=lambda *args:{'mock':True},try_lock=lock,
                launch_and_drain=lambda *args:(calls.append(args[0]) or {'returncode':1}))
            guard=SimpleNamespace(snapshot=lambda gpu:{},admission_errors=lambda *args:[])
            c=SimpleNamespace(load_module=lambda *args:guard)
            with patch.object(w,'helpers',return_value=(q,c,{'original_source_root':d},{})):
                first=runner.worker(reg,'first','GPU-toy',(101,102))
                self.assertEqual(first['status'],'failed');self.assertEqual(len(calls),1)
                second=runner.worker(reg,'second','GPU-toy',(101,102))
                self.assertEqual(second['visited'],[]);self.assertEqual(len(calls),1)
            self.assertTrue((out/'dispatch_claims/one.json').is_file())

    def test_source_warm_scope_is_exact_one_line(self):
        # Real original acceptance source, no mock replacement of its checks.
        original=HERE.parents[1]/'benchmark_extensions/summit_snar_main_repair_20260919_v2/acceptance_repair.py'
        source=original.read_text();tree=ast.parse(source);node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='audit_episodes')
        block=ast.get_source_segment(source,node)
        self.assertEqual(block.count(col.PRIOR_LINE),1)
        warm=block.replace(col.PRIOR_LINE,col.PRIOR_REPLACEMENT)
        a=ast.parse(block);b=ast.parse(warm)
        differences=[]
        def compare(x,y,path=''):
            if type(x)!=type(y):differences.append(path);return
            if isinstance(x,ast.AST):
                for field in x._fields:compare(getattr(x,field),getattr(y,field),path+'.'+field)
            elif isinstance(x,list):
                self.assertEqual(len(x),len(y))
                for i,(xx,yy) in enumerate(zip(x,y)):compare(xx,yy,path+f'[{i}]')
            elif x!=y:differences.append(path)
        compare(a,b);self.assertEqual(len(differences),1);self.assertTrue(differences[0].endswith('.value'))

    def test_real_original_audit_accepts_warm_train_without_test_role(self):
        base=HERE.parents[1]/'benchmark_extensions'
        def load(name,path):
            spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
        old=sys.modules.get('study')
        try:
            study=load('study',base/'summit_snar_20260918/study.py')
            a=load('_warm_test_real_acceptance',base/'summit_snar_main_repair_20260919_v2/acceptance_repair.py')
            with tempfile.TemporaryDirectory() as d:
                root=Path(d);name='toy_new_warm_train';folder=root/'episodes'/name;folder.mkdir(parents=True)
                proto={'policy':{'history_window':6},'objective':{'non_improvement_tolerance':1e-12}}
                prior=[{'parameters':study.lhs(9,1)[0],'objectives':{'sty':13000.,'e_factor':0.}}]
                spec={'name':name,'role':'train','arm':'qwen_base','seed':6101,'budget':1,'scope':'toy_new_warm_scope'}
                header={k:spec[k] for k in ('name','arm','seed','budget')};header.update(protocol_fingerprint=study.digest(proto),prior_fingerprint=study.digest(prior),weight_hash='base')
                qid=name+'/q000';parameters=study.lhs(6101,5)[0];objectives={'sty':6500.,'e_factor':500.}
                terminal={'query_id':qid,'parameters':parameters,'objectives':objectives,'cache_hit':False,'physical_oracle_calls_this_request':1}
                row={'query_id':qid,'parameters':parameters,'objectives':objectives,'oracle_receipt':terminal,'kind':'paired_lhs_initial_design',
                     'risk':None,'proposal_count':0,'invalid_proposals':0,'chosen_generation':None,'hv_increment':0.,'no_hvi':1}
                intent={'query_id':qid,'parameters':parameters,'prior_and_history_hv':study.hypervolume(prior),'chosen_generation':None}
                for file,value in [('episode.json',header),('query000.json',row),('query000_intent.json',intent),('summary.json',{**header,**study.episode_summary([row],prior)})]:
                    (folder/file).write_text(json.dumps(value))
                physical={qid:{'scope':spec['scope'],'result':terminal}}
                with self.assertRaisesRegex(a.AcceptanceError,'scope/prior'):a.audit_episodes(root,proto,{name:spec},physical,a.Evidence(),prior,{'base_state_hash':'base'})
                summaries,rows,_=col.warm_auditor(a)(root,proto,{name:spec},physical,a.Evidence(),prior,{'base_state_hash':'base'})
                self.assertEqual(rows[qid]['no_hvi'],1);self.assertEqual(summaries[name]['prior_rows'],1)
                row['no_hvi']=0;(folder/'query000.json').write_text(json.dumps(row))
                with self.assertRaisesRegex(a.AcceptanceError,'risk label'):col.warm_auditor(a)(root,proto,{name:spec},physical,a.Evidence(),prior,{'base_state_hash':'base'})
        finally:
            if old is None:sys.modules.pop('study',None)
            else:sys.modules['study']=old

    def test_real_four_fits_and_exact_audit(self):
        data,plan=toy()
        with tempfile.TemporaryDirectory() as d:
            result=engine.run(data,plan,d)
            self.assertEqual(result['NN_fits'],4);self.assertLessEqual(result['optimizer_steps'],400)
            self.assertGreater(result['optimizer_steps'],300)
            proof=engine.audit(data,plan,d);self.assertTrue(proof['passed']);self.assertEqual(proof['saved_epoch_states_checked'],300)
            selection=w.read(Path(d)/'training_selection.json')
            self.assertEqual(len(selection['OOF']),18)
            self.assertTrue(all(x['query_id'].startswith('train') for x in selection['OOF']))
            self.assertFalse(result['independent_development_used_for_selection_or_calibration'])
            # Tampering is rejected; no refit/replay is attempted.
            p=Path(d)/'report.json';v=w.read(p);v['development']['calibrated']['brier']+=.01;p.write_text(json.dumps(v))
            with self.assertRaisesRegex(ValueError,'exactly recompute'):engine.audit(data,plan,d)

    def test_development_does_not_enter_training_choice(self):
        data,plan=toy();ready,train,dev=engine.eligible(data);config=engine.cfg(plan)
        folds=[]
        for group in sorted({r['episode'] for r in train}):
            folds.append(engine.fit_path([r for r in train if r['episode']!=group],[r for r in train if r['episode']==group],data['feature_names'],config,3))
        before=engine.choose(folds,train,3)
        for row in dev:row['label']=1-row['label'];row['features']={k:999999 for k in row['features']}
        self.assertEqual(before,engine.choose(folds,train,3))

if __name__=='__main__':unittest.main(verbosity=2)
