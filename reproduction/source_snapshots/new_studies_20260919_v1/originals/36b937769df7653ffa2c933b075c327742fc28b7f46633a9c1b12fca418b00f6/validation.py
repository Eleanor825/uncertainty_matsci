"""Read-only derived-stage validation; never a file-existence acceptance."""
from pathlib import Path
import common as c


def validate(reg,job,paths):
    c.modules(reg)
    import acceptance_repair as a
    from repair_pipeline import eligible
    out=Path(reg['output']);protocol=c.read(reg['original_protocol']['path']);ev=a.Evidence();kind=job['kind']
    if kind in ('evolve','evaluate'):return a.audit_job(reg,job)
    if kind=='bank':
        tc_r,tc_t=c.tc_modules(reg['tc_registry']['path']);tc=tc_r.read_registry(reg['tc_registry']['path'])
        from resource_bank import load
        resource_bank,resource_terminal=load(reg,tc)
        for i in range(32):
            layer=tc_t.validate_layer(tc,i);c.require(layer['status']=='succeeded','New TC layer failed actual saved fidelity')
        selected=ev.read(out/'representation_selected.json');bank=ev.read(selected['path'])
        transfer=ev.read(out/'fresh_snar_transfer_fidelity.json')
        c.require(c.artifact(selected['path'])==resource_bank,'Selected bank omitted resource-sharing provenance')
        c.require(bank['representation_training_registration']==c.artifact(Path(tc['workspace'])/'registration.json')
            and bank['fresh_all32_layers'] is True and bank['passed_layers']==32
            and selected['bank_fingerprint']==bank['bank_fingerprint']==transfer['bank_fingerprint']
            and transfer['passed'] is True and transfer['fingerprint']==selected['fidelity_fingerprint']
            and transfer['test_data_used'] is False and len(transfer['layers'])==32,'Wrong new bank/transfer admission')
        inputs=c.read(tc['input_manifest']['path'])
        for layer,row in zip(transfer['layers'],inputs['layers']):
            c.require(layer['layer_path']==row['layer_path'] and layer['source_files']=={s:[x['shard'] for x in row[s]] for s in ('train','dev')}
                and layer['passed'] is True and layer['dev']['output_fvu']<=.5
                and layer['source']['rows']=={'train':9600,'dev':6400},'Transfer inputs/actual metrics differ')
    elif kind in ('probe','graphs'):
        selected=ev.read(out/'representation_selected.json');bank=ev.read(selected['path']);base=ev.read(out/'base_policy.json')
        expected=eligible(reg)[:1] if kind=='probe' else [x for x in eligible(reg) if x['index']%8==job['chunk']]
        actual=[]
        for row in expected:
            graph=ev.read(row['graph']);actual.append(c.artifact(row['graph']))
            if graph['available']:
                saved=ev.read(row['generation']);gen=saved['generation']
                a._graph(graph['metadata'],graph['features'],ev,base['base_state_hash'],protocol,a._prefix_hash(gen['input_ids_with_completion'],True))
                c.require(graph['metadata']['graph']['graph_configuration']==protocol['graph'],'Graph configuration changed')
                references=graph['metadata']['graph']['native']['transcoders']
                c.require([(x['module_path'],x['checkpoint_hash']) for x in references]==
                    [(x['layer_path'],x['transcoder_hash']) for x in bank['layers']],'Graph uses an old/different bank')
            else:c.require(kind!='probe' and graph.get('error') and graph.get('type'),'Mandatory original failed probe did not pass')
        if kind=='graphs':
            marker=ev.read(out/'jobs'/job['job_id']/'graph_partition.json')
            c.require(marker==dict(indices=[x['index'] for x in expected],total_population=125,old_graphs_used=0,graphs=actual),
                'Graph partition is not exact eligible index modulo8')
    elif kind=='risk':
        import numpy as np
        from matdiscovery.uncertainty import CalibratedRiskModel,risk_metrics
        report=ev.read(out/'risk_fit.json');ev.raw(out/'snar_risk.pt');risk=CalibratedRiskModel.load(out/'snar_risk.pt')
        rows={'train':[],'dev':[]};total=0
        for item in eligible(reg):
            graph=ev.read(item['graph'])
            if graph['available']:
                total+=1;rows[item['split']].append(dict(features=graph['features'],label=item['label'],episode=item['episode'],
                    query_path=str(item['query']),graph_path=str(item['graph']),prefix_hash=item['prefix_hash']))
        c.require(total/125>=.9 and report['graph_fraction']==total/125 and report['rows']==rows and report['test_used'] is False,
                  'Risk used different graphs/labels/availability population')
        names=sorted({k for x in rows['train'] for k in x['features']})
        c.require(report['feature_names']==names and risk.provenance==report['provenance']
            and not ({x['episode'] for x in rows['train']} & {x['episode'] for x in rows['dev']})
            and not ({x['prefix_hash'] for x in rows['train']} & {x['prefix_hash'] for x in rows['dev']}),'Risk source/split/schema differs')
        x=np.array([[row['features'].get(k,float('nan')) for k in names] for row in rows['dev']]);y=np.array([row['label'] for row in rows['dev']])
        prediction=risk.predict_proba(x,names)
        c.require(a._same(report['development_metrics'],risk_metrics(y,prediction)), 'Serialized NN calibration/metrics differ')
        expected=float((np.sum(prediction>=.5)+50-len(rows['dev']))/50)
        c.require(report['random_controller_retry_probability']==expected,'Random-controller matched dev rate differs')
    elif kind=='freeze':
        archive=[];expected=set(a.expected_episodes(protocol));expected={x for x in expected if not x.startswith('test_')}
        folders={x.name for x in (out/'episodes').iterdir() if x.is_dir()}
        c.require(folders==expected,'Cannot freeze incomplete adaptation or any test rows')
        for name in sorted(expected):
            ev.read(out/'episodes'/name/'summary.json')
            archive.extend(ev.read(x) for x in sorted((out/'episodes'/name).glob('query[0-9][0-9][0-9].json')))
        prior=ev.read(out/'shared_prior_archive.json');complete=ev.read(out/'adaptation_complete.json')
        c.require(len(prior)==550 and prior==archive and complete['queries']==550 and complete['archive_fingerprint']==c.digest(prior),
                  'Shared prior is not the exact complete550 pre-test rows')
        c.require(complete['selection']=={b:ev.read(out/f'{b}_checkpoint_selection.json') for b in ('full','es_only')},'Prior changed selected checkpoints')
    elif kind=='report':
        report=ev.read(out/'acceptance.json');c.require(report['passed'] is True and report['complete'] is True
            and report['unique_physical_attempts']==2300 and report['imported_collection_oracle_calls']==150
            and report['new_adaptation_oracle_calls']==400 and report['new_test_oracle_calls']==1750,'Incomplete main acceptance')
    else:raise ValueError('Unknown derived kind')
    for path in paths:ev.raw(path)
    value=dict(schema='snar_repair_derived_validation_v2',passed=True,job_id=job['job_id'],new_scientific_calls_by_audit=0,
               evidence_files=sorted(ev.files.values(),key=lambda x:x['path']))
    value['fingerprint']=c.digest(value);return value
