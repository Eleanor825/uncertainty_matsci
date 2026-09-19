"""UQ-only at theta0; independently trained ES-only with no risk/controller."""
from dataclasses import dataclass
from pathlib import Path
from matdiscovery.accounting import file_sha256,fingerprint
from matdiscovery.component_scope import ARMS,SCHEMA,admitted_parent,parent_sweep,require
from matdiscovery.esopt import tensor_state_hash
from matdiscovery.rollouts import DiscoveryRollout


@dataclass
class ArmRuntime:
    profile:dict
    risk:object
    graph:object
    failure_aware_control:bool


def selected_generation(profile,arm):
    return 0 if arm=='baseline_reference' else profile['selected_generation']


def validate_profile(profile,reg,*,arm_id=None,verify_training=True):
    parent=parent_sweep(reg['parent']);base=parent['execution_profiles']['baseline'];arm=arm_id or profile.get('arm_id')
    if arm=='baseline_reference':
        require(profile==base==reg['baseline_reference_profile'],'Shared baseline is not byte-equivalent to the original baseline profile')
        require(profile['actual_model_state_hash']==profile['initial_actual_model_state_hash'] and profile['selected_es'] is None,'Baseline is not theta0')
        for path,expected in profile['artifacts'].items():require(file_sha256(path)==expected,'Baseline logging/source artifact changed')
        return profile
    require(arm in ARMS and profile['arm_id']==arm and profile['schema']==SCHEMA and profile['component_fingerprint']==reg['fingerprint'],'Wrong component profile')
    require(profile['initial_actual_model_state_hash']==base['actual_model_state_hash']==base['initial_actual_model_state_hash']
            and profile['policy_configuration_fingerprint']==base['policy_configuration_fingerprint']
            and profile['initial_checkpoint_manifest_hash']==base['initial_checkpoint_manifest_hash'],'Paired original theta0/runtime changed')
    require(profile['mace_num_workers']==4 and profile['orb_num_workers']==1 and profile['control_route']==ARMS[arm],'Evaluator/control route changed')
    if arm=='uq_only_schema_fixed':
        require(profile['actual_model_state_hash']==base['actual_model_state_hash'] and profile['selected_generation']==0
                and profile['selected_es'] is None and profile['controller_enabled'] is True and profile['graph_enabled'] is True
                and profile['schema_patch_sha256']==reg['schema_patch_sha256'],'UQ-only is not schema-fixed theta0')
    else:
        require(profile['controller_enabled'] is False and profile['graph_enabled'] is False and profile['selected_es']['independent_es_only'] is True,'ES-only reused a risk controller or old full G2')
        if verify_training:
            from es_only_training import verify_es_only_receipt
            selected=verify_es_only_receipt(registration=reg)
            require(profile['selected_es']==selected and profile['actual_model_state_hash']==selected['actual_model_state_hash']
                    and profile['selected_generation']==selected['selected_generation'],'Unverified independent ES checkpoint')
    for path,expected in profile['artifacts'].items():require(file_sha256(path)==expected,'Component auxiliary/checkpoint artifact changed')
    return profile


def prepare_arm(provider,policy,arm,base_state,base_hash,reg,*,independent_loader=None):
    admitted_parent(reg);require(arm in ARMS,'Unregistered component arm')
    parent=parent_sweep(reg['parent']);base=parent['execution_profiles']['baseline']
    require(base_hash==base['actual_model_state_hash'],'Wrong original theta0')
    if arm=='baseline_reference':
        profile,risk,graph=provider.profile(policy,{'method':'baseline'},base_state,base_hash)
        require(graph is None,'Original baseline unexpectedly has a graph controller')
        validate_profile(profile,reg,arm_id=arm)
        # Preserve original measurement-only heads; the baseline route never
        # invokes the schema controller or selects/retries by risk prediction.
        return ArmRuntime(profile,risk,None,True)
    policy.model.load_state_dict(base_state,strict=True);policy.mark_state('reload',generation=0)
    require(tensor_state_hash(dict(policy.model.state_dict()))==base_hash,'Theta0 restoration failed')
    selected=None;risk=None;graph=None;files={}
    if arm=='es_only_independent':
        from es_only_training import load_es_only_selected
        selected=(independent_loader or load_es_only_selected)(policy,registration=reg)
        require(selected.get('independent_es_only') is True and selected.get('risk_used') is False,'Old full G2 is not independent ES-only')
        files.update(selected['artifacts'])
    else:
        from matdiscovery.training_jobs import load_frozen_controllers
        risk,graph,aux=load_frozen_controllers(policy,model_key='qwen35_4b',benchmark='made',method='graph_risk',
            risk_checkpoint=provider.risk_root/'graph_risk.pt',transcoder_manifest=provider.transcoder_manifest,
            graph_config=provider.protocol['graph'],device=provider.runtime['device'],
            failure_control=provider.protocol['failure_control'],corpus_contract=provider.corpus)
        files.update({str(Path(p).resolve()):file_sha256(p) for p in aux})
    profile={'schema':SCHEMA,'component_fingerprint':reg['fingerprint'],'arm_id':arm,'control_route':ARMS[arm],
        'model_key':'qwen35_4b','benchmark':'made','actual_model_state_hash':tensor_state_hash(dict(policy.model.state_dict())),
        'initial_actual_model_state_hash':base_hash,'initial_checkpoint_manifest_hash':policy.checkpoint_hash,
        'policy_configuration_fingerprint':policy.configuration_fingerprint,'execution_inputs_fingerprint':fingerprint(provider.execution),
        'mace_num_workers':4,'orb_num_workers':1,'selected_es':selected,
        'selected_generation':selected['selected_generation'] if selected else 0,
        'controller_enabled':arm=='uq_only_schema_fixed','graph_enabled':arm=='uq_only_schema_fixed',
        'schema_patch_sha256':reg['schema_patch_sha256'],'artifacts':files,'auxiliary_modules_attached_to_policy':False}
    validate_profile(profile,reg,arm_id=arm)
    return ArmRuntime(profile,risk,graph,arm=='uq_only_schema_fixed')


class ComponentRollout(DiscoveryRollout):
    def _generate(self,job,*args,**kwargs):
        arm=job['arm_id'];require(job['method']==arm and job['control_route']==ARMS[arm],'Component control route changed')
        action,row=super()._generate({**job,'method':ARMS[arm]},*args,**kwargs)
        rows=args[4] if len(args)>4 else kwargs.get('rows',[])
        for candidate in rows:
            candidate['component_arm']=arm;candidate['registered_control_route']=ARMS[arm]
        return action,row

    def run(self,job,output,*,collection=False):
        require(collection is False,'No test fitting/collection through component evaluation')
        measurements_enabled=job['arm_id'] in ('baseline_reference','uq_only_schema_fixed')
        require(self.settings.failure_aware_control==measurements_enabled,'Wrong original failure-measurement setting')
        if job['arm_id']=='es_only_independent':require(self.risk_model is None and self.attributor is None,'ES-only must be entirely risk/graph free')
        if job['arm_id']=='baseline_reference':require(self.attributor is None,'Baseline cannot invoke a graph controller')
        return super().run(job,output,collection=False)
