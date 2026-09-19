"""The same support-aware controller at theta0 and at newly evolved weights."""
from dataclasses import dataclass
from pathlib import Path
from matdiscovery.accounting import file_sha256, fingerprint
from matdiscovery.support_scope import ARMS, SCHEMA, admitted_parent, parent_sweep, require
from matdiscovery.support_control import CONTRACT
from matdiscovery.esopt import tensor_state_hash
from matdiscovery.rollouts import DiscoveryRollout

@dataclass
class ArmRuntime:
    profile: dict
    risk: object
    graph: object
    failure_aware_control: bool = True

def selected_generation(profile,arm):
    require(arm in ARMS,'Not a support-aware arm')
    return profile['selected_generation']

def validate_profile(profile,reg,*,arm_id=None,verify_training=True):
    base=parent_sweep(reg['parent'])['execution_profiles']['baseline'];arm=arm_id or profile.get('arm_id')
    require(arm in ARMS and profile['arm_id']==arm and profile['schema']==SCHEMA
            and profile['support_fingerprint']==reg['fingerprint'],'Wrong support-aware profile')
    require(profile['initial_actual_model_state_hash']==base['actual_model_state_hash']==base['initial_actual_model_state_hash']
            and profile['policy_configuration_fingerprint']==base['policy_configuration_fingerprint']
            and profile['initial_checkpoint_manifest_hash']==base['initial_checkpoint_manifest_hash'],'Original theta0/runtime changed')
    require(profile['mace_num_workers']==4 and profile['orb_num_workers']==1 and profile['control_route']==ARMS[arm]
            and profile['controller_enabled'] is True and profile['graph_enabled'] is True
            and profile['support_contract']==CONTRACT and profile['schema_patch_sha256']==reg['schema_patch_sha256'],
            'Evaluator, support contract, or schema implementation changed')
    if arm=='uq_only_support_aware':
        require(profile['actual_model_state_hash']==base['actual_model_state_hash'] and profile['selected_generation']==0
                and profile['selected_es'] is None,'UQ-only is not original theta0')
    else:
        selected=profile['selected_es']
        require(selected.get('support_aware_full_es') is True and selected.get('risk_used') is True
                and selected.get('support_contract')==CONTRACT,'Old G2 is not newly trained support-aware full ES')
        if verify_training:
            from full_es_training import verify_full_es_receipt
            require(selected==verify_full_es_receipt(registration=reg),'Independent full ES receipt differs')
        require(profile['actual_model_state_hash']==selected['actual_model_state_hash']
                and profile['selected_generation']==selected['selected_generation'],'Wrong selected weights')
    for item in reg.get('frozen_NN_artifacts',[]):
        require(profile['artifacts'].get(item['path'])==item['sha256'],'Profile omitted or changed the original NN')
    for path,expected in profile['artifacts'].items():
        require(file_sha256(path)==expected,'Controller/checkpoint artifact changed')
    return profile

def prepare_arm(provider,policy,arm,base_state,base_hash,reg,*,independent_loader=None,controller_loader=None):
    admitted_parent(reg);require(arm in ARMS,'Unregistered support-aware arm')
    base=parent_sweep(reg['parent'])['execution_profiles']['baseline']
    require(base_hash==base['actual_model_state_hash'],'Wrong original theta0')
    policy.model.load_state_dict(base_state,strict=True);policy.mark_state('reload',generation=0)
    require(tensor_state_hash(dict(policy.model.state_dict()))==base_hash,'Theta0 restoration failed')
    selected=None;files={}
    if arm=='full_support_aware':
        from full_es_training import load_full_es_selected
        selected=(independent_loader or load_full_es_selected)(policy,registration=reg)
        require(selected.get('support_aware_full_es') is True and selected.get('support_contract')==CONTRACT,
                'Old full G2 cannot be relabelled as the new method')
        files.update(selected['artifacts'])
    from matdiscovery.training_jobs import load_frozen_controllers
    risk,graph,aux=(controller_loader or load_frozen_controllers)(policy,model_key='qwen35_4b',benchmark='made',method=ARMS[arm],
        risk_checkpoint=provider.risk_root/'graph_risk.pt',transcoder_manifest=provider.transcoder_manifest,
        graph_config=provider.protocol['graph'],device=provider.runtime['device'],
        failure_control=provider.protocol['failure_control'],corpus_contract=provider.corpus)
    files.update({str(Path(p).resolve()):file_sha256(p) for p in aux})
    profile={'schema':SCHEMA,'support_fingerprint':reg['fingerprint'],'arm_id':arm,'control_route':ARMS[arm],
        'model_key':'qwen35_4b','benchmark':'made','actual_model_state_hash':tensor_state_hash(dict(policy.model.state_dict())),
        'initial_actual_model_state_hash':base_hash,'initial_checkpoint_manifest_hash':policy.checkpoint_hash,
        'policy_configuration_fingerprint':policy.configuration_fingerprint,'execution_inputs_fingerprint':fingerprint(provider.execution),
        'mace_num_workers':4,'orb_num_workers':1,'selected_es':selected,
        'selected_generation':selected['selected_generation'] if selected else 0,
        'controller_enabled':True,'graph_enabled':True,'support_contract':CONTRACT,
        'schema_patch_sha256':reg['schema_patch_sha256'],'artifacts':files,'auxiliary_modules_attached_to_policy':False}
    validate_profile(profile,reg,arm_id=arm)
    return ArmRuntime(profile,risk,graph)

class SupportAwareRollout(DiscoveryRollout):
    def __init__(self,*args,**kwargs):
        require('support_aware' not in kwargs,'Support mode is owned by this registered route')
        super().__init__(*args,support_aware=True,**kwargs)

    def _generate(self,job,*args,**kwargs):
        arm=job.get('arm_id')
        if arm is not None:
            require(arm in ARMS and job['method']==arm and job['control_route']==ARMS[arm],'Support route changed')
            actual={**job,'method':ARMS[arm]}
        else:
            require(job['method']=='esopt_graph_risk' and job.get('ablation_training_scope')=='support_aware_full_es',
                    'Only registered new full-ES training may omit the evaluation arm')
            actual=job
        action,row=super()._generate(actual,*args,**kwargs)
        for candidate in (args[4] if len(args)>4 else kwargs.get('rows',[])):
            candidate['support_arm']=arm or 'full_ES_training'
            candidate['registered_control_route']=actual['method']
        return action,row

    def run(self,job,output,*,collection=False):
        require(collection is False and self.settings.failure_aware_control is True
                and self.risk_model is not None and self.attributor is not None,'Complete support-aware method required')
        return super().run(job,output,collection=False)
