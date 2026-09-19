"""Only the original loader import is repaired; no controller mathematics changed."""
def load_controllers(condition,policy,reg):
    from matdiscovery.core_protocol import collection_manifest_paths
    from matdiscovery.core_protocol import corpus_contract
    from matdiscovery.core_representation import paths_for
    from matdiscovery.training_jobs import load_frozen_controllers
    protocol=condition.protocol();core=condition.core
    from matdiscovery.policy import declared_runtime_options
    runtime=declared_runtime_options(protocol)
    return load_frozen_controllers(policy,model_key='qwen35_4b',benchmark='made',method='esopt_graph_risk',
        risk_checkpoint=condition.project/'experiments/risk_models/qwen35_4b/made/graph_risk.pt',
        transcoder_manifest=paths_for(core)['bank']/'transcoder_manifest.json',graph_config=protocol['graph'],device=runtime['device'],
        failure_control=protocol['failure_control'],corpus_contract=corpus_contract(core,collection_manifest_paths(core)))
