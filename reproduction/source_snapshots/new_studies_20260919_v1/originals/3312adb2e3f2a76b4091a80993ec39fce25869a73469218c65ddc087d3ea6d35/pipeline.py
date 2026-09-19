"""Durable SnAr collection, native graph risk fitting, full ES, and held-out tests.

The official CPU oracle is a separate process. Existing MADE files are read-only.
No test observation is available before adaptation and checkpoint selection end.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import gc
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from study import (append_event, canonical, coordinates, digest, episode_summary, fitness,
                   hypervolume, lhs, observed, prior_summary, read, uniform, write_once)

ROOT=Path(__file__).resolve().parent

class Oracle:
    def __init__(self, runtime, output, scope, maximum_queries):
        from oracle.client import OracleClient
        self.client=OracleClient(runtime["oracle_python"],Path(output)/"oracle_journals"/scope,
            seed=0,max_queries=maximum_queries,classification="benchmark",source_root=runtime["oracle_source_root"],
            protocol=ROOT/"protocol.json",protocol_sha256=runtime["protocol_sha256"])
    def evaluate(self,query_id,parameters):
        response=self.client.evaluate(query_id,parameters)
        objectives=response["objectives"]
        coordinates(objectives)
        return objectives,response
    def close(self,completed=True):
        if self.client.process.poll() is None:
            if completed: self.client.close()
            else:
                self.client.request("pause",{})
                self.client.process.stdin.close(); self.client.process.wait(timeout=30); self.client.stderr.close()

class Pipeline:
    def __init__(self, args, *, gpu):
        self.args=args; self.out=Path(args.output).resolve(); self.out.mkdir(parents=True,exist_ok=True)
        self.protocol=read(ROOT/"protocol.json"); self.runtime=read(args.runtime)
        self.event_path=self.out/"events.jsonl"
        maximum_queries={"collect":150,"fit_risk":1,"evolve":200,"evaluate":250}
        self.oracle=Oracle(self.runtime,self.out,args.stage+"_"+(args.arm or args.branch or "default"),maximum_queries[args.stage]) if args.stage!="freeze" else None
        self.policy=self.adapter=self.optimizer=self.bank=self.attributor=self.risk=None
        self.graph_states=set(); self.feature_names=[]
        self.weight_hash=None
        if gpu: self.load_policy()
    def event(self,event,**kw):
        record={"event":event,**kw}; append_event(self.event_path,record); print(canonical(record),flush=True)
    def load_policy(self):
        import torch
        from method.snar_policy import SnArPolicyAdapter, SnArPolicyConfig
        from matdiscovery.esopt import AgenticESOpt
        from runtime.resource_guard import apply_torch_contract
        contract=apply_torch_contract()
        torch.cuda.reset_peak_memory_stats(0)
        cfg=self.protocol["policy"]
        pcfg=SnArPolicyConfig(**{k:cfg[k] for k in ("context_budget","max_input_tokens","max_new_tokens","history_window","temperature","top_p","top_k")})
        objective=("Seek the Pareto frontier: maximize sty and minimize e_factor. "
            "The fixed normalized hypervolume has coordinates sty/13000 and (1000-e_factor)/1000 "
            "with reference (0,0), without clipping sty. Explore and improve observed tradeoffs.")
        self.adapter=SnArPolicyAdapter.from_verified_checkpoint(self.runtime["model_manifest"],self.runtime["model_directory"],
            objective_description=objective,config=pcfg,device="cuda:0")
        self.policy=self.adapter.policy
        self.optimizer=AgenticESOpt(self.policy.model,policy_model_id=self.policy.model_id+"@"+self.policy.revision,parameter_scope="full")
        self.weight_hash=self.optimizer.base_state_hash
        policy_receipt={"checkpoint_hash":self.policy.checkpoint_hash,"base_state_hash":self.weight_hash,
            "configuration":self.policy._configuration(),"configuration_fingerprint":self.policy.configuration_fingerprint,
            "runtime":self.policy.runtime_precision_record(),"parameter_count":self.optimizer.total_parameters,
            "parameter_manifest":self.optimizer.parameter_manifest()}
        write_once(self.out/"base_policy.json",policy_receipt)
        self.event("policy_loaded",base_state_hash=self.weight_hash,allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved())
        # One real forward/capture qualification, explicitly outside oracle data.
        # This is followed automatically by the complete registered experiment.
        probe_path=self.out/"technical_policy_probe.json"
        if not probe_path.exists():
            from method.features import capture_features
            generation,prompt=self.adapter.generate_action({"query_count":0,"budget":30},[],seed=1729)
            if not generation.success: raise RuntimeError("Technical SnAr JSON qualification failed: "+str(generation.failure_code))
            capture,_,meta=capture_features(self.policy,generation)
            probe={"technical_only":True,"oracle_calls":0,"used_in_fit":False,"generation":generation.to_record(),
                "capture":meta,"layers":len(capture.mlp_inputs),"complete_prefix_tokens":generation.input_ids_with_completion.shape[1],
                "cuda_peak_reserved_bytes":torch.cuda.max_memory_reserved(),"cuda_peak_allocated_bytes":torch.cuda.max_memory_allocated(),
                "scope":"real_generation_and_native_capture; native_graph_gate_follows_SnAr_FVU_admission"}
            del capture; write_once(probe_path,probe); self.event("technical_policy_probe_passed",tokens=probe["complete_prefix_tokens"],peak_reserved=probe["cuda_peak_reserved_bytes"])
    def hydrate(self,record):
        import torch
        from matdiscovery.policy import ActionGeneration
        old=record["model_stamp"]; current=asdict(self.policy.model_stamp)
        for k in ("checkpoint_hash","model_id","generation","perturbation_seed","perturbation_sigma"):
            if old[k]!=current[k]: raise RuntimeError("Generation replay policy boundary differs: "+k)
        if record["configuration_fingerprint"]!=self.policy.configuration_fingerprint:
            raise RuntimeError("Generation replay configuration differs")
        if record["policy_runtime"]!=self.policy.runtime_precision_record(): raise RuntimeError("Replay runtime differs")
        from matdiscovery.esopt import tensor_state_hash
        ids=torch.tensor(record["input_ids_with_completion"],dtype=torch.long)
        if tensor_state_hash({"input_ids":ids})!=record["prefix_hash"]:
            raise RuntimeError("Original generated prefix bytes changed")
        start=record["prompt_token_count"]
        if ids.shape[1]-start!=record["completion_count"]: raise RuntimeError("Original completion count changed")
        if self.policy.tokenizer.decode(ids[0,start:].tolist(),skip_special_tokens=False,clean_up_tokenization_spaces=False)!=record["raw_text"]:
            raise RuntimeError("Original completion text does not match original tokens")
        values=dict(record); values["model_stamp"]=self.policy.model_stamp
        values["input_ids_with_completion"]=torch.tensor(values["input_ids_with_completion"],dtype=torch.long)
        values["entropy"]=tuple(values["entropy"]); values["logprobs"]=tuple(values["logprobs"])
        return ActionGeneration(**values)
    def capture_shards(self,generation,directory,split,group):
        from method.features import capture_features,activation_rows
        from matdiscovery.transcoders import write_activation_shard
        capture,values,provenance=capture_features(self.policy,generation)
        for index,(path,row) in enumerate(activation_rows(capture,token_rows=16).items()):
            shard=directory/f"layer{index:02d}.pt"
            if not shard.exists():
                n=len(row["inputs"])
                write_activation_shard(shard,row["inputs"],row["outputs"],group_ids=[group]*n,
                    prefix_hashes=[row["prefix_hash"]]*n,split=split,policy_fingerprint=self.policy.checkpoint_hash,layer_path=path)
        del capture
        return values,provenance
    def graph(self,generation,directory):
        import numpy as np
        from method.features import attribute_features,capture_features,validate_snar_backend
        stamp=self.policy.model_stamp.state_id
        if stamp not in self.graph_states:
            report=validate_snar_backend(self.attributor,self.policy,generation,config=self.graph_config)
            write_once(directory/("backend_gate_"+digest(stamp)+".json"),{"state_id":stamp,"report":report,"weight_hash":self.weight_hash})
            self.graph_states.add(stamp)
        capture,hidden,capture_meta=capture_features(self.policy,generation); del capture
        values,meta,result=attribute_features(self.attributor,self.policy,generation,config=self.graph_config)
        graph_path=directory/"native_graph.npz"
        arrays={k:v for k,v in vars(result.graph).items() if isinstance(v,np.ndarray)}
        if not graph_path.exists(): np.savez_compressed(graph_path,**arrays)
        from matdiscovery.accounting import file_sha256
        return {**hidden,**values},{"capture":capture_meta,"graph":meta,"graph_file":str(graph_path),"node_records":result.node_records,
            "graph_provenance":dict(result.graph.provenance),"graph_sha256":file_sha256(graph_path),
            "weight_hash":self.weight_hash,"source_stamp":asdict(generation.model_stamp)}
    def score(self,features):
        import numpy as np
        return float(self.risk.predict_proba(np.array([[features.get(k,float("nan")) for k in self.feature_names]]),self.feature_names)[0])
    def candidate(self,path,observation,history,prior,seed,*,capture_split=None,group=None,guided=False):
        path.mkdir(parents=True,exist_ok=True)
        completed=path/"candidate.json"
        if completed.exists(): return read(completed)
        raw=path/"generation.json"
        if raw.exists():
            saved=read(raw)
            if saved["weight_hash"]!=self.weight_hash: raise RuntimeError("Pending proposal belongs to another exact weight state")
            generation=self.hydrate(saved["generation"]); prompt=saved["prompt"]
        else:
            generation,prompt=self.adapter.generate_action(observation,history,seed=seed,prior_history=prior)
            write_once(raw,{"generation":generation.to_record(),"prompt":prompt,"weight_hash":self.weight_hash})
        result={"success":generation.success,"parameters":generation.parsed_action if generation.success else None,
            "generation_path":str(raw),"risk":None,"features":None,"failure_code":generation.failure_code}
        if generation.success:
            if capture_split:
                values,meta=self.capture_shards(generation,path,capture_split,group)
                result.update(features=values,capture=meta)
            if guided:
                try:
                    values,meta=self.graph(generation,path)
                    result.update(features=values,graph=meta,risk=self.score(values))
                except Exception as exc:
                    # No proxy graph/probability or counterfactual outcome is invented.
                    result["graph_unavailable"]={"type":type(exc).__name__,"error":str(exc)}
                    self.event("graph_unavailable",path=str(path),error=str(exc))
        write_once(completed,result)
        return result
    def episode(self,name,*,arm,seed,budget,prior=(),capture_split=None):
        directory=self.out/"episodes"/name; directory.mkdir(parents=True,exist_ok=True)
        header={"name":name,"arm":arm,"seed":seed,"budget":budget,"prior_fingerprint":digest(prior),
            "weight_hash":self.weight_hash if arm not in ("random","gp_ei_scalarized") else None,"protocol_fingerprint":digest(self.protocol)}
        write_once(directory/"episode.json",header)
        done=directory/"summary.json"
        if done.exists(): return read(done)
        records=[]; design=lhs(seed,5); summary_prior=prior_summary(prior,4)
        for i in range(budget):
            target=directory/f"query{i:03d}.json"
            if target.exists():
                records.append(read(target)); continue
            start=time.time(); chosen=None; proposals=[]
            if i<5:
                parameters=design[i]; kind="paired_lhs_initial_design"
            elif arm in ("random","gp_ei_scalarized"):
                from baselines.planners import propose
                parameters=propose([observed(r) for r in [*prior,*records]],seed,i,strategy="gp_ei" if arm=="gp_ei_scalarized" else "random")
                kind=arm
            else:
                observation={"query_count":i,"budget":budget}; history=[observed(r) for r in records]
                guided=arm in ("uq_esopt","uq_only")
                random_retry=False
                if arm=="random_controller":
                    import random
                    random_retry=random.Random(seed*100000+i*10+3).random()<read(self.out/"risk_fit.json")["random_controller_retry_probability"]
                for j in range(2):
                    candidate=self.candidate(directory/f"query{i:03d}_proposal{j}",observation,history,summary_prior,
                        seed*100000+i*10+j,capture_split=capture_split,group=name,guided=guided)
                    proposals.append(candidate)
                    if candidate["success"] and ((not guided and not random_retry) or (guided and candidate["risk"] is not None and candidate["risk"]<.5)):
                        break
                valid=[p for p in proposals if p["success"]]
                if valid:
                    chosen=min(valid,key=lambda p: p["risk"] if p["risk"] is not None else math.inf)
                    if arm=="random_controller":
                        import random
                        chosen=random.Random(seed*100000+i*10+4).choice(valid)
                    parameters=chosen["parameters"]; kind="llm_proposal"
                else:
                    parameters=uniform(seed*100000+i*10+9); kind="explicit_invalid_fallback"
            intent={"query_id":name+f"/q{i:03d}","parameters":parameters,"kind":kind,
                "chosen_generation":chosen["generation_path"] if chosen else None,"prior_and_history_hv":hypervolume([*prior,*records])}
            write_once(directory/f"query{i:03d}_intent.json",intent)
            objectives,receipt=self.oracle.evaluate(intent["query_id"],parameters)
            row={"query_id":intent["query_id"],"parameters":parameters,"objectives":objectives,"kind":kind,
                "risk":chosen["risk"] if chosen else None,"proposal_count":len(proposals),
                "invalid_proposals":sum(not p["success"] for p in proposals),"chosen_generation":intent["chosen_generation"],
                "chosen_features_path":str(Path(chosen["generation_path"]).with_name("candidate.json")) if chosen else None,
                "oracle_receipt":receipt,"elapsed_seconds":time.time()-start}
            after=hypervolume([*prior,*records,row]); row["hv_increment"]=after-intent["prior_and_history_hv"]
            row["no_hvi"]=int(row["hv_increment"]<=1e-12)
            write_once(target,row); records.append(row)
            self.event("query_completed",episode=name,query=i+1,budget=budget,arm=arm,objectives=objectives,hv=after,risk=row["risk"])
        result={**header,**episode_summary(records,prior)}
        if arm in ("uq_esopt","uq_only"):
            attempted=[read(p) for p in directory.glob("query*_proposal*/candidate.json")]
            valid=[p for p in attempted if p["success"]]
            available=[p for p in valid if p["risk"] is not None]
            result["valid_proposal_graph_fraction"]=len(available)/len(valid) if valid else 0.
            if result["valid_proposal_graph_fraction"]<.9:
                write_once(directory/"failed_graph_coverage.json",result)
                raise RuntimeError("Registered minimum 90% native graph availability failed")
        write_once(done,result); self.event("episode_completed",episode=name,summary=str(done))
        return result
    def collect(self):
        for split,key in (("train","train_seeds"),("dev","dev_seeds")):
            for seed in self.protocol["risk"][key]:
                self.episode(f"collection_{split}_{seed}",arm="qwen_base",seed=seed,budget=30,capture_split=split)
        write_once(self.out/"collection_complete.json",{"episodes":5,"queries":150})
    def load_bank(self):
        from method.features import load_candidate_bank,validate_transfer_on_shards,make_attributor
        from matdiscovery.representation_training import GraphStageConfig
        self.graph_config=GraphStageConfig(**self.protocol["graph"])
        train={}; dev={}
        for index,path in enumerate(self.policy.mlp_paths):
            train[path]=sorted(str(p) for p in (self.out/"episodes").glob(f"collection_train_*/query*_proposal*/layer{index:02d}.pt"))
            dev[path]=sorted(str(p) for p in (self.out/"episodes").glob(f"collection_dev_*/query*_proposal*/layer{index:02d}.pt"))
        # Remove exact dev prefixes occurring in train, uniformly for all32 layers.
        from matdiscovery.transcoders import _load_shard
        first=self.policy.mlp_paths[0]
        train_prefixes={h for p in train[first] for h in _load_shard(p)["metadata"]["prefix_hashes"]}
        rejected={str(Path(p).parent) for p in dev[first] if train_prefixes.intersection(_load_shard(p)["metadata"]["prefix_hashes"])}
        dev={layer:[p for p in paths if str(Path(p).parent) not in rejected] for layer,paths in dev.items()}
        write_once(self.out/"representation_dev_prefix_filter.json",{"rejected_proposal_directories":sorted(rejected),"test_used":False})
        selected=self.out/"representation_selected.json"
        path=read(selected)["path"] if selected.exists() else self.runtime["candidate_bank"]
        self.bank=load_candidate_bank(path,self.policy)
        report=validate_transfer_on_shards(self.bank,train,dev)
        if not selected.exists():
            write_once(self.out/"candidate_transfer_fidelity.json",report)
            if not report["passed"]:
                from method.features import train_snar_bank
                path=train_snar_bank(self.policy,train,dev,output_directory=self.out/"snar_transcoders",registration=self.protocol,
                    training_backend="isolated_cuda_2gib",expected_gpu_uuid=self.runtime["gpu_uuid"])
                self.bank=load_candidate_bank(path,self.policy)
                report=validate_transfer_on_shards(self.bank,train,dev)
                write_once(self.out/"fresh_snar_transfer_fidelity.json",report)
            if not report["passed"]: raise RuntimeError("All32 SnAr fidelity gates must pass before native graph use")
            write_once(selected,{"path":str(path),"bank_fingerprint":self.bank.bank_fingerprint,"fidelity_fingerprint":report["fingerprint"]})
        if not report["passed"]: raise RuntimeError("Recorded representation no longer passes SnAr fidelity")
        self.attributor=make_attributor(self.policy,self.bank,config=self.graph_config)
        self.event("representation_ready",bank_fingerprint=self.bank.bank_fingerprint)
    def fit_risk(self):
        import numpy as np
        from matdiscovery.uncertainty import CalibratedRiskModel,RiskTrainingConfig,risk_metrics
        self.load_bank(); rows={"train":[],"dev":[]}; total=0; graph_ok=0; seen=set(); dev_total=0
        rejected=set(read(self.out/"representation_dev_prefix_filter.json")["rejected_proposal_directories"])
        for split in ("train","dev"):
            for episode in sorted((self.out/"episodes").glob(f"collection_{split}_*")):
                for query in sorted(episode.glob("query[0-9][0-9][0-9].json")):
                    row=read(query)
                    if not row["chosen_generation"]: continue
                    directory=Path(row["chosen_generation"]).parent
                    if split=="dev" and str(directory) in rejected: continue
                    saved=read(row["chosen_generation"])
                    if saved["weight_hash"]!=self.optimizer.base_state_hash: raise RuntimeError("Risk collection is not original model")
                    key=saved["generation"]["prefix_hash"]
                    if split=="dev" and key in seen: continue
                    if split=="train": seen.add(key)
                    else: dev_total+=1
                    total+=1; graph_path=directory/"risk_graph.json"
                    if graph_path.exists(): result=read(graph_path)
                    else:
                        try:
                            generation=self.hydrate(saved["generation"])
                            values,meta=self.graph(generation,directory)
                            result={"available":True,"features":values,"metadata":meta}
                        except Exception as exc:
                            result={"available":False,"error":str(exc),"type":type(exc).__name__}
                        write_once(graph_path,result)
                    if not result["available"]: continue
                    graph_ok+=1; rows[split].append({"features":result["features"],"label":row["no_hvi"],"episode":episode.name,
                        "query_path":str(query),"graph_path":str(graph_path),"prefix_hash":key})
                    self.event("training_graph_completed",split=split,completed=graph_ok,total=total)
        if not total or graph_ok/total<.9: raise RuntimeError("SnAr native training graph availability below registered 90%")
        feature_names=sorted({k for row in rows["train"] for k in row["features"]})
        arrays={s:np.array([[r["features"].get(k,float("nan")) for k in feature_names] for r in rs]) for s,rs in rows.items()}
        ys={s:np.array([r["label"] for r in rs]) for s,rs in rows.items()}
        cfg=self.protocol["risk"]
        fit_config=RiskTrainingConfig(label_kind=cfg["label"],**{k:cfg[k] for k in
            ("seed","hidden_width","epochs","batch_size","learning_rate","weight_decay","patience","include_error_similarity")})
        self.risk=CalibratedRiskModel().fit(arrays["train"],ys["train"],arrays["dev"],ys["dev"],feature_names=feature_names,
            train_groups=[r["episode"] for r in rows["train"]],dev_groups=[r["episode"] for r in rows["dev"]],
            train_episode_ids=[r["episode"] for r in rows["train"]],config=fit_config)
        self.risk.save(self.out/"snar_risk.pt"); self.feature_names=feature_names
        report={"feature_names":feature_names,"rows":rows,"graph_fraction":graph_ok/total,"provenance":self.risk.provenance,
            "development_metrics":risk_metrics(ys["dev"],self.risk.predict_proba(arrays["dev"],feature_names)),
            "random_controller_retry_probability":float((np.sum(self.risk.predict_proba(arrays["dev"],feature_names)>=.5)+dev_total-len(rows["dev"]))/dev_total),
            "random_controller_rate_population":{"valid_development_actions_after_prefix_filter":dev_total,"available_graphs":len(rows["dev"]),
                "unavailable_graphs_counted_as_retry":dev_total-len(rows["dev"])},"test_used":False}
        write_once(self.out/"risk_fit.json",report); self.event("risk_fitted",development_metrics=report["development_metrics"])
    def load_method(self):
        from matdiscovery.uncertainty import CalibratedRiskModel
        self.load_bank(); self.risk=CalibratedRiskModel.load(self.out/"snar_risk.pt")
        self.feature_names=read(self.out/"risk_fit.json")["feature_names"]
    def evolve(self,branch):
        if branch=="full": self.load_method()
        cfg=self.protocol["evolution"]
        arm="uq_esopt" if branch=="full" else "es_only"
        def reward(summary):
            return fitness(summary) if branch=="full" else summary["mean_querywise_hv"]-.1*summary["invalid_proposal_rate"]
        # Resume only from the last durable, hash-verified complete update history.
        for generation in range(2,-1,-1):
            checkpoint=self.out/f"{branch}_es_history_G{generation}.json"
            if checkpoint.exists():
                self.optimizer.replay_history(checkpoint)
                self.policy.mark_state("snar_history_resume",generation=generation)
                self.weight_hash=self.optimizer.model_state_hash(); break
        current=len(self.optimizer.history)
        for generation in range(current,3):
            if len(self.optimizer.history)!=generation: raise RuntimeError("Evolution generation chain differs")
            scores=[]
            for seed in cfg["checkpoint_dev_seeds"]:
                s=self.episode(f"es_{branch}_dev_G{generation}_{seed}",arm=arm,seed=seed,budget=20)
                scores.append(reward(s))
            write_once(self.out/f"{branch}_es_development_G{generation}.json",{"generation":generation,"fitness":sum(scores)/len(scores),"scores":scores,"weight_hash":self.weight_hash})
            if generation==2: break
            seeds=cfg["mutation_seeds"][generation]; rewards=[]
            for mutation in seeds:
                self.optimizer.begin_perturbation(mutation,cfg["sigma"][generation])
                try:
                    self.policy.mark_state("snar_perturb",generation=generation,perturbation_seed=mutation,perturbation_sigma=cfg["sigma"][generation])
                    self.weight_hash=self.optimizer.model_state_hash()
                    s=self.episode(f"es_{branch}_population_G{generation}_m{mutation}",arm=arm,seed=cfg["fitness_episode_seeds"][generation],budget=20)
                    rewards.append(reward(s))
                finally:
                    self.optimizer.end_perturbation(); self.policy.mark_state("snar_restore",generation=generation)
                    self.weight_hash=self.optimizer.model_state_hash(); gc.collect()
            update=self.optimizer.step(seeds,rewards,alpha=cfg["alpha"],sigma=cfg["sigma"][generation],
                metadata={"protocol_fingerprint":digest(self.protocol),"branch":branch,
                    "fitness_definition":cfg["fitness"] if branch=="full" else "mean_querywise_HV - 0.1*invalid_proposal_fraction"})
            self.policy.mark_state("snar_update",generation=generation+1); self.weight_hash=update["state_hash_after"]
            self.optimizer.save_history(self.out/f"{branch}_es_history_G{generation+1}.json")
            self.event("es_update_completed",generation=generation+1,rewards=rewards,weight_hash=self.weight_hash)
        dev=[read(self.out/f"{branch}_es_development_G{g}.json") for g in range(3)]
        selected=max(dev,key=lambda x:(x["fitness"],-x["generation"]))
        source=read(self.out/f"{branch}_es_history_G2.json"); source["history"]=source["history"][:selected["generation"]]
        source["current_state_hash"]=selected["weight_hash"]
        write_once(self.out/f"{branch}_selected_es_history.json",source)
        write_once(self.out/f"{branch}_checkpoint_selection.json",{"selected":selected,"development":dev,"test_used":False,
            "checkpoint_form":"original_verified_weights_plus_exact_full_parameter_ES_history", "test_replay_hash_required":True})
        write_once(self.out/f"{branch}_evolution_complete.json",{"queries":200,"selected":selected})
    def freeze(self):
        for branch in ("full","es_only"):
            if not (self.out/f"{branch}_evolution_complete.json").exists(): raise RuntimeError("Both independent ES branches must complete")
        archive=[]
        for episode in sorted((self.out/"episodes").iterdir()):
            if not (episode/"summary.json").exists(): raise RuntimeError("Cannot freeze prior with incomplete adaptation episode")
            if not episode.name.startswith(("collection_","es_")): raise RuntimeError("Test data exists before prior freeze")
            archive.extend(read(p) for p in sorted(episode.glob("query[0-9][0-9][0-9].json")))
        if len(archive)!=550: raise RuntimeError("Registered adaptation query count mismatch")
        write_once(self.out/"shared_prior_archive.json",archive)
        write_once(self.out/"adaptation_complete.json",{"queries":550,"archive_fingerprint":digest(archive),
            "selection":{b:read(self.out/f"{b}_checkpoint_selection.json") for b in ("full","es_only")}})
    def evaluate(self,arm):
        if not (self.out/"adaptation_complete.json").exists(): raise RuntimeError("Adaptation/prior must finish before test")
        prior=read(self.out/"shared_prior_archive.json")
        if arm in ("uq_esopt","uq_only"): self.load_method()
        if arm in ("uq_esopt","es_only"):
            branch="full" if arm=="uq_esopt" else "es_only"
            self.optimizer.replay_history(self.out/f"{branch}_selected_es_history.json")
            generation=len(self.optimizer.history); self.policy.mark_state("snar_fixed_test_checkpoint",generation=generation)
            self.weight_hash=self.optimizer.model_state_hash()
            if self.weight_hash!=read(self.out/f"{branch}_checkpoint_selection.json")["selected"]["weight_hash"]:
                raise RuntimeError("Clean selected-weight replay did not match")
            write_once(self.out/f"{branch}_selected_clean_replay.json",{"verified":True,"weight_hash":self.weight_hash,"generation":generation})
        for seed in self.protocol["evaluation"]["seeds"]:
            self.episode(f"test_{arm}_{seed}",arm=arm,seed=seed,budget=50,prior=prior)
        write_once(self.out/f"test_{arm}_complete.json",{"episodes":5,"oracle_queries":250})

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--runtime",required=True); parser.add_argument("--output",required=True)
    parser.add_argument("--stage",choices=["collect","fit_risk","evolve","freeze","evaluate"],required=True)
    parser.add_argument("--arm",choices=["random","gp_ei_scalarized","qwen_base","uq_esopt","uq_only","es_only","random_controller"])
    parser.add_argument("--branch",choices=["full","es_only"])
    args=parser.parse_args(); p=None; completed=False
    try:
        gpu=args.stage!="freeze" and (args.stage!="evaluate" or args.arm not in ("random","gp_ei_scalarized"))
        p=Pipeline(args,gpu=gpu)
        if args.stage=="collect": p.collect()
        elif args.stage=="fit_risk": p.fit_risk()
        elif args.stage=="evolve": p.evolve(args.branch)
        elif args.stage=="freeze": p.freeze()
        else: p.evaluate(args.arm)
        p.event("stage_completed",stage=args.stage,arm=args.arm)
        completed=True
    except BaseException as exc:
        if p: p.event("stage_failed",stage=args.stage,arm=args.arm,type=type(exc).__name__,error=str(exc))
        traceback.print_exc(); raise
    finally:
        if p:
            if p.oracle: p.oracle.close(completed=completed)
            if p.policy:
                import torch
                append_event(p.event_path,{"event":"gpu_peak","stage":args.stage,"allocated_bytes":torch.cuda.max_memory_allocated(),"reserved_bytes":torch.cuda.max_memory_reserved()})

if __name__=="__main__": main()
