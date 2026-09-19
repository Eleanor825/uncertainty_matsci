# Scientific source excerpt: pipeline.py

Logical source: `benchmark_extensions/summit_snar_20260918/pipeline.py`.

SHA256 of the **complete original source**: `3312adb2e3f2a76b4091a80993ec39fce25869a73469218c65ddc087d3ea6d35`. This is a quoted excerpt for review, not a portable executable or the complete source file. Line numbers below refer to the original file; the export manifest separately hashes this excerpt.

## Lines 135–241

```text
135:         arrays={k:v for k,v in vars(result.graph).items() if isinstance(v,np.ndarray)}
136:         if not graph_path.exists(): np.savez_compressed(graph_path,**arrays)
137:         from matdiscovery.accounting import file_sha256
138:         return {**hidden,**values},{"capture":capture_meta,"graph":meta,"graph_file":str(graph_path),"node_records":result.node_records,
139:             "graph_provenance":dict(result.graph.provenance),"graph_sha256":file_sha256(graph_path),
140:             "weight_hash":self.weight_hash,"source_stamp":asdict(generation.model_stamp)}
141:     def score(self,features):
142:         import numpy as np
143:         return float(self.risk.predict_proba(np.array([[features.get(k,float("nan")) for k in self.feature_names]]),self.feature_names)[0])
144:     def candidate(self,path,observation,history,prior,seed,*,capture_split=None,group=None,guided=False):
145:         path.mkdir(parents=True,exist_ok=True)
146:         completed=path/"candidate.json"
147:         if completed.exists(): return read(completed)
148:         raw=path/"generation.json"
149:         if raw.exists():
150:             saved=read(raw)
151:             if saved["weight_hash"]!=self.weight_hash: raise RuntimeError("Pending proposal belongs to another exact weight state")
152:             generation=self.hydrate(saved["generation"]); prompt=saved["prompt"]
153:         else:
154:             generation,prompt=self.adapter.generate_action(observation,history,seed=seed,prior_history=prior)
155:             write_once(raw,{"generation":generation.to_record(),"prompt":prompt,"weight_hash":self.weight_hash})
156:         result={"success":generation.success,"parameters":generation.parsed_action if generation.success else None,
157:             "generation_path":str(raw),"risk":None,"features":None,"failure_code":generation.failure_code}
158:         if generation.success:
159:             if capture_split:
160:                 values,meta=self.capture_shards(generation,path,capture_split,group)
161:                 result.update(features=values,capture=meta)
162:             if guided:
163:                 try:
164:                     values,meta=self.graph(generation,path)
165:                     result.update(features=values,graph=meta,risk=self.score(values))
166:                 except Exception as exc:
167:                     # No proxy graph/probability or counterfactual outcome is invented.
168:                     result["graph_unavailable"]={"type":type(exc).__name__,"error":str(exc)}
169:                     self.event("graph_unavailable",path=str(path),error=str(exc))
170:         write_once(completed,result)
171:         return result
172:     def episode(self,name,*,arm,seed,budget,prior=(),capture_split=None):
173:         directory=self.out/"episodes"/name; directory.mkdir(parents=True,exist_ok=True)
174:         header={"name":name,"arm":arm,"seed":seed,"budget":budget,"prior_fingerprint":digest(prior),
175:             "weight_hash":self.weight_hash if arm not in ("random","gp_ei_scalarized") else None,"protocol_fingerprint":digest(self.protocol)}
176:         write_once(directory/"episode.json",header)
177:         done=directory/"summary.json"
178:         if done.exists(): return read(done)
179:         records=[]; design=lhs(seed,5); summary_prior=prior_summary(prior,4)
180:         for i in range(budget):
181:             target=directory/f"query{i:03d}.json"
182:             if target.exists():
183:                 records.append(read(target)); continue
184:             start=time.time(); chosen=None; proposals=[]
185:             if i<5:
186:                 parameters=design[i]; kind="paired_lhs_initial_design"
187:             elif arm in ("random","gp_ei_scalarized"):
188:                 from baselines.planners import propose
189:                 parameters=propose([observed(r) for r in [*prior,*records]],seed,i,strategy="gp_ei" if arm=="gp_ei_scalarized" else "random")
190:                 kind=arm
191:             else:
192:                 observation={"query_count":i,"budget":budget}; history=[observed(r) for r in records]
193:                 guided=arm in ("uq_esopt","uq_only")
194:                 random_retry=False
195:                 if arm=="random_controller":
196:                     import random
197:                     random_retry=random.Random(seed*100000+i*10+3).random()<read(self.out/"risk_fit.json")["random_controller_retry_probability"]
198:                 for j in range(2):
199:                     candidate=self.candidate(directory/f"query{i:03d}_proposal{j}",observation,history,summary_prior,
200:                         seed*100000+i*10+j,capture_split=capture_split,group=name,guided=guided)
201:                     proposals.append(candidate)
202:                     if candidate["success"] and ((not guided and not random_retry) or (guided and candidate["risk"] is not None and candidate["risk"]<.5)):
203:                         break
204:                 valid=[p for p in proposals if p["success"]]
205:                 if valid:
206:                     chosen=min(valid,key=lambda p: p["risk"] if p["risk"] is not None else math.inf)
207:                     if arm=="random_controller":
208:                         import random
209:                         chosen=random.Random(seed*100000+i*10+4).choice(valid)
210:                     parameters=chosen["parameters"]; kind="llm_proposal"
211:                 else:
212:                     parameters=uniform(seed*100000+i*10+9); kind="explicit_invalid_fallback"
213:             intent={"query_id":name+f"/q{i:03d}","parameters":parameters,"kind":kind,
214:                 "chosen_generation":chosen["generation_path"] if chosen else None,"prior_and_history_hv":hypervolume([*prior,*records])}
215:             write_once(directory/f"query{i:03d}_intent.json",intent)
216:             objectives,receipt=self.oracle.evaluate(intent["query_id"],parameters)
217:             row={"query_id":intent["query_id"],"parameters":parameters,"objectives":objectives,"kind":kind,
218:                 "risk":chosen["risk"] if chosen else None,"proposal_count":len(proposals),
219:                 "invalid_proposals":sum(not p["success"] for p in proposals),"chosen_generation":intent["chosen_generation"],
220:                 "chosen_features_path":str(Path(chosen["generation_path"]).with_name("candidate.json")) if chosen else None,
221:                 "oracle_receipt":receipt,"elapsed_seconds":time.time()-start}
222:             after=hypervolume([*prior,*records,row]); row["hv_increment"]=after-intent["prior_and_history_hv"]
223:             row["no_hvi"]=int(row["hv_increment"]<=1e-12)
224:             write_once(target,row); records.append(row)
225:             self.event("query_completed",episode=name,query=i+1,budget=budget,arm=arm,objectives=objectives,hv=after,risk=row["risk"])
226:         result={**header,**episode_summary(records,prior)}
227:         if arm in ("uq_esopt","uq_only"):
228:             attempted=[read(p) for p in directory.glob("query*_proposal*/candidate.json")]
229:             valid=[p for p in attempted if p["success"]]
230:             available=[p for p in valid if p["risk"] is not None]
231:             result["valid_proposal_graph_fraction"]=len(available)/len(valid) if valid else 0.
232:             if result["valid_proposal_graph_fraction"]<.9:
233:                 write_once(directory/"failed_graph_coverage.json",result)
234:                 raise RuntimeError("Registered minimum 90% native graph availability failed")
235:         write_once(done,result); self.event("episode_completed",episode=name,summary=str(done))
236:         return result
237:     def collect(self):
238:         for split,key in (("train","train_seeds"),("dev","dev_seeds")):
239:             for seed in self.protocol["risk"][key]:
240:                 self.episode(f"collection_{split}_{seed}",arm="qwen_base",seed=seed,budget=30,capture_split=split)
241:         write_once(self.out/"collection_complete.json",{"episodes":5,"queries":150})
```

## Lines 328–388

```text
328:     def evolve(self,branch):
329:         if branch=="full": self.load_method()
330:         cfg=self.protocol["evolution"]
331:         arm="uq_esopt" if branch=="full" else "es_only"
332:         def reward(summary):
333:             return fitness(summary) if branch=="full" else summary["mean_querywise_hv"]-.1*summary["invalid_proposal_rate"]
334:         # Resume only from the last durable, hash-verified complete update history.
335:         for generation in range(2,-1,-1):
336:             checkpoint=self.out/f"{branch}_es_history_G{generation}.json"
337:             if checkpoint.exists():
338:                 self.optimizer.replay_history(checkpoint)
339:                 self.policy.mark_state("snar_history_resume",generation=generation)
340:                 self.weight_hash=self.optimizer.model_state_hash(); break
341:         current=len(self.optimizer.history)
342:         for generation in range(current,3):
343:             if len(self.optimizer.history)!=generation: raise RuntimeError("Evolution generation chain differs")
344:             scores=[]
345:             for seed in cfg["checkpoint_dev_seeds"]:
346:                 s=self.episode(f"es_{branch}_dev_G{generation}_{seed}",arm=arm,seed=seed,budget=20)
347:                 scores.append(reward(s))
348:             write_once(self.out/f"{branch}_es_development_G{generation}.json",{"generation":generation,"fitness":sum(scores)/len(scores),"scores":scores,"weight_hash":self.weight_hash})
349:             if generation==2: break
350:             seeds=cfg["mutation_seeds"][generation]; rewards=[]
351:             for mutation in seeds:
352:                 self.optimizer.begin_perturbation(mutation,cfg["sigma"][generation])
353:                 try:
354:                     self.policy.mark_state("snar_perturb",generation=generation,perturbation_seed=mutation,perturbation_sigma=cfg["sigma"][generation])
355:                     self.weight_hash=self.optimizer.model_state_hash()
356:                     s=self.episode(f"es_{branch}_population_G{generation}_m{mutation}",arm=arm,seed=cfg["fitness_episode_seeds"][generation],budget=20)
357:                     rewards.append(reward(s))
358:                 finally:
359:                     self.optimizer.end_perturbation(); self.policy.mark_state("snar_restore",generation=generation)
360:                     self.weight_hash=self.optimizer.model_state_hash(); gc.collect()
361:             update=self.optimizer.step(seeds,rewards,alpha=cfg["alpha"],sigma=cfg["sigma"][generation],
362:                 metadata={"protocol_fingerprint":digest(self.protocol),"branch":branch,
363:                     "fitness_definition":cfg["fitness"] if branch=="full" else "mean_querywise_HV - 0.1*invalid_proposal_fraction"})
364:             self.policy.mark_state("snar_update",generation=generation+1); self.weight_hash=update["state_hash_after"]
365:             self.optimizer.save_history(self.out/f"{branch}_es_history_G{generation+1}.json")
366:             self.event("es_update_completed",generation=generation+1,rewards=rewards,weight_hash=self.weight_hash)
367:         dev=[read(self.out/f"{branch}_es_development_G{g}.json") for g in range(3)]
368:         selected=max(dev,key=lambda x:(x["fitness"],-x["generation"]))
369:         source=read(self.out/f"{branch}_es_history_G2.json"); source["history"]=source["history"][:selected["generation"]]
370:         source["current_state_hash"]=selected["weight_hash"]
371:         write_once(self.out/f"{branch}_selected_es_history.json",source)
372:         write_once(self.out/f"{branch}_checkpoint_selection.json",{"selected":selected,"development":dev,"test_used":False,
373:             "checkpoint_form":"original_verified_weights_plus_exact_full_parameter_ES_history", "test_replay_hash_required":True})
374:         write_once(self.out/f"{branch}_evolution_complete.json",{"queries":200,"selected":selected})
375:     def freeze(self):
376:         for branch in ("full","es_only"):
377:             if not (self.out/f"{branch}_evolution_complete.json").exists(): raise RuntimeError("Both independent ES branches must complete")
378:         archive=[]
379:         for episode in sorted((self.out/"episodes").iterdir()):
380:             if not (episode/"summary.json").exists(): raise RuntimeError("Cannot freeze prior with incomplete adaptation episode")
381:             if not episode.name.startswith(("collection_","es_")): raise RuntimeError("Test data exists before prior freeze")
382:             archive.extend(read(p) for p in sorted(episode.glob("query[0-9][0-9][0-9].json")))
383:         if len(archive)!=550: raise RuntimeError("Registered adaptation query count mismatch")
384:         write_once(self.out/"shared_prior_archive.json",archive)
385:         write_once(self.out/"adaptation_complete.json",{"queries":550,"archive_fingerprint":digest(archive),
386:             "selection":{b:read(self.out/f"{b}_checkpoint_selection.json") for b in ("full","es_only")}})
387:     def evaluate(self,arm):
388:         if not (self.out/"adaptation_complete.json").exists(): raise RuntimeError("Adaptation/prior must finish before test")
```
