# Scientific source excerpt: study.py

Logical source: `benchmark_extensions/summit_snar_20260918/study.py`.

SHA256 of the **complete original source**: `cda812f771cbb10e4b27e54316c2e2dd3b39d31733c734b37d54b0e19d084db4`. This is a quoted excerpt for review, not a portable executable or the complete source file. Line numbers below refer to the original file; the export manifest separately hashes this excerpt.

## Lines 40–105

```text
40:     return {"parameters": row["parameters"], "objectives": row["objectives"]}
41:
42: def coordinates(row):
43:     y=row["objectives"] if "objectives" in row else row
44:     sty,e=float(y["sty"]),float(y["e_factor"])
45:     if not all(map(math.isfinite,(sty,e))) or sty < 0 or e > 1000 or e < 0:
46:         raise ValueError("Official objective outside its code-defined finite domain")
47:     return sty/13000.,(1000.-e)/1000.
48:
49: def hypervolume(rows):
50:     points=sorted((coordinates(row) for row in rows),reverse=True)
51:     area=height=0.
52:     for x,y in points:
53:         if y>height:
54:             area += x*(y-height); height=y
55:     return area
56:
57: def pareto(rows):
58:     # Stable first occurrence for equal objective pairs.
59:     unique={}
60:     for r in rows: unique.setdefault(coordinates(r),r)
61:     result=[]; height=-math.inf
62:     for xy,r in sorted(unique.items(),reverse=True):
63:         if xy[1]>height:
64:             result.append(r); height=xy[1]
65:     return result
66:
67: def prior_summary(rows, maximum=4):
68:     front=pareto(rows)
69:     if len(front)<=maximum: return [observed(r) for r in front]
70:     indices=sorted({round(i*(len(front)-1)/(maximum-1)) for i in range(maximum)})
71:     return [observed(front[i]) for i in indices]
72:
73: def uniform(seed):
74:     rng=random.Random(seed)
75:     return {k:rng.uniform(lo,hi) for k,(lo,hi) in BOUNDS.items()}
76:
77: def lhs(seed,n=5):
78:     # Explicit deterministic Latin hypercube shared identically by all arms;
79:     # this is an initialization design, not the named Summit Random baseline.
80:     rng=random.Random(seed); rows=[{} for _ in range(n)]
81:     for key,(lo,hi) in BOUNDS.items():
82:         order=list(range(n)); rng.shuffle(order)
83:         for i,cell in enumerate(order): rows[i][key]=lo+(hi-lo)*(cell+rng.random())/n
84:     return rows
85:
86: def episode_summary(records, prior):
87:     initial=hypervolume(prior); all_rows=list(prior); curve=[]
88:     for row in records:
89:         all_rows.append(row); curve.append(hypervolume(all_rows))
90:     risk=[r for r in records if r.get("risk") is not None]
91:     brier=sum((r["risk"]-r["no_hvi"])**2 for r in risk)/len(risk) if risk else None
92:     attempts=sum(r.get("proposal_count",0) for r in records)
93:     invalid=sum(r.get("invalid_proposals",0) for r in records)
94:     rate=invalid/attempts if attempts else 0.
95:     return {"queries":len(records),"prior_rows":len(prior),"prior_hv":initial,
96:         "final_hv":curve[-1] if curve else initial,"final_hv_gain":curve[-1]-initial if curve else 0.,
97:         "mean_querywise_hv":sum(curve)/len(curve) if curve else initial,
98:         "mean_querywise_hv_gain":sum(x-initial for x in curve)/len(curve) if curve else 0.,
99:         "hv_curve":curve,"brier":brier,"risk_observations":len(risk),"invalid_proposal_rate":rate,
100:         "proposal_count":attempts,"invalid_proposals":invalid,"raw_pareto_front":[observed(r) for r in pareto(all_rows)]}
101:
102: def fitness(summary):
103:     if summary["brier"] is None:
104:         raise RuntimeError("Evolution fitness needs actually observed risk predictions")
105:     return summary["mean_querywise_hv"]-.1*summary["brier"]-.1*summary["invalid_proposal_rate"]
```
