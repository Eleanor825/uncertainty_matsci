"""Pure accounting and metrics for the preregistered real SnAr study."""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

BOUNDS = {"tau": (.5, 2.), "equiv_pldn": (1., 5.), "conc_dfnb": (.1, .5), "temperature": (30., 120.)}

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)

def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()

def write_once(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    content = canonical(value)+"\n"
    if path.exists():
        if path.read_text() != content:
            raise RuntimeError(f"Immutable artifact conflict: {path}")
        return
    temporary = path.with_name(path.name+f".{os.getpid()}.tmp")
    with temporary.open("x") as f:
        f.write(content); f.flush(); os.fsync(f.fileno())
    os.link(temporary, path); temporary.unlink()

def read(path):
    return json.loads(Path(path).read_text())

def append_event(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("a") as f:
        f.write(canonical({"timestamp": time.time(), **value})+"\n"); f.flush(); os.fsync(f.fileno())

def observed(row):
    return {"parameters": row["parameters"], "objectives": row["objectives"]}

def coordinates(row):
    y=row["objectives"] if "objectives" in row else row
    sty,e=float(y["sty"]),float(y["e_factor"])
    if not all(map(math.isfinite,(sty,e))) or sty < 0 or e > 1000 or e < 0:
        raise ValueError("Official objective outside its code-defined finite domain")
    return sty/13000.,(1000.-e)/1000.

def hypervolume(rows):
    points=sorted((coordinates(row) for row in rows),reverse=True)
    area=height=0.
    for x,y in points:
        if y>height:
            area += x*(y-height); height=y
    return area

def pareto(rows):
    # Stable first occurrence for equal objective pairs.
    unique={}
    for r in rows: unique.setdefault(coordinates(r),r)
    result=[]; height=-math.inf
    for xy,r in sorted(unique.items(),reverse=True):
        if xy[1]>height:
            result.append(r); height=xy[1]
    return result

def prior_summary(rows, maximum=4):
    front=pareto(rows)
    if len(front)<=maximum: return [observed(r) for r in front]
    indices=sorted({round(i*(len(front)-1)/(maximum-1)) for i in range(maximum)})
    return [observed(front[i]) for i in indices]

def uniform(seed):
    rng=random.Random(seed)
    return {k:rng.uniform(lo,hi) for k,(lo,hi) in BOUNDS.items()}

def lhs(seed,n=5):
    # Explicit deterministic Latin hypercube shared identically by all arms;
    # this is an initialization design, not the named Summit Random baseline.
    rng=random.Random(seed); rows=[{} for _ in range(n)]
    for key,(lo,hi) in BOUNDS.items():
        order=list(range(n)); rng.shuffle(order)
        for i,cell in enumerate(order): rows[i][key]=lo+(hi-lo)*(cell+rng.random())/n
    return rows

def episode_summary(records, prior):
    initial=hypervolume(prior); all_rows=list(prior); curve=[]
    for row in records:
        all_rows.append(row); curve.append(hypervolume(all_rows))
    risk=[r for r in records if r.get("risk") is not None]
    brier=sum((r["risk"]-r["no_hvi"])**2 for r in risk)/len(risk) if risk else None
    attempts=sum(r.get("proposal_count",0) for r in records)
    invalid=sum(r.get("invalid_proposals",0) for r in records)
    rate=invalid/attempts if attempts else 0.
    return {"queries":len(records),"prior_rows":len(prior),"prior_hv":initial,
        "final_hv":curve[-1] if curve else initial,"final_hv_gain":curve[-1]-initial if curve else 0.,
        "mean_querywise_hv":sum(curve)/len(curve) if curve else initial,
        "mean_querywise_hv_gain":sum(x-initial for x in curve)/len(curve) if curve else 0.,
        "hv_curve":curve,"brier":brier,"risk_observations":len(risk),"invalid_proposal_rate":rate,
        "proposal_count":attempts,"invalid_proposals":invalid,"raw_pareto_front":[observed(r) for r in pareto(all_rows)]}

def fitness(summary):
    if summary["brier"] is None:
        raise RuntimeError("Evolution fitness needs actually observed risk predictions")
    return summary["mean_querywise_hv"]-.1*summary["brier"]-.1*summary["invalid_proposal_rate"]
