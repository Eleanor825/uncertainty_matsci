"""Independent stdlib metric formulas reused from the verified parent archive; no model imports."""
from collections import defaultdict
import math,hashlib,json

def require(ok,message):
    if not ok:raise ValueError(message)

def sha(raw):return hashlib.sha256(raw).hexdigest()

def digest(value):return sha(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode())

def mean(values):return math.fsum(values)/len(values)

def metrics(labels,probabilities):
    require(len(labels)==len(probabilities)>0,'Missing labels/scores')
    require(all(type(y) is int and y in (0,1) for y in labels),'Invalid binary label')
    require(all(type(p) in (int,float) and math.isfinite(p) and 0<=p<=1 for p in probabilities),'Invalid probability')
    n=len(labels);positives=sum(labels);negatives=n-positives
    p1=[p for y,p in zip(labels,probabilities) if y];p0=[p for y,p in zip(labels,probabilities) if not y]
    auroc=mean([float(a>b)+.5*float(a==b) for a in p1 for b in p0]) if positives and negatives else None
    # Average precision at tied-score group endpoints, as in sklearn (not trapezoidal PR area).
    ranked=sorted(zip(probabilities,labels),key=lambda r:r[0],reverse=True)
    terms=[];tp=0;i=0
    while i<n:
        j=i+1
        while j<n and ranked[j][0]==ranked[i][0]:j+=1
        block=sum(y for _,y in ranked[i:j]);tp+=block
        if positives:terms.append((block/positives)*(tp/j))
        i=j
    ap=math.fsum(terms) if positives else None
    bins=defaultdict(list)
    for y,p in zip(labels,probabilities):bins[min(int(p*10),9)].append((y,p))
    ece=math.fsum(len(rs)/n*abs(mean([p for _,p in rs])-mean([y for y,_ in rs])) for rs in bins.values())
    order=sorted(range(n),key=lambda k:probabilities[k]);running=0;coverage=[]
    for count,k in enumerate(order,1):running+=labels[k];coverage.append(running/count)
    losses=[]
    for y,p in zip(labels,probabilities):
        p=max(1e-8,min(1-1e-8,p));losses.append(-math.log(p) if y else -math.log1p(-p))
    return {'n':n,'error_rate':positives/n,'auroc':auroc,'error_auprc':ap,
        'brier':mean([(p-y)**2 for y,p in zip(labels,probabilities)]),'nll':mean(losses),'ece':ece,
        'risk_coverage_auc':mean(coverage),'overconfident_error_rate_p_le_0_1':sum(y==1 and p<=.1 for y,p in zip(labels,probabilities))/n,
        'threshold':.5,'at_or_above_fixed_threshold':sum(p>=.5 for p in probabilities)}

def calibration_bins(labels,probabilities):
    """Fixed ten equal-width bins; empty cells remain explicitly unknown."""
    result=[]
    for b in range(10):
        chosen=[(y,p) for y,p in zip(labels,probabilities) if min(int(p*10),9)==b]
        result.append({'bin':b,'lower':b/10,'upper':(b+1)/10,'upper_inclusive':b==9,
            'count':len(chosen),'mean_predicted_no_HVI':mean([p for _,p in chosen]) if chosen else None,
            'observed_no_HVI_fraction':mean([y for y,_ in chosen]) if chosen else None})
    return result
