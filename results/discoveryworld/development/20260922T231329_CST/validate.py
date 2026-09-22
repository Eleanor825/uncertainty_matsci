import csv,hashlib,json
from collections import Counter
from pathlib import Path
P=Path(__file__).resolve().parent
def need(x,m):
    if not x:raise ValueError(m)
def main():
    d=json.loads((P/'behavior_projection.json').read_text());rows=list(csv.DictReader((P/'pair_actions.csv').open()));need(len(rows)==60,'Not60 action rows')
    for arm,x in d['pair'].items():
        raw=x['rows'];out=[r for r in rows if r['arm']==arm];need(len(raw)==len(out)==30,'Not30 per arm')
        need(x['final_score']==0 and not x['task_success'],'Task-score claim changed')
        counts=Counter(r['action_type']for r in raw);success=sum(r['official_success']is True for r in raw)
        need(success==x['summary']['official_successes']and 30-success==x['summary']['official_failures'],'Success denominator differs')
        packets=[json.dumps(r['action'],sort_keys=True)for r in raw];longest=current=0;previous=None
        for packet in packets:current=current+1 if packet==previous else 1;longest=max(longest,current);previous=packet
        need(longest==x['summary']['longest_consecutive_identical_packet']and len(set(packets))==x['summary']['unique_exact_action_packets'],'Repeat/unique statistics differ')
        for a,b in zip(raw,out):need(json.loads(b['action'])==a['action']and int(b['attempt_index'])==a['attempt_index']and b['official_success']==str(a['official_success']),'CSV differs from raw projection')
        if arm=='SFT':need(counts=={'ROTATE_DIRECTION':28,'MOVE_DIRECTION':2}and success==30,'SFT behavior differs')
        else:need(counts=={'PICKUP':29,'TELEPORT_TO_LOCATION':1}and success==1,'Native behavior differs')
    t=d['training_world0p102'];need(len(t['rows'])==100 and t['summary']['observed_positive_score_steps']==[12,19,71],'Original train sequence changed')
    need([r['score_delta']for r in t['rows']if r['score_delta']and r['score_delta']>0]==[.125,.125,.125],'Train increments differ')
    need(len(list(csv.DictReader((P/'training_milestone_windows.csv').open())))==27,'Training window membership missing')
    for v in json.loads((P/'manifest.json').read_text())['files']:
        b=(P/v['path']).read_bytes();need(len(b)==v['bytes']and hashlib.sha256(b).hexdigest()==v['sha256'],'Hash mismatch')
    print(json.dumps({'passed':True,'closed_development_episodes':2,'development_actions':60,'candidate_graphs':0,'risk_controller_calls':0,'training_actions_audited_separately':100,'training_window_memberships':27,'new_scientific_calls':0}))
if __name__=='__main__':main()
