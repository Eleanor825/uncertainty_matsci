"""CPU contract checks; synthetic fixtures are never scientific outcomes."""
import tempfile
from pathlib import Path
import unittest
from study import hypervolume,pareto,prior_summary,lhs,write_once,read,digest
from pipeline import Pipeline

def point(x,y):
    return {"parameters":{"tau":1.,"equiv_pldn":2.,"conc_dfnb":.3,"temperature":80.},
            "objectives":{"sty":x*13000,"e_factor":1000*(1-y)}}

class MetricsTests(unittest.TestCase):
    def test_pareto_union_and_no_upper_clip(self):
        rows=[point(.2,.8),point(.6,.4),point(.1,.1)]
        self.assertAlmostEqual(hypervolume(rows),.32)
        self.assertEqual(len(pareto(rows)),2)
        self.assertAlmostEqual(hypervolume([point(2.,1.)]),2.)
        self.assertEqual(hypervolume([]),0.)
        self.assertEqual(hypervolume([point(.5,0.)]),0.)
    def test_prior_summary_keeps_extremes_and_lhs_is_paired(self):
        rows=[point(i/10,1-i/10) for i in range(1,10)]
        s=prior_summary(rows)
        self.assertEqual(len(s),4)
        self.assertEqual(s[0],rows[-1]); self.assertEqual(s[-1],rows[0])
        self.assertEqual(lhs(5101),lhs(5101)); self.assertNotEqual(lhs(5101),lhs(5102))
    def test_immutable_artifact(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"r.json"; write_once(p,{"a":1}); write_once(p,{"a":1})
            with self.assertRaises(RuntimeError): write_once(p,{"a":2})

class FakeOracle:
    def __init__(self): self.calls=[]
    def evaluate(self,q,x):
        self.calls.append((q,x))
        return {"sty":1300.,"e_factor":500.},{"unit_fixture_only":True}

class ControllerTests(unittest.TestCase):
    def make(self,d,risks):
        p=Pipeline.__new__(Pipeline); p.out=Path(d); p.weight_hash="synthetic"; p.protocol={"unit_fixture_only":True}
        p.oracle=FakeOracle(); p.event=lambda *a,**kw:None
        self.proposals=[]
        def candidate(path,observation,history,prior,seed,**kw):
            j=len(self.proposals); risk=risks[j]
            path.mkdir(parents=True)
            raw=path/"generation.json"; write_once(raw,{"unit_fixture_only":True})
            result={"success":risk is not None,"parameters":point(.1,.1)["parameters"] if risk is not None else None,
                "risk":risk,"generation_path":str(raw),"features":None}
            result["parameters"] = ({**result["parameters"],"temperature":70.+j} if result["parameters"] else None)
            write_once(path/"candidate.json",result); self.proposals.append(result)
            return result
        p.candidate=candidate
        return p
    def test_risk_selects_second_without_labeling_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=self.make(d,[.9,.2]); s=p.episode("unit",arm="uq_only",seed=5,budget=6)
            q=read(Path(d)/"episodes/unit/query005.json")
            self.assertEqual(len(p.oracle.calls),6); self.assertEqual(q["parameters"]["temperature"],71.)
            self.assertEqual(q["risk"],.2); self.assertEqual(q["no_hvi"],1)
            self.assertNotIn("no_hvi",self.proposals[0]); self.assertEqual(s["risk_observations"],1)
            p.episode("unit",arm="uq_only",seed=5,budget=6)
            self.assertEqual(len(p.oracle.calls),6)
    def test_low_risk_accepts_first(self):
        with tempfile.TemporaryDirectory() as d:
            p=self.make(d,[.2]); p.episode("unit",arm="uq_esopt",seed=5,budget=6)
            self.assertEqual(len(self.proposals),1)
    def test_independent_es_only_has_no_guided_selection(self):
        with tempfile.TemporaryDirectory() as d:
            p=self.make(d,[.9]); p.episode("unit",arm="es_only",seed=5,budget=6)
            self.assertEqual(len(self.proposals),1)
    def test_invalid_fallback_is_observed_but_not_a_graph_label(self):
        with tempfile.TemporaryDirectory() as d:
            p=self.make(d,[None,None]); p.episode("unit",arm="qwen_base",seed=5,budget=6)
            q=read(Path(d)/"episodes/unit/query005.json")
            self.assertEqual(q["kind"],"explicit_invalid_fallback")
            self.assertIsNone(q["chosen_generation"]); self.assertIsNone(q["risk"])
            self.assertEqual(q["invalid_proposals"],2)

if __name__=="__main__": unittest.main()
