"""One supervised driver; at most one new GPU policy process at a time."""
import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time
from study import append_event,canonical,digest,read,write_once

ROOT=Path(__file__).resolve().parent

def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda:f.read(1024*1024),b""): h.update(part)
    return h.hexdigest()

def verify_sources(registration):
    for item in registration["sources"]:
        if file_hash(item["path"])!=item["sha256"]:
            raise RuntimeError("Registered source changed: "+item["path"])

def main():
    p=argparse.ArgumentParser(); p.add_argument("--runtime",required=True); p.add_argument("--output",required=True)
    a=p.parse_args(); output=Path(a.output).resolve(); output.mkdir(parents=True,exist_ok=True)
    runtime=read(a.runtime); registration=read(output/"registration.json")
    verify_sources(registration)
    if registration["protocol_fingerprint"]!=digest(read(ROOT/"protocol.json")):
        raise RuntimeError("Protocol fingerprint changed")
    env=dict(os.environ,OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",NUMEXPR_NUM_THREADS="2",
             TOKENIZERS_PARALLELISM="false",PYTHONDONTWRITEBYTECODE="1",PYTHONUNBUFFERED="1")
    env["PYTHONPATH"]=str(ROOT)+os.pathsep+runtime["method_source_root"]
    command=["-B",str(ROOT/"pipeline.py"),"--runtime",a.runtime,"--output",str(output)]
    def run(stage,receipt,arm=None,branch=None):
        if (output/receipt).exists(): return
        verify_sources(registration)
        python=runtime["oracle_python"] if stage=="freeze" or arm in ("random","gp_ei_scalarized") else runtime["policy_python"]
        args=[python,*command,"--stage",stage]+(["--arm",arm] if arm else [])+(["--branch",branch] if branch else [])
        label=stage+("_"+arm if arm else "")+("_"+branch if branch else "")
        with (output/f"{label}.log").open("a") as log:
            append_event(output/"driver_events.jsonl",{"event":"stage_start","stage":stage,"arm":arm,"argv":args})
            result=subprocess.run(args,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        append_event(output/"driver_events.jsonl",{"event":"stage_exit","stage":stage,"arm":arm,"returncode":result.returncode})
        if result.returncode or not (output/receipt).exists(): raise RuntimeError("Stage did not commit: "+label)
    run("collect","collection_complete.json")
    run("fit_risk","risk_fit.json")
    run("evolve","full_evolution_complete.json",branch="full")
    run("evolve","es_only_evolution_complete.json",branch="es_only")
    run("freeze","adaptation_complete.json")
    # One CPU baseline worker may overlap one GPU test worker, only after the
    # exact common prior and selected policy are frozen. No second GPU model.
    cpu_args=[runtime["oracle_python"],"-B",str(ROOT/"driver.py"),"--runtime",a.runtime,"--output",str(output),"--cpu-tests-only"]
    cpu_log=(output/"cpu_baseline_driver.log").open("a")
    cpu=subprocess.Popen(cpu_args,cwd=ROOT,env=env,stdout=cpu_log,stderr=subprocess.STDOUT)
    try:
        run("evaluate","test_qwen_base_complete.json","qwen_base")
        run("evaluate","test_uq_esopt_complete.json","uq_esopt")
        run("evaluate","test_uq_only_complete.json","uq_only")
        run("evaluate","test_es_only_complete.json","es_only")
        run("evaluate","test_random_controller_complete.json","random_controller")
        if cpu.wait()!=0: raise RuntimeError("CPU baseline worker failed; original evidence retained")
    finally:
        if cpu.poll() is None: cpu.terminate(); cpu.wait(timeout=30)
        cpu_log.close()
    subprocess.run([runtime["oracle_python"],"-B",str(ROOT/"report.py"),"--output",str(output)],check=True,env=env)

def cpu_main():
    # No torch import or GPU allocation in this branch.
    parser=argparse.ArgumentParser(); parser.add_argument("--runtime",required=True); parser.add_argument("--output",required=True)
    parser.add_argument("--cpu-tests-only",action="store_true"); args=parser.parse_args()
    root=Path(args.output); runtime=read(args.runtime); registration=read(root/"registration.json")
    for arm in ("random","gp_ei_scalarized"):
        if (root/f"test_{arm}_complete.json").exists(): continue
        verify_sources(registration)
        command=[runtime["oracle_python"],"-B",str(ROOT/"pipeline.py"),"--runtime",args.runtime,"--output",args.output,"--stage","evaluate","--arm",arm]
        with (root/f"evaluate_{arm}.log").open("a") as log:
            subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        if not (root/f"test_{arm}_complete.json").exists(): raise RuntimeError("CPU test did not commit")

if __name__=="__main__":
    if "--cpu-tests-only" in sys.argv: cpu_main()
    else: main()
