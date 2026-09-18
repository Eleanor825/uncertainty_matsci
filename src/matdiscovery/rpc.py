"""Audited process boundary between a policy and the official scientific environment."""

from __future__ import annotations

import json
import os
from pathlib import Path
import selectors
import subprocess
import time


class EnvironmentRPCError(RuntimeError):
    def __init__(self, response):
        super().__init__(str(response.get("error", response)))
        self.response = response


class EnvironmentClient:
    def __init__(self, project: Path, benchmark: str, work_dir: Path, *, timeout=3600):
        self.project, self.benchmark, self.timeout = Path(project).resolve(), benchmark, timeout
        self.work_dir = Path(work_dir).resolve()
        self.work_dir.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        for name in ("PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV", "CONDA_PREFIX"):
            env.pop(name, None)
        env.update(PYTHONPATH=str(self.project / "src"), PYTHONNOUSERSITE="1", PYTHONHASHSEED="0", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
        if benchmark == "crystalgym":
            env["CUDA_VISIBLE_DEVICES"] = ""
            env["OPENBLAS_NUM_THREADS"] = "1"
            qe_env = self.project / "environments/qe-build"
            env["PATH"] = str(qe_env / "bin") + os.pathsep + env.get("PATH", "")
            env["LD_LIBRARY_PATH"] = str(qe_env / "lib") + os.pathsep + env.get("LD_LIBRARY_PATH", "")
            env["OMPI_ALLOW_RUN_AS_ROOT"] = "1"
            env["OMPI_ALLOW_RUN_AS_ROOT_CONFIRM"] = "1"
        python = self.project / "environments" / ("made-py312" if benchmark == "made" else "crystalgym-py311") / "bin/python"
        self.stderr = (self.work_dir / "environment.stderr.log").open("a")
        self.audit = (self.work_dir / "rpc.jsonl").open("a", buffering=1)
        self.process = subprocess.Popen([str(python), "-m", "matdiscovery.env_server", "--benchmark", benchmark], cwd=self.project, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.stderr, text=True, bufsize=1)
        self.sequence = 0
        self.closed = False

    def request(self, operation, arguments=None, *, check=True):
        if self.closed:
            raise RuntimeError("Environment process already closed")
        request = {"id": self.sequence, "op": operation, "args": arguments or {}}
        self.sequence += 1
        started = time.time()
        self.audit.write(json.dumps({"time": started, "direction": "request", "payload": request}, allow_nan=False) + "\n")
        self.audit.flush()
        os.fsync(self.audit.fileno())
        self.process.stdin.write(json.dumps(request, allow_nan=False) + "\n")
        self.process.stdin.flush()
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stdout, selectors.EVENT_READ)
            if not selector.select(self.timeout):
                raise TimeoutError("Environment request still has unknown outcome; do not automatically replay it")
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"Environment exited without a response; inspect {self.work_dir / 'environment.stderr.log'}")
        response = json.loads(line)
        if response.get("id") != request["id"]:
            raise RuntimeError("Environment response correlation mismatch")
        self.audit.write(json.dumps({"time": time.time(), "elapsed_seconds": time.time() - started, "direction": "response", "payload": response}, allow_nan=False) + "\n")
        self.audit.flush()
        if check and not response.get("ok"):
            raise EnvironmentRPCError(response)
        return response["result"] if check else response

    def close(self):
        if self.closed:
            return
        try:
            if self.process.poll() is None:
                self.request("close", check=False)
                self.process.wait(timeout=30)
        finally:
            self.closed = True
            self.audit.close()
            self.stderr.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        if exc_type is None:
            self.close()
        else:
            # An in-flight DFT call must not be silently killed or replayed.
            self.audit.write(json.dumps({"time": time.time(), "worker_exception": str(exc), "environment_pid": self.process.pid, "outcome": "requires_reconciliation"}) + "\n")
            self.audit.flush()
