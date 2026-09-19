"""Stdlib-only transport; safe to import from the separate policy environment."""
import json
import os
from pathlib import Path
import selectors
import subprocess
import time


class OracleClient:
    def __init__(self, python, journal, *, seed, max_queries, classification,
                 source_root=None, protocol=None, protocol_sha256=None, timeout=180):
        here = Path(__file__).resolve().parent
        journal = Path(journal)
        journal.mkdir(parents=True, exist_ok=True)
        self.stderr = (journal / ("worker_" + str(time.time_ns()) + ".stderr.log")).open("x")
        command = [str(python), "-B", str(here / "snar_rpc.py"), "--journal", str(journal),
                   "--source-root", str(source_root or here / "vendor/summit"),
                   "--max-queries", str(max_queries), "--classification", classification]
        if protocol is not None:
            command += ["--protocol", str(protocol), "--protocol-sha256", protocol_sha256]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
                   OPENBLAS_NUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1")
        env.pop("PYTHONPATH", None)
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=self.stderr, text=True, bufsize=1, env=env)
        self.timeout, self.sequence = timeout, 0
        self.identity = self.request("initialize", {"seed": seed, "noise_level": 0.0})

    def request(self, method, params):
        self.sequence += 1
        request = {"jsonrpc": "2.0", "id": self.sequence, "method": method, "params": params}
        self.process.stdin.write(json.dumps(request, allow_nan=False) + "\n")
        self.process.stdin.flush()
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stdout, selectors.EVENT_READ)
            if not selector.select(self.timeout):
                raise TimeoutError("Oracle response pending; inspect durable intent, never blind-replay")
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("Oracle worker ended before response; inspect journal")
        response = json.loads(line)
        if response.get("jsonrpc") != "2.0" or response.get("id") != self.sequence:
            raise ValueError("Oracle response identity mismatch")
        if "error" in response:
            raise RuntimeError(json.dumps(response["error"], sort_keys=True))
        return response["result"]

    def evaluate(self, query_id, parameters):
        return self.request("evaluate", {"query_id": query_id, "parameters": parameters})

    def close(self):
        summary = self.request("close", {})
        self.process.stdin.close()
        code = self.process.wait(timeout=self.timeout)
        self.stderr.close()
        if code != 0:
            raise RuntimeError("Oracle worker exited with code " + str(code))
        return summary

    def pause(self):
        receipt = self.request("pause", {})
        self.process.stdin.close()
        code = self.process.wait(timeout=self.timeout)
        self.stderr.close()
        if code != 0:
            raise RuntimeError("Oracle worker exited with code " + str(code))
        return receipt
