#!/usr/bin/env python3
"""Build the verified official QE 7.3.1 source for CrystalGym, CPU only.

Uses only this project's fresh micromamba compiler environment. QE's own CMake
code fetches external submodules at external/submodule_commit_hash_records pins.
Only pw.x and ev.x are built/copied; no vendor source or host environment is edited.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


QE_ARCHIVE_SHA256 = "2c58b8fadfe4177de5a8b69eba447db5e623420b070dea6fd26c1533b081d844"
SSSP_ARCHIVE_MD5 = "a58f1b3373f330179fd0832c48bb9a52"


def digest(path, algorithm="sha256"):
    value = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""):
            value.update(block)
    return value.hexdigest()


def freeze_toolchain(root, build):
    """Record compiler packages and numerical library bytes used by this build."""
    tool_env = root/"environments/qe-build"
    inventory = json.loads(subprocess.check_output(
        [str(root/"tools/micromamba"), "--no-rc", "list", "-p", str(tool_env), "--json"], text=True))
    packages = inventory["packages"] if isinstance(inventory, dict) else inventory
    libraries = {}
    for name in ("libopenblas.so.0", "libfftw3.so", "libfftw3_omp.so", "libmpi.so", "libgfortran.so.5", "libgomp.so.1"):
        path = (tool_env/"lib"/name).resolve()
        if not path.is_file():
            raise RuntimeError(f"Missing built runtime library: {name}")
        libraries[name] = {"path": str(path), "sha256": digest(path), "size_bytes": path.stat().st_size}
    cache = build/"CMakeCache.txt"
    return {
        "compiler_packages": [{key: item.get(key) for key in ("name", "version", "build_string", "channel")} for item in packages],
        "numerical_libraries": libraries,
        "cmake_cache": {"path": str(cache), "sha256": digest(cache)},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--build-dir", required=True, type=Path)
    parser.add_argument("--jobs", type=int, choices=(2,3,4), default=4)
    parser.add_argument("--stage", choices=("configure", "build", "freeze", "all"), default="all")
    args = parser.parse_args()
    root, source, build = args.project.resolve(), args.source.resolve(), args.build_dir.resolve()
    tool_env, install = root/"environments/qe-build", root/"tools/qe-7.3.1"
    archive = root/"cache/downloads/qe-7.3.1-verified.tar.gz"
    if digest(archive) != QE_ARCHIVE_SHA256:
        raise RuntimeError("QE archive does not match the verified official 7.3.1 archive.")
    if not (source/"external/submodule_commit_hash_records").is_file():
        raise RuntimeError("Expected official QE submodule pin records are missing.")
    for line in (source/"external/submodule_commit_hash_records").read_text().splitlines():
        if not line.strip():
            continue
        commit, name = line.split()
        path = source/"external"/name
        if (path/".git").exists():
            actual = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
            dirty = subprocess.check_output(["git", "-C", str(path), "diff", "--name-only", "HEAD"], text=True).strip()
            if actual != commit or dirty:
                raise RuntimeError(f"Incomplete or modified official external source: {name}")
    blas = tool_env/"lib/libopenblas.so.0"
    if not blas.is_file():
        raise RuntimeError("The fresh build environment lacks libopenblas.so.0.")
    prefix = [str(root/"tools/micromamba"), "--no-rc", "run", "-p", str(tool_env)]
    environment = dict(os.environ)
    environment.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1",
                       MKL_NUM_THREADS="1", CMAKE_BUILD_PARALLEL_LEVEL=str(args.jobs),
                       OMPI_ALLOW_RUN_AS_ROOT="1", OMPI_ALLOW_RUN_AS_ROOT_CONFIRM="1")
    (root/"logs").mkdir(exist_ok=True)
    records = []

    def run(name, command):
        log_path = root/"logs"/f"{name}.log"
        start = time.monotonic()
        print(json.dumps({"stage": name, "status": "started", "command": command, "log": str(log_path)}), flush=True)
        with log_path.open("a") as log:
            log.write("\nBUILD INVOCATION " + datetime.now(timezone.utc).isoformat() + "\n")
            log.flush()
            result = subprocess.run(command, cwd=root, env=environment, stdout=log, stderr=subprocess.STDOUT)
        record = {"stage": name, "returncode": result.returncode, "wall_seconds": time.monotonic()-start, "log": str(log_path)}
        records.append(record)
        print(json.dumps(record), flush=True)
        if result.returncode:
            raise RuntimeError(f"{name} failed; see {log_path}")

    configuration = prefix + ["cmake", "-S", str(source), "-B", str(build),
        "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_POLICY_VERSION_MINIMUM=3.5",
        f"-DCMAKE_INSTALL_PREFIX={install}", f"-DCMAKE_PREFIX_PATH={tool_env}",
        "-DCMAKE_Fortran_COMPILER=mpif90", "-DCMAKE_C_COMPILER=mpicc",
        "-DQE_ENABLE_OPENMP=ON", "-DQE_ENABLE_MPI=ON", "-DQE_ENABLE_SCALAPACK=OFF",
        "-DQE_ENABLE_HDF5=OFF", "-DQE_ENABLE_CUDA=OFF", "-DQE_ENABLE_OPENACC=OFF",
        # QE consumes libMBD's Fortran API. Its unused optional C API collides
        # with the new ISO_C_BINDING F_C_STRING intrinsic in gfortran 16.
        "-DENABLE_C_API=OFF",
        "-DBLA_VENDOR=OpenBLAS", f"-DBLAS_LIBRARIES={blas}", f"-DLAPACK_LIBRARIES={blas}",
        f"-DCMAKE_INSTALL_RPATH={tool_env/'lib'}", "-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON"]
    if args.stage in {"configure", "all"}:
        run("qe-cpu-configure", configuration)
    if args.stage in {"build", "all"}:
        run("qe-cpu-build", prefix + ["cmake", "--build", str(build), "--target", "qe_pw_exe", "qe_pw_tools_ev_exe", "--parallel", str(args.jobs)])
    if args.stage in {"freeze", "all"}:
        (install/"bin").mkdir(parents=True, exist_ok=True)
        binaries = {}
        for name in ("pw.x", "ev.x"):
            candidates = [build/"bin"/name, build/"PW"/name]
            binary = next((p for p in candidates if p.is_file()), None)
            if binary is None:
                raise RuntimeError(f"Requested binary {name} not found in this fresh build.")
            target = install/"bin"/name
            shutil.copy2(binary, target)
            target.chmod(target.stat().st_mode | 0o111)
            binaries[name] = {"path": str(target), "sha256": digest(target), "size_bytes": target.stat().st_size}
        sssp_archive = root/"cache/downloads/SSSP_1.3.0_PBE_efficiency.tar.gz"
        if digest(sssp_archive, "md5") != SSSP_ARCHIVE_MD5:
            raise RuntimeError("SSSP archive MD5 mismatch.")
        pseudo_dir = root/"data/assets/SSSP_1.3.0_PBE_efficiency"
        upfs = {p.name: {"sha256": digest(p), "size_bytes": p.stat().st_size}
                for p in sorted(pseudo_dir.iterdir()) if p.is_file() and p.suffix.lower() == ".upf"}
        if not upfs:
            raise RuntimeError("SSSP UPF extraction is empty.")
        submodules = {}
        for line in (source/"external/submodule_commit_hash_records").read_text().splitlines():
            if not line.strip():
                continue
            commit, name = line.split()
            path = source/"external"/name
            if (path/".git").exists():
                actual = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
                if actual != commit:
                    raise RuntimeError(f"QE external {name} checkout differs from official pin.")
                submodules[name] = {"commit": commit, "path": str(path)}
        config = {"schema_version": 1, "version": "7.3.1", "build_mode": "cpu_mpi_openmp",
            "gpu_enabled": False, "build_parallel_jobs": args.jobs, "qe_dir": str(install),
            "source_dir": str(source), "build_dir": str(build), "compiler_environment": str(tool_env),
            "source_archive": {"path": str(archive), "sha256": QE_ARCHIVE_SHA256},
            "official_external_submodules": submodules, "configure_command": configuration,
            "install_mode": "copy_only_requested_official_cmake_targets_pw_and_ev_with_fixed_install_rpath",
            "binaries": binaries, "mpirun": str(tool_env/"bin/mpirun"),
            "path_prepend": [str(tool_env/"bin"), str(install/"bin")],
            "ld_library_path_prepend": [str(tool_env/"lib")],
            "environment": {"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "1",
                            "MKL_NUM_THREADS": "1", "OMPI_ALLOW_RUN_AS_ROOT": "1", "OMPI_ALLOW_RUN_AS_ROOT_CONFIRM": "1"},
            "sssp": {"version": "1.3.0_PBE_efficiency", "directory": str(pseudo_dir),
                     "archive_path": str(sssp_archive), "archive_md5": SSSP_ARCHIVE_MD5,
                     "archive_sha256": digest(sssp_archive), "upf_count": len(upfs), "files": upfs},
            "operations": records, "runtime_dft_verified": False,
            "runtime_verification_note": "Set true only after a separate technical_not_main real CrystalGym terminal DFT result."}
        config.update(freeze_toolchain(root, build))
        output = root/"configs/qe.runtime.json"
        output.write_text(json.dumps(config, indent=2) + "\n")
        print(json.dumps({"status": "built_and_frozen", "runtime_config": str(output), "binaries": binaries, "upf_count": len(upfs)}), flush=True)


if __name__ == "__main__":
    main()
