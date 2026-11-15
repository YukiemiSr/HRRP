"""Run the 3 widths x 3 model conditions, at most one job per requested GPU."""

from __future__ import annotations

import argparse
from collections import deque
import os
from pathlib import Path
import subprocess
import sys
import time

from ..utils import write_json
from .backend import MODES
from .model import CONDITIONS
from .split import ROOT, WIDTHS, load_dataset, load_splits, read_setting


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    devices = parser.add_mutually_exclusive_group(required=True)
    devices.add_argument(
        "--gpus",
        nargs="+",
        help="Physical GPU indices, one simultaneous job per listed GPU.",
    )
    devices.add_argument(
        "--cpu",
        action="store_true",
        help="Serial CPU execution, useful for smoke checks.",
    )
    parser.add_argument(
        "--widths", nargs="+", type=int, choices=WIDTHS, default=list(WIDTHS)
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=CONDITIONS,
        default=["none", "yaw", "yaw_pitch"],
    )
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--cudnn-mode", choices=MODES, default="strict")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Keep selected weights and compact comparison evidence.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive.")
    slots = args.gpus if args.gpus else [""]
    if len(set(slots)) != len(slots) or any(not x.isdigit() for x in slots if x):
        parser.error("GPU indices must be unique nonnegative integers.")
    if len(set(args.widths)) != len(args.widths) or len(set(args.conditions)) != len(
        args.conditions
    ):
        parser.error("Widths and conditions must not repeat.")
    output = args.output_root.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite experiment root: {output}")
    project = ROOT.parents[1]
    jobs = []
    for width in args.widths:
        config = ROOT / f"configs/yaw{width}.json"
        cfg, path = read_setting(config)
        dataset = load_dataset(cfg, path, args.data_dir)
        load_splits(dataset, cfg, path)
        for condition in args.conditions:
            name = f"yaw{width}_{condition}"
            command = [
                sys.executable,
                "-u",
                "-m",
                "TCN.angles.train",
                "--config",
                str(config),
                "--conditioning",
                condition,
                "--device",
                "cpu" if args.cpu else "cuda",
                "--threads",
                str(args.threads),
                "--cudnn-mode",
                args.cudnn_mode,
                "--output",
                str(output / f"yaw{width}" / condition / "seed_56"),
            ]
            if args.data_dir:
                command.extend(
                    ["--data-dir", str(args.data_dir.expanduser().resolve())]
                )
            if args.smoke:
                command.append("--smoke")
            if args.compact:
                command.append("--compact")
            jobs.append({"name": name, "command": command, "status": "pending"})
    if args.dry_run:
        import shlex

        for i, job in enumerate(jobs):
            print(
                f"CUDA_VISIBLE_DEVICES={slots[i % len(slots)]} "
                + shlex.join(job["command"])
            )
        return
    output.mkdir(parents=True, exist_ok=False)
    (output / "logs").mkdir()
    manifest = {
        "phase": "smoke" if args.smoke else "train",
        "status": "running",
        "gpus": slots,
        "jobs": jobs,
        "compact": args.compact,
        "cudnn_mode": args.cudnn_mode,
    }
    pending = deque(jobs)
    running = {}
    failed = False
    try:
        while pending or running:
            for slot in slots:
                if slot not in running and pending and not failed:
                    job = pending.popleft()
                    log = (output / "logs" / f"{job['name']}.log").open("w")
                    env = {**os.environ, "CUDA_VISIBLE_DEVICES": slot}
                    process = subprocess.Popen(
                        job["command"],
                        cwd=project,
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                    job.update(status="running", gpu=slot, pid=process.pid)
                    running[slot] = (process, log, job)
                    print(
                        f"started {job['name']} GPU={slot or 'cpu'} pid={process.pid}",
                        flush=True,
                    )
            for slot, (process, log, job) in list(running.items()):
                code = process.poll()
                if code is not None:
                    log.close()
                    job.update(
                        status="complete" if code == 0 else "failed", exit_code=code
                    )
                    failed = failed or code != 0
                    del running[slot]
                    print(f"{job['status']} {job['name']} exit={code}", flush=True)
            if failed:
                while pending:
                    pending.popleft()["status"] = "not_started_after_failure"
            write_json(output / "launch_manifest.json", manifest)
            if running:
                time.sleep(0.5)
    except BaseException:
        for process, log, job in running.values():
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            log.close()
            job["status"] = "interrupted"
        manifest["status"] = "interrupted"
        write_json(output / "launch_manifest.json", manifest)
        raise
    manifest["status"] = "failed" if failed else "complete"
    write_json(output / "launch_manifest.json", manifest)
    if failed:
        raise RuntimeError(f"A run failed; inspect {output / 'logs'}.")


if __name__ == "__main__":
    main()
