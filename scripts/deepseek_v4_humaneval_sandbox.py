"""Trusted container entry point; never execute this against samples on the host."""

import importlib.metadata
import json
import os
import sys


def main():
    assert os.getuid() == 65534, "Container-only unprivileged evaluator"
    from execution import check_correctness

    payload = json.load(sys.stdin)
    if payload.get("probe"):
        assert not os.path.exists("/mnt/disks/deepseek-models")
        assert not os.path.exists("/var/run/docker.sock")
        assert not os.path.exists("/home/koala")
        try:
            with open("/rootfs-write-probe", "x"):
                pass
        except OSError:
            readonly = True
        else:
            readonly = False
        assert readonly
        interfaces = os.listdir("/sys/class/net")
        assert interfaces == ["lo"]
        with open("/proc/self/status") as handle:
            status = dict(line.rstrip().split(":", 1) for line in handle if ":" in line)
        assert int(status["CapEff"].strip(), 16) == 0
        assert status["NoNewPrivs"].strip() == "1"
        assert status["Seccomp"].strip() == "2"
        import numpy

        versions = {
            name: importlib.metadata.version(name)
            for name in ("numpy", "fire", "tqdm", "termcolor")
        }
        assert versions == {
            "numpy": "2.2.6",
            "fire": "0.7.0",
            "tqdm": "4.67.1",
            "termcolor": "3.1.0",
        }
        assert numpy.add(1, 2) == 3
        print(
            json.dumps(
                {
                    "isolated": True,
                    "uid": os.getuid(),
                    "readonly": readonly,
                    "python": sys.version,
                    "dependencies": versions,
                }
            )
        )
        return
    results = []
    for task in payload["tasks"]:
        value = check_correctness(task["problem"], task["completion"], 3.0, 0)
        value["result"] = value["result"][:512]
        results.append(value)
    print(json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
