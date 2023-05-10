#!/usr/bin/env python3

import argparse
import datetime
import tempfile
from pathlib import Path
import os
import subprocess

project_root_path = Path(os.path.dirname(os.path.realpath(__file__))).parent
print(project_root_path)
parser = argparse.ArgumentParser(description="Create release artifacts.")
parser.add_argument(
    "destination", help="Destination folder for the package.", type=Path
)
parser.add_argument("version", help="Package version in x.x.x format.", type=str)

args = parser.parse_args()

package_path = Path(args.destination)
package_version = args.version

Path(package_path).mkdir(parents=True, exist_ok=True)


def get_artifact(config: str):
    path = Path(tmpdirname + "/" + config)
    path.mkdir()

    env = os.environ.copy()
    env["CXX"] = "clang++"

    subprocess.run(
        [
            "cmake",
            "-DCMAKE_BUILD_TYPE=" + config,
            "-DENABLE_DPCPP=on",
            "-DENABLE_CUDA=on",
            "-S",
            project_root_path,
            "-B",
            str(path),
        ],
        env=env,
    )
    subprocess.run(["cmake", "--build", str(path)], env=env)
    subprocess.run(
        ["cmake", "--install", str(path), "--prefix", str(path) + "/install"], env=env
    )
    subprocess.run(
        [
            "tar",
            "-czvf",
            str(
                package_path / ("dbench-" + package_version + "-" + config + ".tar.gz")
            ),
            "-C",
            str(path) + "/install",
            ".",
        ]
    )


with tempfile.TemporaryDirectory() as tmpdirname:
    get_artifact("Release")
