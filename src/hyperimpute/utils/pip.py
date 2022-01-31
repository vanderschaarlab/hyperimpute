# stdlib
from pathlib import Path
import subprocess
import sys

# hyperimpute absolute
import hyperimpute.logger as log

current_dir = Path(__file__).parent

predefined = {
    "catboost": "catboost==1.0.3",
    "scipy": "scipy==1.7.3",
    "torch": "torch==1.9.1",
    "xgboost": "xgboost==1.5.1",
    "miracle-imputation": "git+https://github.com/vanderschaarlab/MIRACLE@main",
}


def install(packages: list) -> None:
    for package in packages:
        install_pack = package
        if package in predefined:
            install_pack = predefined[package]
        log.error(f"Installing {install_pack}")

        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", install_pack],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except BaseException as e:
            log.error(f"failed to install package {package}: {e}")
