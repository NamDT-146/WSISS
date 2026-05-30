#!/usr/bin/env python3
import subprocess, sys
from pathlib import Path
subprocess.run([sys.executable, str(Path(__file__).parent / "_run_exp.py"), "2C", *sys.argv[1:]], check=False)
