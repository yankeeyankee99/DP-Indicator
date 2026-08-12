#!/usr/bin/env python3
"""Run DP-Indicator: explore, hypothesize, design, and report."""
import sys
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ['PYTHONUNBUFFERED'] = '1'

# The pipeline prints emoji/unicode status markers throughout. On Windows the
# console's default codepage (e.g. GBK/cp936) cannot encode them, which crashes
# the whole run with UnicodeEncodeError on the very first print(). Force UTF-8
# on stdout/stderr so this works regardless of the host locale.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

import asyncio
from dp_indicator.cli import cmd_run_all
import argparse

args = argparse.Namespace(
    api_key=os.environ.get('BH_API_KEY', ''),
    input='Explore new therapeutic indications for Kv1.3 pathway inhibitors',
    focus=None,
    func=cmd_run_all,
)

asyncio.run(args.func(args))
