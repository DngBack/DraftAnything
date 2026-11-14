"""Refresh the continuation report when a new completed benchmark appears."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/'runs/component_continuation_250k_v1'


def main():
    lock = os.open(RUN/'reporter.lock', os.O_CREAT | os.O_RDWR, 0o600)
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    previous = None
    while True:
        status = json.loads((RUN/'status.json').read_text())
        files = [p for p in sorted((RUN/'expanded').glob('benchmark_*.json')) if p.stem.split('_')[-1].isdigit()]
        signature = (files[-1].name if files else None, status['state'])
        if files and signature != previous:
            result = subprocess.run([status['command'][0], str(ROOT/'scripts/report_component_250k.py')],
                                    cwd=ROOT, capture_output=True, text=True)
            (RUN/'report.log').write_text(result.stdout+result.stderr)
            payload = {'pid': os.getpid(), 'state': status['state'], 'latest_benchmark': files[-1].name,
                       'report_exit_code': result.returncode}
            temp = RUN/'report_status.partial'
            temp.write_text(json.dumps(payload, indent=2)); temp.replace(RUN/'report_status.json')
            if result.returncode == 0: previous = signature
        if status['state'] not in ('prepared', 'running'): return
        time.sleep(30)


if __name__ == '__main__': main()
