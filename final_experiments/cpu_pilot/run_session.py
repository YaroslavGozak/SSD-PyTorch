"""Execute a fresh smoke and full pilot, preserving logs and exit status."""
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

OUTPUT = Path(__file__).resolve().parent
ROOT = OUTPUT.parents[1]
status = {"status": "RUNNING", "started_utc": datetime.now(timezone.utc).isoformat(), "stages": []}

def save():
    (OUTPUT / "session_status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")

save()
for stage in ("smoke", "all"):
    command = [sys.executable, "-u", "-m", "experiments.cpu_pilot.run_pilot", "--output-dir", str(OUTPUT), "--stage", stage]
    with (OUTPUT / f"{stage}_execution.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    status["stages"].append({"stage": stage, "command": command, "returncode": result.returncode})
    save()
    if result.returncode:
        status["status"] = "STOPPED_REVIEW_LOG"
        save()
        sys.exit(result.returncode)
status["status"] = "COMPLETE_PENDING_REVIEW"
status["finished_utc"] = datetime.now(timezone.utc).isoformat()
save()
