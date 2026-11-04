import json,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parents[1]
out=root/'final_experiments/pilot'
status={'status':'RUNNING','stages':[],'validation_evidence':'final_experiments/01_audit/mobilenet_completed_sweeps.json'}
def save():
 (out/'restart_status.json').write_text(json.dumps(status,indent=2)+'\n')
save()
for stage in ('smoke','all'):
 with (out/f'restart_{stage}.log').open('w',encoding='utf-8') as log:
  result=subprocess.run([sys.executable,'-u','-m','experiments.cpu_pilot.run_pilot','--stage',stage],cwd=root,stdout=log,stderr=subprocess.STDOUT)
 status['stages'].append({'stage':stage,'returncode':result.returncode})
 save()
 if result.returncode:
  status['status']='STOPPED_REVIEW_LOG'; save(); sys.exit(result.returncode)
status['status']='COMPLETE_PENDING_REVIEW'; save()

