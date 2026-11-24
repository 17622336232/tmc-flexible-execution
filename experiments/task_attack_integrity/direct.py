"""Direct effect-vector baseline, with identical observation/compute protocol."""
import argparse
import json
from pathlib import Path

from .belief import run
from .learned import DispatchValuePolicy
from experiments.task_attack.runtime import write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--gateway',required=True);p.add_argument('--policy',choices=('direct_effect',),required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--timing-model',type=Path,required=True);p.add_argument('--public-perception',type=Path,required=True)
    p.add_argument('--effort',default='high');p.add_argument('--call-budget',type=int,default=1);a=p.parse_args()
    try:
        run(a.gateway,'belief',a.output,json.loads(a.timing_model.read_text()),json.loads(a.public_perception.read_text()),
            predictor_type=DispatchValuePolicy,method_label='direct_effect')
    except (Exception,KeyboardInterrupt) as error:
        path=a.output/'result.json';summary=json.loads(path.read_text()) if path.exists() else {}
        summary.update(decision='observer_error' if not isinstance(error,KeyboardInterrupt) else 'interrupted_at_episode_end',error_type=type(error).__name__)
        write_json(path,summary)
        if not isinstance(error,KeyboardInterrupt):raise


if __name__=='__main__':main()
