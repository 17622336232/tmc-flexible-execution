"""Make a contact sheet of actual recorded images, with an auditable source index."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from experiments.flexible_execution.revision_audit import BASE, OUT, digest, read, write_table


def main():
    examples=[]
    def attach(title,caption,p,case):
        examples.append(dict(title=title,caption=caption,case=case,path=str(p),sha256=digest(p)))
    pub=read(BASE/"formal-progress/first-public-closed-loop.json")
    ep=next(e for e in pub["methods"] if e["method"]=="conditional-a")
    p=Path(ep["artifact_dir"])
    assert read(p/"report.json")["side_sha256"]==digest(p/"side.jpg")
    attach("Early evidence used", "Side photo entered a passing report.", p/"side.jpg",pub["case"])
    permit=read(BASE/"formal-progress/registered-early-results.json")
    def arm(number):
        case=next(c for c in permit["cases"] if c["case"].endswith(number))
        return case, next(e for e in case["episodes"] if e["method"]=="conditional-a")
    case,ep=arm("112")
    w=next(a for a in ep["early_action_witnesses"] if a["outcome"]=="negative" and a["photo_seconds_before_reply"]>0)
    p=Path(ep["artifact_dir"])/"private/robot"/(w["photo_id"]+".jpg")
    assert digest(p)==w["photo_sha256"]
    attach("Early photo unused","Negative result made the label unnecessary.",p,case["case"])
    case,ep=arm("152")
    assert not ep["early_action_witnesses"]
    attach("Early movement declined","Permission granted; cost check declined.",Path(ep["artifact_dir"])/"private/robot/probe_0001.jpg",case["case"])
    case,ep=arm("172")
    claim=read(OUT/"shared_fixes.json")["evidence_projection"]["verified_sources"][0]
    p=Path(ep["artifact_dir"])/"private/robot"/(claim["photo"]+".jpg")
    assert digest(p)==claim["image_sha256"]
    attach("Fact recovered in replay","Crack fact verified; historical task failed.",p,case["case"])
    fig,axes=plt.subplots(2,2,figsize=(9,6.7))
    for ax,e in zip(axes.flat,examples):
        with Image.open(e["path"]) as img:
            ax.imshow(img.convert("RGB"))
        ax.set(title=e["title"],xlabel=e["caption"],xticks=[],yticks=[])
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT/"recorded_examples.png",dpi=165)
    plt.close(fig)
    write_table(OUT/"recorded_examples.csv",examples)


if __name__=="__main__":
    main()
