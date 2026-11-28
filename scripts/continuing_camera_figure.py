"""Display all four paired route results as actually obtained from the simulator."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from experiments.flexible_execution.revision_audit import digest, read


ROOT=Path("outputs/continuing/local-view-navigation01")
DEST=Path("outputs/continuing/results")


def main():
    cases=read(ROOT/"registration.json")["cases"]
    fig,axes=plt.subplots(len(cases),2,figsize=(9,3.05*len(cases)),squeeze=False,layout="constrained")
    for i,case in enumerate(cases):
        for j,arm in enumerate(("radial_same_distance","frontal_same_distance")):
            ax=axes[i,j]
            folder=ROOT/case["source_case"]/arm
            receipt=read(folder/"navigation.json")
            ax.set_xticks([]);ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_facecolor("#f1f3f3")
            if receipt["outward_arrived"]:
                rgb=folder/"calibration_camera.jpg"
                if digest(rgb)!=receipt["captured_sha256"]:
                    raise ValueError("Simulator RGB changed after capture")
                with Image.open(rgb) as photo:
                    ax.imshow(photo)
                labels=", ".join(r["value"] for r in receipt["readings"])
                caption=("OCR: "+labels) if labels else "OCR abstained"
                ax.set_xlabel(caption+"  |  "+("returned" if receipt["return_arrived"] else "return failed"),fontsize=10)
            else:
                ax.text(.5,.5,"Map rejected this route\nNo photograph acquired",ha="center",va="center",transform=ax.transAxes)
            if i==0:
                ax.set_title(("Original direction, matched distance","Facing the sign, matched distance")[j],fontsize=12)
            if j==0:
                ax.set_ylabel("Layout "+case["source_case"].split("-")[-1],fontsize=11)
    DEST.mkdir(parents=True,exist_ok=True)
    fig.savefig(DEST/"measured_views_all_four.png",dpi=145)
    plt.close(fig)

if __name__=="__main__":
    main()
