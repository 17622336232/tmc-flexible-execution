"""Retrospective coding of the 13 distinct instruction forms, independent of A outputs."""
import csv
import re
from collections import Counter
from pathlib import Path

from experiments.flexible_execution.revision_audit import BASE, OUT, read, save, write_table


def design(instruction):
    # These labels are only for auditing the frozen corpus, never used by a policy.
    if re.match(r"找到编号为(?:EQ|BAY|CP)-\d+的",instruction):
        return "named-target",False,"The facts are required for the named target; finding the target may require movement."
    if instruction.startswith("盘点眼前这一排仓储货位，报告每个货位"):
        return "all-targets",False,"All object counts are required; a second view may still be needed for readability."
    if "候选中温度最低" in instruction:
        return "filtered-minimum",True,"The result can change which identity and evidence appear among tied minima."
    if "只报告两种点位" in instruction or "红箱超过2箱或蓝箱达到4箱" in instruction:
        return "compound-alternatives",True,"The compound filter selects identities for the final evidence set."
    if "黄色禁停线内已经停放车辆" in instruction:
        return "filtered-extra-field",True,"Vehicle color is a separate report field only for qualifying occupied spaces."
    if any(text in instruction for text in ("被实物堵住的通道编号","存在裂纹的设备编号",
                                            "实际箱数严格多于2箱的货位编号","同时配置了灭火器且通道没有被实物堵住的点位编号")):
        return "filtered-report",True,"The qualifying identities and supporting photos depend on a visual result."
    raise ValueError("Review previously uncoded instruction wording")


def main():
    root=BASE/"custom-final"
    with (OUT/"opportunities.csv").open() as stream:
        observed={r["case"]:r for r in csv.DictReader(stream)}
    rows=[]
    for c in read(root/"pilot_manifest.json")["cases"]:
        path=Path(c["study_task_file"]) if c.get("study_task_file") else root/"reference-scenes"/c["id"]/str(c["seed"])/"layout.json"
        task=read(path)["instruction"]
        role,filtered,reason=design(task)
        row=observed[c["id"]]
        rows.append({**row,"design":role,"instruction_filtered_evidence":filtered,
            "textual_reason":reason,"instruction":task,"separate_view_required_from_text":"undetermined",
            "additional_view_readability":"verified-for-recorded-read-only" if int(row["verified_photos_from_early_starts"]) else "unobserved",
            "scarce_physical_opportunity":"cannot infer from report filter alone"})
    write_table(OUT/"opportunities_with_instruction_reference.csv",rows)
    summary=dict(n=len(rows),conditional_report=sum(r["instruction_filtered_evidence"] for r in rows),
                 named_target=sum(r["design"]=="named-target" for r in rows),
                 all_target=sum(r["design"]=="all-targets" for r in rows),
                 filtered_and_runtime_submitted=sum(r["instruction_filtered_evidence"] and int(r["submitted_queries"])>0 for r in rows),
                 filtered_and_logged_conditional_material=sum(r["instruction_filtered_evidence"] and
                     r["branch_dependent_material_on_logged_contract"]=="True" for r in rows))
    save(OUT/"instruction_reference.json",summary)
    print(summary)


if __name__=="__main__":
    main()
