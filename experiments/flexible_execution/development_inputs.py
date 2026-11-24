"""Materialize separate RGB simulator layouts for calibration and selection."""

from collections import Counter
from hashlib import sha256
from pathlib import Path
from random import Random
from shutil import disk_usage
from xml.etree import ElementTree

from PIL import __version__ as PILLOW_VERSION

from experiments.task_attack.runtime import write_json
from experiments.task_attack.tasks import SCENES
from experiments.task_attack_recessed.scene import build_scene

from .runner import _hash, read_json


RECIPES = {
    "independent_work": ("cracks", "inventory", "parking"),
    "legal_preparation": ("cracks",),
    "cost_exceeds_gain": ("cracks", "large_inventory", "blocked_exit"),
    "permission_requires_result": ("cracks",),
    "evidence_changes": ("locate_condition", "inventory", "fire_and_access"),
    "no_opportunity": ("locate_crack", "locate_boxes", "locate_access"),
}


def scene_fingerprint(row):
    keys = ("layout_group_sha256", "asset_group_sha256", "instruction_sha256")
    return _hash({key: row[key] for key in keys})


def _cases(seed, splits):
    for role, count in splits.items():
        if count != 8 * len(RECIPES):
            raise ValueError("Development needs eight separate layouts per mechanism in each split")
        offset = {"calibration": 3_000_000, "selection": 3_200_000}[role]
        random = Random(_hash([seed, role, "visual-layouts"]))
        design = [(mechanism, index) for mechanism in RECIPES for index in range(8)]
        random.shuffle(design)
        layouts = random.sample(range(offset, offset + 100_000), count)
        states = random.sample(range(offset + 100_000, offset + 200_000), count)
        for position, ((mechanism, index), layout_seed, scene_seed) in enumerate(zip(design, layouts, states), 1):
            families = RECIPES[mechanism]
            family = families[index % len(families)]
            preparation = mechanism in {"legal_preparation", "permission_requires_result"}
            side = preparation or (mechanism == "cost_exceeds_gain" and family == "cracks")
            clause = None
            if mechanism == "legal_preparation":
                clause = "采集规则：允许在判读结果返回前采集以下材料：identity。"
            elif mechanism == "permission_requires_result":
                clause = "采集规则：只有判读结果为 positive 才允许采集以下材料：identity。"
            yield {"id": f"visual-{role}-{position:03d}", "split": role, "seed": scene_seed,
                   "layout_seed": layout_seed, "family": family, "domain": SCENES[family],
                   "equipment_count": 2 if index % 2 == 0 else 4,
                   "source_profile": "independent" if side else "single", "mechanism_requested": mechanism,
                   "collection_clause": clause, "backend": "cpu"}


def _verify_scene(case, scene):
    world = scene / "inspection.world"
    truth = read_json(scene / "private/truth.json")
    keys = ("seed", "layout_seed", "family", "source_profile")
    if any(truth.get(key) != case[key] for key in keys) or len(truth["devices"]) != case["equipment_count"]:
        raise ValueError("The generated scene does not match its assignment: " + case["id"])
    models = ElementTree.parse(world).getroot().find("world")
    coordinates = []
    for index, device in enumerate(truth["devices"], 1):
        station = f"{case['domain']}_station_{index:02d}"
        pose = models.find(f"./model[@name='{station}']/pose")
        if pose is None:
            raise ValueError("The generated world is missing a registered inspection station")
        xyz = [float(value) for value in pose.text.split()[:3]]
        if xyz != device["position"]:
            raise ValueError("The inspection station position does not match the rendered world")
        coordinates.append(xyz)
    textures = {path.name: sha256(path.read_bytes()).hexdigest()
                for path in sorted((scene / "assets/materials/textures").glob("*.png"))}
    if not textures:
        raise ValueError("The generated scene lacks physical visual materials")
    instruction = truth["task"] + ("\n" + case["collection_clause"] if case["collection_clause"] else "")
    return {**case, "base_group_id": case["id"], "layout_group_sha256": _hash(coordinates),
            "asset_group_sha256": _hash(textures), "world_sha256": sha256(world.read_bytes()).hexdigest(),
            "instruction_sha256": sha256(instruction.encode("utf8")).hexdigest(),
            "instruction": instruction, "source_directory": str(scene), "visual_episodes": 0}


def _reuse_or_build(case, root):
    marker = root / "layout.json"
    if not marker.is_file():
        build_scene(root, case["seed"], case["family"], case["equipment_count"], case["source_profile"],
                    layout_seed=case["layout_seed"])
        write_json(marker, _verify_scene(case, root))
    row = _verify_scene(case, root)
    if read_json(marker) != row:
        raise ValueError("An existing development layout changed: " + case["id"])
    return row


def _separate_asset_groups(output, seed, cases, rows):
    calibration = {row["asset_group_sha256"] for row in rows if row["split"] == "calibration"}
    retained = list(rows)
    changes = []
    used_seeds = {row["seed"] for row in rows}
    for index, (case, row) in enumerate(zip(cases, rows)):
        if row["split"] != "selection" or row["asset_group_sha256"] not in calibration:
            continue
        # No model results enter this check. Retain the generated geometry and
        # replace only the overlapping station materials in the selection split.
        rng = Random(_hash([seed, case["id"], "independent-asset-source"]))
        for alternative in rng.sample(range(3_400_000, 3_500_000), 8):
            if alternative in used_seeds:
                continue
            staged = output / "selection" / "asset-replacements" / f"{case['id']}-{alternative}"
            candidate = _reuse_or_build({**case, "seed": alternative}, staged)
            if candidate["asset_group_sha256"] in calibration:
                continue
            if candidate["layout_group_sha256"] != row["layout_group_sha256"]:
                raise ValueError("A texture replacement changed the existing layout")
            retained[index] = candidate
            used_seeds.add(alternative)
            changes.append({"id": case["id"], "previous_seed": row["seed"], "accepted_seed": alternative,
                            "previous_source_directory": row["source_directory"],
                            "reason": "exact asset digest was shared across development roles"})
            break
        else:
            raise ValueError("Could not assign a distinct asset source to " + row["id"])
    return retained, changes


def prepare(output, defaults):
    data = defaults["data"]
    splits = {"calibration": data["development_calibration_instances"],
              "selection": data["development_selection_instances"]}
    if sum(splits.values()) != data["development_instances"]:
        raise ValueError("Development splits do not match the registered total")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if disk_usage(output).free < defaults["resources"]["minimum_free_disk_gb"] * 1024**3:
        raise OSError("Disk reserve is below the registered minimum")
    cases = list(_cases(data["split_seed"], splits))
    design = {"role": "VISUAL_DEVELOPMENT_LAYOUTS", "split_seed": data["split_seed"], "cases": cases,
              "asset_renderer": {"package": "Pillow", "version": PILLOW_VERSION}}
    plan = output / "layout_plan.json"
    if plan.is_file() and read_json(plan) != design:
        raise ValueError("The existing development assignment differs from this protocol")
    if not plan.is_file():
        write_json(plan, design)
    rows = []
    for case in cases:
        root = output / case["split"] / case["id"]
        rows.append(_reuse_or_build(case, root))
    if len({row["layout_group_sha256"] for row in rows}) != len(rows):
        raise ValueError("Two registered development cases share the same actual geometry")
    counts = {role: dict(Counter(row["mechanism_requested"] for row in rows if row["split"] == role))
              for role in splits}
    manifest = {"role": design["role"], "state": "LAYOUTS_PREPARED", "layout_plan_sha256": _hash(design),
                "asset_renderer": design["asset_renderer"],
                "layout_count": len(rows), "visual_episode_count": 0, "planned_mechanisms": counts,
                "distinct_asset_groups": len({row["asset_group_sha256"] for row in rows}), "cases": rows}
    initial = output / "split_manifest.json"
    if initial.is_file() and read_json(initial) != manifest:
        raise ValueError("The recorded initial layout manifest changed")
    if not initial.is_file():
        write_json(initial, manifest)
    retained, changes = _separate_asset_groups(output, data["split_seed"], cases, rows)
    final = {**manifest, "cases": retained, "reassigned_assets": changes,
             "split_roles_share_exact_assets": False,
             "distinct_asset_groups": len({row["asset_group_sha256"] for row in retained})}
    final_path = output / "split_manifest_v2.json"
    if final_path.is_file() and read_json(final_path) != final:
        raise ValueError("The recorded separate source groups changed")
    if not final_path.is_file():
        write_json(final_path, final)
    return {key: value for key, value in final.items() if key != "cases"}
