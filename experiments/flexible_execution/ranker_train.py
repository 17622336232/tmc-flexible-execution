"""Train and evaluate matched CPU rankers on grouped counterfactual data."""

import json
from statistics import mean
from time import monotonic

import numpy as np

from experiments.task_attack.runtime import write_json

from .ranker_data import LABEL_MODE, STATES_PER_GROUP, generate
from .runner import _hash


SIZES = (1000, 3000, 10000)
SEEDS = (17, 29, 43)
LABELS = {"ordinary": "success_label", "cost": "cost_gain_label"}
CHECK_BUDGETS = (1, 2, 4, 8)
STOP_THRESHOLDS = (0., .005, .01, .02)


def _load_jsonl(path):
    with path.open(encoding="utf8") as handle:
        return [json.loads(line) for line in handle]


def _data(directory, count, seed):
    directory.mkdir(parents=True, exist_ok=True)
    result = {}
    for split, size in (("train", count), ("calibration", 48), ("selection", 48)):
        path = directory / f"{split}-{size}.jsonl"
        if not path.is_file():
            pending = path.with_suffix(".jsonl.tmp")
            with pending.open("w", encoding="utf8") as handle:
                for row in generate(split, size, seed):
                    handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            pending.replace(path)
        rows = _load_jsonl(path)
        if len(rows) != size or any(row["mode"] != LABEL_MODE or row["split"] != split for row in rows):
            raise ValueError("The stored training set does not match its planned fold or size")
        result[split] = rows
    groups = [{row["group_sha256"] for row in rows} for rows in result.values()]
    if len(groups[0]) != (count + STATES_PER_GROUP - 1) // STATES_PER_GROUP or len(groups[1]) != 48 or len(groups[2]) != 48:
        raise ValueError("The data set does not have the required independent scene counts")
    if any(groups[first].intersection(groups[second]) for first, second in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("A scene source crossed the locked data split")
    return result


def _train(rows, objective, seed):
    selected = [row for state in rows for row in state["candidates"]]
    if not selected:
        raise ValueError("Training needs labelled candidate outcomes")
    features = np.asarray([row["features"] for row in selected], dtype=np.float64)
    label = np.asarray([row[LABELS[objective]] for row in selected], dtype=np.float64)
    offset = features.mean(axis=0)
    scale = np.maximum(features.std(axis=0), 1e-6)
    features = (features - offset) / scale
    features = np.column_stack((features, np.ones(len(features))))
    rng = np.random.default_rng(seed)
    weights = np.zeros(features.shape[1], dtype=np.float64)
    for _ in range(20):
        order = rng.permutation(len(features))
        for start in range(0, len(order), 128):
            batch = order[start:start + 128]
            samples = features[batch]
            error = samples @ weights - label[batch]
            weights -= .03 * ((samples.T @ error) / len(batch) + .0001 * weights)
    return {"objective": objective, "seed": seed, "weights": weights.tolist(),
            "offset": offset.tolist(), "scale": scale.tolist(), "trained_candidates": len(selected),
            "training_state_count": len(rows)}


def _scores(candidates, model):
    if model is None:
        return [-row["features"][1] for row in candidates]
    x = np.asarray([row["features"] for row in candidates])
    x = (x - model["offset"]) / model["scale"]
    return x @ np.asarray(model["weights"][:-1]) + model["weights"][-1]


def evaluate(rows, model, cap, threshold=None):
    results, checked = [], []
    started = monotonic()
    for state in rows:
        candidates = state["candidates"]
        scores = _scores(candidates, model) if candidates else []
        order = sorted(range(len(candidates)), key=lambda index: (-scores[index], candidates[index]["candidate"]))
        visited = 0
        chosen = None
        for index in order[:cap]:
            if threshold is not None and visited and scores[index] <= threshold:
                break
            candidate = candidates[index]
            visited += 1
            if candidate["verified_common_continuation"] and candidate["estimated_safe_gain"] > 0:
                if chosen is None or candidate["estimated_safe_gain"] > chosen["estimated_safe_gain"]:
                    chosen = candidate
        baseline = mean(row["restricted_seconds"] for row in state["baseline"])
        results.append(chosen["counterfactual_restricted_seconds"] if chosen else baseline)
        checked.append(visited)
    return {"states": len(rows), "mean_restricted_seconds": mean(results),
            "mean_candidates_verified": mean(checked), "selector_wall_seconds": monotonic() - started}


def experiments(directory, sizes, split_seed):
    if any(size not in SIZES for size in sizes):
        raise ValueError("Use one of the preregistered training sizes")
    directory.mkdir(parents=True, exist_ok=True)
    protocol = {"label_mode": LABEL_MODE, "split_seed": split_seed, "seeds": list(SEEDS), "epochs": 20,
                "optimizer": "cpu_squared_loss_minibatch_128_rate_0.03", "training_sizes": list(SIZES),
                "candidate_check_caps": list(CHECK_BUDGETS), "stop_thresholds": list(STOP_THRESHOLDS)}
    lock = directory / "protocol.json"
    if lock.is_file() and json.loads(lock.read_text(encoding="utf8")) != protocol:
        raise ValueError("The ranking diagnostic protocol is already frozen")
    if not lock.is_file():
        write_json(lock, protocol)
    data = _data(directory, max(sizes), split_seed)
    results = []
    for size in sizes:
        training = data["train"][:size]
        checkpoint_key = _hash(training)
        for seed in SEEDS:
            for objective in LABELS:
                path = directory / "models" / f"{size}-{seed}-{objective}.json"
                if path.is_file():
                    model = json.loads(path.read_text(encoding="utf8"))
                    if model["training_sha256"] != checkpoint_key or model["protocol_sha256"] != _hash(protocol):
                        raise ValueError("The model checkpoint does not match the frozen training data")
                else:
                    model = {**_train(training, objective, seed), "training_sha256": checkpoint_key,
                             "protocol_sha256": _hash(protocol)}
                    write_json(path, model)
                fixed = [{"cap": cap, **evaluate(data["selection"], model, cap)} for cap in CHECK_BUDGETS]
                row = {"training_states": size, "seed": seed, "objective": objective, "fixed": fixed}
                if objective == "cost":
                    calibrated = [(threshold, evaluate(data["calibration"], model, max(CHECK_BUDGETS), threshold))
                                  for threshold in STOP_THRESHOLDS]
                    threshold, calibration = min(calibrated, key=lambda pair: (
                        pair[1]["mean_restricted_seconds"], pair[1]["mean_candidates_verified"], pair[0]))
                    row["adaptive"] = {"threshold": threshold, "calibration": calibration,
                                       "selection": evaluate(data["selection"], model, max(CHECK_BUDGETS), threshold)}
                results.append(row)
    summary = {"mode": LABEL_MODE, "status": "DIAGNOSTIC_ONLY", "run_rows": results,
               "fixed_heuristic": [{"cap": cap, **evaluate(data["selection"], None, cap)} for cap in CHECK_BUDGETS]}
    write_json(directory / "latest_summary.json", summary)
    return summary
