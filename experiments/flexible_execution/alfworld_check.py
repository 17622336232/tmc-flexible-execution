"""Replay the published walkthrough of one official ALFWorld validation game."""

import argparse
import fcntl
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
from time import monotonic
from zipfile import ZipFile

from experiments.task_attack.runtime import write_json


ARCHIVE_SHA256 = "5df77ea759f2211a4106082839ddbbb790f1ba4e7d097ed732cf453f72aa36cf"
ARCHIVE_BYTES = 36_507_267
RELEASE = "https://github.com/alfworld/alfworld/releases/tag/0.4.2"
SAMPLE = ("json_2.1.1/valid_seen/pick_and_place_simple-Candle-None-Cart-401/"
          "trial_T20190909_081908_677973/game.tw-pddl")


def run(archive, output):
    if archive.stat().st_size != ARCHIVE_BYTES or sha256(archive.read_bytes()).hexdigest() != ARCHIVE_SHA256:
        raise ValueError("ALFWorld text archive does not match the registered release")
    with ZipFile(archive) as zipped:
        payload = zipped.read(SAMPLE)
    game = json.loads(payload)
    if game.get("solvable") is not True or not {"pddl_domain", "pddl_problem", "grammar"} <= set(game):
        raise ValueError("The registered ALFWorld game is missing playable data")
    walkthrough = game["walkthrough"]
    if not isinstance(walkthrough, list) or not 1 <= len(walkthrough) <= 200 or not all(
            isinstance(command, str) and command for command in walkthrough):
        raise ValueError("The registered ALFWorld validation game has no usable official walkthrough")
    output.mkdir(parents=True, exist_ok=False)
    game_path = output / "game.tw-pddl"
    game_path.write_bytes(payload)
    source = {"mode": "ALFWORLD_TEXT_ONLY_DIAGNOSTIC", "release": RELEASE, "official_split": "valid_seen",
              "case": SAMPLE, "archive_sha256": ARCHIVE_SHA256, "game_sha256": sha256(payload).hexdigest(),
              "policy": "author-authored walkthrough replay", "actual_model_requests": 0, "visual_episodes": 0}
    write_json(output / "source.json", source)
    started = monotonic()
    actions = []
    env = None
    try:
        import textworld
        import textworld.gym
        from alfworld.agents.environment.alfred_tw_env import AlfredDemangler

        info = textworld.EnvInfos(won=True, admissible_commands=True, score=True)
        env_id = textworld.gym.register_game(str(game_path), info, max_episode_steps=200,
                                            wrappers=[AlfredDemangler()])
        env = textworld.gym.make(env_id)
        observation, state = env.reset()
        write_json(output / "initial_observation.json", {"observation": observation})
        score, ended = 0, False
        for command in walkthrough:
            admissible = command in state["admissible_commands"]
            observation, score, ended, state = env.step(command)
            actions.append({"action": command, "observation": observation, "score": int(score), "done": bool(ended),
                            "listed_as_admissible": admissible})
            if ended:
                break
        result = {"complete": bool(ended and state["won"]), "final_score": int(score),
                  "planned_actions": len(walkthrough),
                  "alfworld_version": version("alfworld"), "textworld_version": version("textworld")}
    except Exception as error:
        result = {"complete": False, "error": {"type": type(error).__name__, "detail": str(error)[:400]}}
    finally:
        if env is not None:
            env.close()
    write_json(output / "actions.json", actions)
    result.update(seconds=monotonic() - started, actions=len(actions), mode=source["mode"], visual_episodes=0)
    write_json(output / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("archive", "output", "simulator-lock"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    with args.simulator_lock.open("a+") as lease:
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("The shared simulator lock is held by another episode")
        result = run(args.archive, args.output)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
