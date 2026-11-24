"""Import OWN trusted model files locally; nothing is added to Git.

Numeric model parameters are preserved. Absolute file names and source/build
identity contracts are rebound to this portable source checkout. The old
contract hash remains as provenance; this is a new local build, not an original
binary-identical benchmark. No inference or flight is performed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
HELPERS = ROOT/'research/attack_learning'
ASSETS = ('model.pkl', 'ours_critic_stopping.pkl', 'blind_stopping.pkl',
          'direct_stopping.pkl', 'known_critic_stopping.pkl')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def local_source(name):
    normalized = str(name).replace('\\', '/')
    if 'operator_diagnosis_20261115/' in normalized:
        normalized = 'research/attack_learning/'+normalized.rsplit('operator_diagnosis_20261115/', 1)[1]
    marker = '/Diff-VLA/'
    if marker in normalized:
        normalized = normalized.split(marker, 1)[1]
    path = Path(normalized)
    if path.is_absolute():
        raise ValueError('不能映射的源码路径：'+str(name))
    result = (ROOT/path).resolve()
    if ROOT not in result.parents:
        raise ValueError('源码路径超出仓库')
    return result


def rebind_hashes(mapping):
    result = {}
    for name in mapping:
        path = local_source(name)
        if not path.is_file():
            raise ValueError('模型契约需要未提供的源码：'+str(path.relative_to(ROOT)))
        result[str(path.relative_to(ROOT))] = sha(path)
    return result


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, separators=(',', ':'))+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path, required=True, help='自己的 isolated_waiting_candidate_01 目录')
    parser.add_argument('--baseline', type=Path, help='可选：自己的视觉攻击 baseline_bundle 目录')
    parser.add_argument('--supporting-models', type=Path, help='可选：自己的旧 models/task_attack 目录，供研究训练脚本使用')
    args = parser.parse_args()
    if not (ROOT/'sim_ws/devel/lib/vla_sim/vla_diff_planner_node').is_file():
        parser.error('先运行 bash scripts/build_sim.sh，再导入，以绑定本机实际编译。')
    source = args.candidate.resolve()
    old_contract = json.loads((source/'candidate_contract.json').read_text())
    if sha(source/'bundle.json') != old_contract['bundle_sha256']:
        parser.error('输入模型包的 bundle.json 与原契约不一致。')
    for name in ASSETS:
        if sha(source/name) != old_contract['asset_sha256'][name]:
            parser.error('输入模型文件校验失败：'+name)
    target = ROOT/'models/task_attack/showcase_candidate'
    if target.exists():
        parser.error('目标模型目录已存在。先将它重命名备份，再重新导入，不覆盖既有模型。')
    baseline_target = ROOT/'models/task_attack/baseline_bundle'
    if args.baseline:
        if not (args.baseline/'bundle.json').is_file():
            parser.error('baseline 输入目录缺少 bundle.json。')
        if baseline_target.exists():
            parser.error('baseline_bundle 已存在，未覆盖；请省略 --baseline 或先将已有目录改名备份。')
    if args.supporting_models and not args.supporting_models.is_dir():
        parser.error('--supporting-models 必须是自己的旧 models/task_attack 目录。')
    bundle = json.loads((source/'bundle.json').read_text())
    bundle['feature_source_sha256'] = rebind_hashes(bundle['feature_source_sha256'])
    candidate = bundle['conditional_critic_candidate']
    candidate['model'] = str(target/'model.pkl')
    candidate['stopping'] = {arm: str(target/Path(path).name) for arm, path in candidate['stopping'].items()}
    runtime = json.loads((source/'runtime_contract.json').read_text())
    runtime['training_feature_source_sha256'] = rebind_hashes(runtime['training_feature_source_sha256'])
    # Everything above validates before writing the new local directory.
    target.mkdir(parents=True)
    for name in ASSETS:
        shutil.copy2(str(source/name), str(target/name))
    write(target/'bundle.json', bundle)
    runtime['model_sha256'] = sha(target/'bundle.json')
    write(target/'runtime_contract.json', runtime)
    sys.path[:0] = [str(ROOT), str(ROOT/'bridge'), str(HELPERS)]
    from experiments.task_attack_sensor_proxy.audit import implementation_hash
    contract = dict(old_contract)
    contract.update(native_sha256=implementation_hash(), bundle_sha256=sha(target/'bundle.json'),
        asset_sha256={name: sha(target/name) for name in ASSETS},
        extension_source_sha256={path.name: sha(path) for path in HELPERS.glob('*.py')},
        portable_import=dict(original_contract_sha256=sha(source/'candidate_contract.json'),
            original_bundle_sha256=old_contract['bundle_sha256'], numeric_model_parameters_changed=False,
            source_and_build_contract_rebound=True, historical_results_imported_as_new_results=False))
    write(target/'candidate_contract.json', contract)
    if args.baseline:
        shutil.copytree(str(args.baseline.resolve()), str(baseline_target))
    if args.supporting_models:
        for child in args.supporting_models.resolve().iterdir():
            if not child.is_dir() or child.name in ('baseline_bundle', 'showcase_candidate'):
                continue
            destination = ROOT/'models/task_attack'/child.name
            if not destination.exists():
                shutil.copytree(str(child), str(destination))
    print('已导入本机 models/task_attack/showcase_candidate；数值权重与原包一致。')
    print('模型目录被 .gitignore 排除，不会上传。下一步：bash scripts/doctor.sh --require-build --require-model')


if __name__ == '__main__':
    main()
