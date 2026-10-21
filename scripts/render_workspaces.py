"""Render every agent workspace setup with the real generator entry points.

Golden check for the workspace generator: each case builds the files an agent would
see (role x observation profile x environments/subagents, options, plus Claude), then
records a hash index. A refactor must reproduce the index byte for byte unless a
change is intended; `--out` writes the trees so two revisions can be diffed.

Usage:
  render_workspaces.py --index local_tests/goldens/workspaces.json   # write the index
  render_workspaces.py --check local_tests/goldens/workspaces.json   # compare
  render_workspaces.py --out DIR [--case NAME]                        # write trees
"""
import argparse
import base64
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from services.benchmarks import DEFAULT  # noqa: E402
from services.controller.config import Configuration, Limits  # noqa: E402
from services.storage_root import artifact_root  # noqa: E402
from services.dexjoco.objects import PRIVILEGED_ENV  # noqa: E402
from services.dexjoco.lookahead import KEPT_ENV as LOOKAHEAD_ENV  # noqa: E402

PROFILES = ('rgbd', 'rgb-only')
LAYOUTS = ((1, 0), (3, 0), (1, 2))  # (exploration environments, subagents)
# Configuration of each benchmark's cases.
BENCHMARK_CASES = {
    # DexJoCo: fresh exploration layouts (collection 0), the official evaluation episodes (collection 1).
    'dexjoco': dict(benchmark='dexjoco', task='pick_bucket', exploration_seeds=(0, 1, 2), formal_seed=0,
                    formal_eval_seed=1),
}


def configuration(root, **changes):
    limits = Limits(60, 512, 1, 32, 32, 4 * 1024 * 1024)
    # Validated under the artifact root like a real session, then rendered into ``root``.
    case = {**BENCHMARK_CASES[DEFAULT], **changes}
    config = Configuration(artifact_root()/'workspace-goldens', case.pop('task'),
                           case.pop('exploration_seeds'), case.pop('formal_seed'), 0, 'GPU-aaaa',
                           limits, limits, limits, demonstration_context='none',
                           formal_eval_seed=case.pop('formal_eval_seed'), benchmark=case.pop('benchmark'))
    config = replace(config, **case)
    object.__setattr__(config, 'root', Path(root))  # Rendering only: no artifact storage.
    Path(root).mkdir(parents=True, exist_ok=True)
    return config


# A tiny fixed 8x6 grey JPEG standing in for every demonstration image.
TINY_JPEG = base64.b64decode(
    '/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDABALDA4MChAODQ4SERATGCgaGBYWGDEjJR0oOjM9PDkzODdASFxOQERXRTc4UG1R'
    'V19iZ2hnPk1xeXBkeFxlZ2P/2wBDARESEhgVGC8aGi9jQjhCY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2Nj'
    'Y2NjY2NjY2NjY2NjY2P/wAARCAAGAAgDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAA'
    'AgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6'
    'Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXG'
    'x8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREA'
    'AgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5'
    'OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPE'
    'xcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwAooooA/9k=')


def demonstration_stub(config, workspace, cache_root, *, record=True):
    """A fixed demonstration in the providers' layout: one terminal.jpg, or a three-image
    completion sequence under frames/; no video cache."""
    kind = config.demonstration_context
    target = workspace/'runtime/demonstrations'
    target.mkdir(parents=True, exist_ok=True)
    images = (['terminal.jpg'] if kind == 'terminal_state' else
              [f'frames/frame_{index:06d}.jpg' for index in range(3)])
    for name in images:
        (target/name).parent.mkdir(parents=True, exist_ok=True)
        (target/name).write_bytes(TINY_JPEG)
    (target/'manifest.json').write_text(json.dumps({
        'schema_version': 1, 'task': config.task, 'context_kind': kind, 'demonstrator': 'robot',
        'image_count': len(images), 'ordered_images': images, 'contains': ['rgb_images']}, indent=2) + '\n')
    return {'kind': kind, 'image_count': len(images), 'root': 'runtime/demonstrations',
            'manifest_path': 'runtime/demonstrations/manifest.json'}


def demonstration_context(demonstrations):
    """demonstrations: False, True (one terminal-state image) or a demonstration_context kind."""
    if not demonstrations:
        return {}
    return {'demonstration_context': 'terminal_state' if demonstrations is True else demonstrations}


def auto_research(target, *, profile='rgbd', envs=1, subagents=0, agent_formal=False, claude=False, handoff=False,
                  benchmark=None, demonstrations=False, experts=0, expert_content='full', task=None):
    from harness.codex_cli import auto_research as module
    case = {**BENCHMARK_CASES.get(benchmark, {}), **({'task': task} if task else {})}
    config = configuration(target.parent/'root', **case,
                           observation_profile=profile, exploration_envs=envs,
                           subagents=subagents, agent_formal=agent_formal, write_handoff=handoff,
                           agent_cli='claude' if claude else 'codex',
                           agent_provider='claude-login' if claude else 'openrouter',
                           expert_demonstrations=experts, expert_content=expert_content,
                           **demonstration_context(demonstrations))
    target.mkdir()
    provision = module.provision_demonstration
    if demonstrations:
        module.provision_demonstration = demonstration_stub
    try:
        module.populate_workspace(config, target)
    finally:
        module.provision_demonstration = provision
    if claude:
        from harness.claude_cli.auto_research import claude_workspace
        claude_workspace(target)


def formal_copy(target, *, profile='rgbd', claude=False, benchmark=None, demonstrations=False, experts=0,
                expert_content='full', task=None):
    from harness.codex_cli import auto_research as module
    from services.mcp_contract import Contract
    case = {**BENCHMARK_CASES.get(benchmark, {}), **({'task': task} if task else {})}
    config = configuration(target.parent/'root', **case,
                           observation_profile=profile, agent_formal=True,
                           agent_cli='claude' if claude else 'codex',
                           agent_provider='claude-login' if claude else 'openrouter',
                           expert_demonstrations=experts, expert_content=expert_content,
                           **demonstration_context(demonstrations))
    target.mkdir()
    provision = module.provision_demonstration
    if demonstrations:
        module.provision_demonstration = demonstration_stub
    try:
        module.populate_workspace(config, target, contract=Contract.from_config(config, isolated=True),
                                  formal_episode=True)
    finally:
        module.provision_demonstration = provision
    if claude:
        from harness.claude_cli.auto_research import claude_workspace
        claude_workspace(target)


def cases():
    result = {}
    # DexJoCo: both profiles, without expert episodes (the setup default) or with three.
    dexjoco = dict(benchmark='dexjoco', experts=0)
    for profile in PROFILES:
        for envs, subagents in LAYOUTS:
            result[f'dexjoco/auto-research/exploration/{profile}/env{envs}-sub{subagents}'] = (
                auto_research, dict(dexjoco, profile=profile, envs=envs, subagents=subagents))
        result[f'dexjoco/auto-research/agent-formal/{profile}/env1-sub0'] = (
            auto_research, dict(dexjoco, profile=profile, agent_formal=True))
        result[f'dexjoco/auto-research/formal-copy/{profile}'] = (formal_copy, dict(dexjoco, profile=profile))
    result['dexjoco/auto-research/agent-formal/rgbd/env3-sub0'] = (
        auto_research, dict(dexjoco, envs=3, agent_formal=True))
    result['dexjoco/auto-research/exploration-experts/rgbd/env1-sub0'] = (
        auto_research, dict(dexjoco, experts=3))
    result['dexjoco/auto-research/agent-formal-experts/rgb-only/env1-sub0'] = (
        auto_research, dict(dexjoco, experts=3, profile='rgb-only', agent_formal=True))
    result['dexjoco/auto-research/formal-copy-experts/rgbd'] = (formal_copy, dict(dexjoco, experts=3))
    result['dexjoco/auto-research/exploration-bimanual/rgbd/env1-sub0'] = (
        auto_research, dict(dexjoco, task='bimanual_hanoi'))
    result['dexjoco/auto-research/exploration-random/rgb-only/env1-sub0'] = (
        auto_research, dict(dexjoco, task='pick_bucket_random', profile='rgb-only'))
    result['dexjoco/auto-research/demonstrations/rgbd/env1-sub0'] = (
        auto_research, dict(dexjoco, demonstrations='completion_sequence'))
    result['dexjoco/auto-research/exploration-claude/rgbd/env1-sub0'] = (auto_research, dict(dexjoco, claude=True))
    result['dexjoco/auto-research/formal-copy-claude/rgbd'] = (formal_copy, dict(dexjoco, claude=True))
    return result


def facts(name, workspace):
    """The setup facts the validator checks a rendered case against (services/workspace/validate.py)."""
    from services.mcp_contract import Contract
    builder, arguments = cases()[name]
    contract = json.loads((Path(workspace)/'MCP_CONTRACT.json').read_text())
    envs = contract['exploration_envs']
    subagents = contract['subagents'] if envs == 1 else 0
    formal = builder is formal_copy
    live_copy = bool(arguments.get('agent_formal')) and not formal
    tools = Contract(contract['phase'], contract['observation_profile'], envs, subagents,
                     contract.get('benchmark', DEFAULT), bundle_tools=not live_copy).tool_names
    if set(contract['tools']) != tools | ({'status'} if formal else set()):
        raise AssertionError(f'{name}: MCP_CONTRACT.json tools differ from the setup')
    return dict(tools=tools | ({'status'} if formal else set()),  # The formal copy's local episode status.
                cli='claude' if arguments.get('claude') else 'codex', environments=envs,
                subagents=0 if formal else max(subagents, envs if envs > 1 else 0),
                benchmark=contract.get('benchmark', DEFAULT),
                demonstrations=bool(arguments.get('demonstrations')),
                demonstration_kind=demonstration_context(arguments.get('demonstrations')).get('demonstration_context'),
                expert_demonstrations=arguments.get('experts', 0),
                expert_content=arguments.get('expert_content', 'full'),
                formal_copy=formal, live_copy=live_copy)


def problems(name, workspace):
    from services.workspace.validate import validate
    known = frozenset(path.parent.name for path in (PROJECT/'agent_skills').rglob('SKILL.md'))
    return validate(workspace, known_skills=known, **facts(name, workspace))


def render_case(name, directory):
    builder, arguments = cases()[name]
    target = directory/'workspace'
    builder(target, **arguments)
    return target


def index(tree):
    """Path -> sha256 (plus mode) of every entry an agent sees; links recorded by target."""
    tree = Path(tree)
    entries = {}
    for path in sorted(tree.rglob('*')):
        relative = path.relative_to(tree).as_posix()
        if '__pycache__' in path.parts:
            continue
        if path.is_symlink():
            target = os.readlink(path)
            entries[relative] = 'link:' + target
        elif path.is_file() and not any(p.is_symlink() for p in path.parents if p != tree and tree in p.parents):
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            entries[relative] = digest + (' x' if os.access(path, os.X_OK) else '')
    return entries


def render_all(selected=None):
    """The index of every case at the default privileged-state and lookahead settings, whatever
    the shell sets (the goldens record the defaults; --out renders at the shell's for reviews)."""
    result = {}
    saved = {name: os.environ.pop(name) for name in (PRIVILEGED_ENV, *sorted(LOOKAHEAD_ENV)) if name in os.environ}
    try:
        for name in cases():
            if selected and name not in selected:
                continue
            with tempfile.TemporaryDirectory(prefix='workspace-golden-') as scratch:
                result[name] = index(render_case(name, Path(scratch)))
    finally:
        os.environ.update(saved)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--index', type=Path, help='write the hash index')
    action.add_argument('--check', type=Path, help='compare against a hash index')
    action.add_argument('--out', type=Path, help='write every rendered tree under this directory')
    parser.add_argument('--case', action='append', help='limit to these case names')
    args = parser.parse_args()
    if args.out:
        for name in cases():
            if args.case and name not in args.case:
                continue
            directory = args.out/name
            if directory.exists():
                shutil.rmtree(directory)
            directory.mkdir(parents=True)
            found = problems(name, render_case(name, directory))
            print(directory, *(['', *found] if found else []), sep='\n  ' if found else '')
        return
    rendered = render_all(args.case)
    if args.index:
        args.index.parent.mkdir(parents=True, exist_ok=True)
        args.index.write_text(json.dumps(rendered, indent=1, sort_keys=True) + '\n')
        print(f'{len(rendered)} cases, {sum(map(len, rendered.values()))} entries -> {args.index}')
        return
    expected = json.loads(args.check.read_text())
    differences = []
    for name in sorted(set(expected) | set(rendered)):
        if args.case and name not in args.case:
            continue
        old, new = expected.get(name, {}), rendered.get(name, {})
        for path in sorted(set(old) | set(new)):
            if old.get(path) != new.get(path):
                state = 'added' if path not in old else 'removed' if path not in new else 'changed'
                differences.append(f'{name}: {state} {path}')
    print('\n'.join(differences) or 'identical')
    raise SystemExit(1 if differences else 0)


if __name__ == '__main__':
    main()
