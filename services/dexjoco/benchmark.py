"""DexJoCo (Franka Panda + Allegro hands, official evaluation) hooks for the shared harness;
see services/benchmarks.py.

Import-light: the host never imports dexjoco or mujoco, and task, rule and demonstration
modules load inside the hooks.
"""
from pathlib import Path


def tool_definitions():
    from services.dexjoco.tools import tool_definitions
    return tool_definitions()


def backend(config):
    from services.dexjoco.backend import DexjocoBackend
    return DexjocoBackend(config)


def task_names():
    from services.dexjoco.tasks import task_names
    return task_names()


def task_timeouts(task):
    from services.dexjoco.tasks import task_timeouts
    return task_timeouts(task)


def validate_config(config):
    from services.dexjoco.tasks import EXPLORATION_COLLECTION, FORMAL_COLLECTION, official_task
    from services.dexjoco.task_rules import TASK_RULES
    official_task(config.task)
    if (config.eval_seed, config.formal_collection) != (EXPLORATION_COLLECTION, FORMAL_COLLECTION):
        raise ValueError('DexJoCo explores its own layouts (collection 0) and evaluates the official evaluation '
                         'layouts (collection 1)')
    if config.task_rules not in TASK_RULES:
        raise ValueError(f"DexJoCo task_rules must be one of {', '.join(TASK_RULES)}")


def validate_settings(values):
    """Dashboard setup values (fixed fields already applied): formal uses the official evaluation layouts."""
    from services.dexjoco.tasks import FORMAL_LAYOUTS
    if values['formal_seed'] + values['formal_episodes'] > FORMAL_LAYOUTS:
        raise ValueError(f'DexJoCo has {FORMAL_LAYOUTS} official evaluation layouts per task: '
                         'first formal layout + formal episodes must be at most that')


def validate_layouts(config):
    from services.dexjoco.tasks import validate_layouts as dexjoco_layouts
    dexjoco_layouts(config.task, config.formal_collection,
                    range(config.formal_seed, config.formal_seed + config.formal_episodes))
    dexjoco_layouts(config.task, config.eval_seed, config.exploration_seeds)


def exploration_seeds(task, collection, start, episodes, **_):
    """Exploration layout ids map to seeds outside the official evaluation seeds, so never repeat."""
    return tuple(range(start, start + episodes))


def task_context(config, demonstration, *, live_formal):
    from services.dexjoco.task_rules import task_context
    return task_context(config.task, task_rules=config.task_rules, live_formal=live_formal,
                        evaluation=not config.agent_formal)


def refresh_workspace(config, workspace):
    """An exploring agent's TASK.md and AGENTS.md follow the current episode-review setting."""
    from services.dexjoco.task_rules import review_section
    from services.episode_reviews.workspace import refresh
    refresh(config, workspace, review_section)


def reference_mounts(config):
    """Read-only reference data for the agent: the task's expert episodes."""
    if not config.expert_demonstrations:
        return []
    from services.dexjoco.demonstrations import expert_root
    return [(expert_root(config.task), '/workspace/runtime/experts')]


def fetch_demonstration(demos_root, task, kind):
    if kind != 'none':
        from services.dexjoco.demonstrations import cached_experts
        cached_experts(Path(demos_root)/'dexjoco', task)  # installed by setup_dexjoco_demos.py


def provision_demonstration(config, target, demos_root):
    from services.dexjoco.demonstrations import provision_trial_demonstration
    return provision_trial_demonstration(cache_root=Path(demos_root)/'dexjoco', target=target,
                                         task=config.task, context_kind=config.demonstration_context)


def episode_reviewer(root):
    """Closing an exploration episode or rehearsal starts its review while reviews are on."""
    from services.episode_reviews.runner import EpisodeReviews
    return EpisodeReviews(root)


def review_analysis():
    """Episode reviews read the goal report: success, the end reason, the environment's own info,
    and the task objects' reset and final poses."""
    from services.dexjoco import analysis
    return analysis


def task_robot(task):
    """Per-task robot words: the task's own evaluation cameras (its official camera_mapping)."""
    from services.dexjoco.tasks import official_config, task_info
    cameras = tuple(official_config(task)['camera_mapping'].values())
    overview = next(c for c in cameras if 'wrist' not in c)
    wrists = [f'`{c}`' for c in cameras if 'wrist' in c]
    words = {'overview_camera': overview, 'cameras': cameras, 'wrist_images': ' or '.join(wrists)}
    words['arms'] = 'dual' if task_info(task)['dual_arm'] else 'single'
    # Experiment (exp/add-state-info/object): dexjoco_object_states' level (RAX_PRIVILEGED_STATE).
    from services.dexjoco.objects import privileged_level
    words['privileged'] = privileged_level()
    if task_info(task)['dual_arm']:
        words['overview'] = 'third-person (behind the robots)'
    return words


def unscoped_call(name, arguments):
    """dexjoco_rotation_math: pure rotation and pose arithmetic for action rows, with or without an episode."""
    from services.dexjoco.rotation_math import TOOL, rotation_math
    if name != TOOL:
        raise ValueError(f'{name!r} is not a DexJoCo unscoped tool')
    return rotation_math(arguments)
