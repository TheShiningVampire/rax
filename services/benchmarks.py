"""Simulator backends behind the shared harness: one registry entry per benchmark.

Pure configuration, like ``services.mcp_contract``: no simulator or heavy imports at module
load. Each benchmark is a peer: an entry here holds its data (robot tool names, robot facts,
setup defaults and fixed fields, capabilities), and its adapter module
(``services/<benchmark>/benchmark.py``) implements the hooks the shared harness calls. Shared
code dispatches through the entry and never compares benchmark names. The benchmark is fixed
at session creation; configurations and manifests without the field use DEFAULT.
"""
from dataclasses import dataclass, field
import importlib

DEFAULT = 'dexjoco'  # Configurations and manifests without a benchmark field.

DEXJOCO_TOOLS = frozenset({
    'dexjoco_status', 'dexjoco_observe', 'dexjoco_step', 'dexjoco_hold', 'dexjoco_pixel_to_position',
    'dexjoco_object_states', 'dexjoco_rotation_math', 'dexjoco_preview_describe', 'dexjoco_preview_run',
})


@dataclass(frozen=True, eq=False)  # Identity: entries are singletons with dict fields.
class Benchmark:
    name: str
    label: str                      # Display name: 'DexJoCo'.
    adapter: str                    # Module implementing HOOKS.
    robot_tools: frozenset
    geometry_tools: frozenset       # Removed by the RGB-only profile.
    unscoped_tools: frozenset       # Pure computation: not environment-scoped, served without an episode.
    mutating: frozenset             # Tools that may advance physics.
    previewable: frozenset          # preview_only=true makes these planning-only.
    status_tool: str
    observe_tool: str
    tool_prefixes: tuple
    episode_tool_prefixes: tuple    # Tools whose replies carry the live episode's step_id.
    simulator_modules: tuple        # Simulator server modules (the episode server first).
    startup_seconds: int            # Environment creation deadline (seconds).
    # Words shared workspace sources use for the simulated robot instead of branching on it.
    robot: dict = field(default_factory=dict)
    partial_credit: bool = False    # Episodes are scored with partial credit, which workspaces may name.
    # Formal episodes earn a graded native score (partial credit in the average); False: each
    # episode succeeds or fails and the result is the share of successes.
    graded_score: bool = True
    wall_limit: bool = True         # Episodes have a wall-clock limit (task_timeouts episode_seconds).
    episode_reviews: bool = False   # Simulator ground truth supports exploration episode reviews.
    # Planning and stepping sources (project-relative) an offline parallel formal batch
    # fingerprints, so a change to them during the batch stops it.
    planner_files: tuple = ()
    # Workspaces take the research-method skills (agent_skills/method/); off until each
    # benchmark's workspaces are reviewed with them.
    method_skills: bool = False
    # Regex of this benchmark's task nouns, which its workspaces' skills never name ('' for none):
    # skills describe the method and the robot, TASK.md the task.
    task_nouns: str = ''
    # Regex of this benchmark's own robot, camera, tool and skill names, which another
    # benchmark's workspace never mentions ('' for none).
    vocabulary: str = ''
    task_tag: str = ''              # Suffix of the task in the dashboard's task menu.
    score_label: str = 'Formal average score'
    layout_note: str = ''           # Dashboard note on the layouts exploration and formal use.
    # Setup values applied when the dashboard's task switches to this benchmark (editable).
    setup_defaults: dict = field(default_factory=dict)
    # configure_auto_research.py defaults for flags the operator did not pass.
    cli_defaults: dict = field(default_factory=dict)
    # Setup fields that do not apply: name -> (fixed value, reason).
    fixed: dict = field(default_factory=dict)

    def _hooks(self):
        return importlib.import_module(self.adapter)

    def __getattr__(self, name):
        # The adapter's hooks (HOOKS), loaded only when first called.
        if name not in HOOKS:
            raise AttributeError(name)
        return getattr(self._hooks(), name)

    def planning_only(self, name, arguments):
        return name in self.previewable and isinstance(arguments, dict) and arguments.get('preview_only') is True

    def definitions(self):
        return [t for t in self._hooks().tool_definitions() if t['name'] in self.robot_tools]

    def robot_facts(self, task=None):
        """The robot words for one task: the entry's, with the adapter's optional per-task
        overrides (``task_robot(task)``), e.g. a task's own overview camera."""
        hook = getattr(self._hooks(), 'task_robot', None) if task else None
        return {**self.robot, **(hook(task) if hook else {})}

    def apply_fixed(self, values):
        """values with this benchmark's fixed setup fields."""
        return {**values, **{name: value for name, (value, _) in self.fixed.items()}}


# What every adapter module implements (besides tool_definitions(), the robot tools' schemas):
#   backend(config)                        the controller backend of one environment
#   task_names(), task_timeouts(task)      task catalog and budgets
#   validate_config(config)                benchmark rules of a session configuration
#   validate_settings(values)              benchmark rules of dashboard setup values
#   validate_layouts(config)               formal and exploration layouts exist
#   exploration_seeds(task, collection, start, episodes, *, formal_collection, formal_seed,
#                     formal_episodes)     one layout per exploration episode
#   task_context(config, demonstration, *, live_formal)   TASK.md text
#   refresh_workspace(config, workspace)   update instructions on deploy and resume
#   fetch_demonstration(demos_root, task, kind)           cache visual context before deploy
#   provision_demonstration(config, target, demos_root)   copy it into a workspace
#   episode_reviewer(root)                 reviewer of closed episodes, or None
#   review_analysis()                      the episode-review analysis module (ground truth and
#                                          wording; services/episode_reviews), or None
#   reference_mounts(config)               read-only reference data for the agent, such as expert
#                                          episodes: (host path, path under /workspace) pairs
#   unscoped_call(name, arguments)         the JSON reply of one of the entry's unscoped_tools: pure
#                                          computation, with or without an episode
HOOKS = frozenset({
    'backend', 'task_names', 'task_timeouts', 'validate_config', 'validate_settings', 'validate_layouts',
    'exploration_seeds', 'task_context', 'refresh_workspace', 'provision_demonstration',
    'fetch_demonstration', 'episode_reviewer', 'review_analysis', 'reference_mounts', 'unscoped_call',
})

BENCHMARKS = {
    'dexjoco': Benchmark(
        name='dexjoco', label='DexJoCo', adapter='services.dexjoco.benchmark',
        robot_tools=DEXJOCO_TOOLS, geometry_tools=frozenset({'dexjoco_pixel_to_position'}),
        unscoped_tools=frozenset({'dexjoco_rotation_math'}),
        mutating=frozenset({'dexjoco_step', 'dexjoco_hold'}),
        previewable=frozenset(), status_tool='dexjoco_status', observe_tool='dexjoco_observe',
        tool_prefixes=('dexjoco_',),
        # The DexJoCo preview reports the live step as `live_step_id`, so it is never an episode step.
        episode_tool_prefixes=('dexjoco_',),
        simulator_modules=('services.dexjoco.server', 'services.dexjoco.preview_server'), startup_seconds=300,
        # The official DexJoCo control: no motion planner ('plan' empty); the overview camera is
        # per task (services/dexjoco/benchmark.py task_robot).
        robot={'benchmark': 'DexJoCo', 'status': 'dexjoco_status', 'observe': 'dexjoco_observe',
               'step': 'dexjoco_step', 'plan': '', 'pixel': 'dexjoco_pixel_to_position',
               # Experiment (exp/add-state-info/object): the privileged object-state tool.
               'objects': 'dexjoco_object_states',
               'hz': 30,
               'overview_camera': 'front', 'overview': 'third-person',
               'cameras': ('front', 'random_camera', 'ego', 'ego_right', 'wrist', 'wrist_left', 'wrist_right'),
               # Per-task values come from task_robot(); these are the single-arm defaults.
               'arms': 'single', 'wrist_images': '`wrist`',
               # The skill that characterizes how the controller tracks commands (method skills link it).
               'tracking_skill': 'robot-interface'},
        graded_score=False, method_skills=True, episode_reviews=True,
        planner_files=('services/dexjoco/session.py', 'services/simulator/protocol.py'),
        vocabulary=r'dexjoco_|\bAllegro\b|\bPanda\b|\bFranka\b|\brotvec\b',
        task_nouns=(r'\b(hammer|nails?|mouse|mouse ?pad|bucket|boxed food|tongs|glasses|temples?|plant|watering|'
                    r'spray|ipad|password|hanoi|disks?|pegs?|tray|socket|microwave|hot ?dog|photograph|shutter|'
                    r'logo)\b'),
        task_tag='DexJoCo', score_label='Formal success rate (official evaluation layouts)',
        layout_note='DexJoCo: exploration draws fresh seeded layouts; formal plays the 150 official evaluation episodes '
                    '(seeds 0-2, 50 each).',
        setup_defaults={'formal_episodes': 150, 'formal_workers': 4, 'formal_eval_seed': 1, 'exploration_start_seed': 0,
                        'demonstration_context': 'terminal_state', 'expert_demonstrations': 0},
        cli_defaults={'exploration_start_seed': 0, 'formal_episodes': 150, 'formal_workers': 4},
        fixed={
            'formal_eval_seed': (1, 'Formal always uses the official DexJoCo evaluation layouts.'),
            'expert_content': ('full', 'DexJoCo expert episodes are complete recordings.'),
        },
    ),
}
ALL_ROBOT_TOOLS = frozenset().union(*(b.robot_tools for b in BENCHMARKS.values()))
ALL_MUTATING = frozenset().union(*(b.mutating for b in BENCHMARKS.values()))
ALL_UNSCOPED_TOOLS = frozenset().union(*(b.unscoped_tools for b in BENCHMARKS.values()))
ALL_TOOL_PREFIXES = tuple(p for b in BENCHMARKS.values() for p in b.tool_prefixes)
EPISODE_TOOL_PREFIXES = tuple(p for b in BENCHMARKS.values() for p in b.episode_tool_prefixes)
SIMULATOR_MODULES = tuple(m for b in BENCHMARKS.values() for m in b.simulator_modules)
EPISODE_SERVERS = tuple(b.simulator_modules[0] for b in BENCHMARKS.values())
STATUS_TOOLS = frozenset(b.status_tool for b in BENCHMARKS.values())
OBSERVE_TOOLS = frozenset(b.observe_tool for b in BENCHMARKS.values())


def benchmark(name=DEFAULT):
    try:
        return BENCHMARKS[name]
    except KeyError:
        raise ValueError(f"Unknown benchmark {name!r}; choose one of {', '.join(sorted(BENCHMARKS))}") from None


def for_config(config):
    return benchmark(getattr(config, 'benchmark', DEFAULT))


def for_task(task):
    """The benchmark whose catalog has the task; task names are unique across benchmarks."""
    owner = next((b for b in BENCHMARKS.values() if task in b.task_names()), None)
    if owner is None:
        raise ValueError(f'Unknown task {task!r}')
    return owner


def tool_benchmark(name):
    """The benchmark that owns a robot tool name, or None."""
    return next((b for b in BENCHMARKS.values() if name in b.robot_tools), None)


def planning_only(name, arguments):
    owner = tool_benchmark(name)
    return bool(owner and owner.planning_only(name, arguments))
