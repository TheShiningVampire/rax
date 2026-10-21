"""Experiment (exp/add-state-info/object): the privileged object-state tool.

No simulator: object_states reads a fake MuJoCo data object, and the backend routes the tool to
the fake simulator process of test_dexjoco_backend.
"""
import json
from types import SimpleNamespace

import pytest

from services.benchmarks import BENCHMARKS
from services.dexjoco import objects as O
from services.dexjoco import tools as T
from services.mcp_contract import Contract
from services.simulator.io import RejectedBeforeAction

from local_tests.unit.test_dexjoco_backend import PRIVATE_WORDS, backend_for


class FakeData:
    """Answers every MuJoCo name lookup with a fixed value."""

    def joint(self, name):
        return SimpleNamespace(qpos=[0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0])

    def body(self, name):
        return SimpleNamespace(xpos=[1.0, 2.0, 3.0], xquat=[1.0, 0.0, 0.0, 0.0])

    def site(self, name):
        return SimpleNamespace(xpos=[0.5, 0.5, 0.5])

    def geom(self, name):
        return SimpleNamespace(xpos=[0.4, 0.4, 0.4], xmat=[0, -1, 0, 1, 0, 0, 0, 0, 1])

    def sensor(self, name):
        return SimpleNamespace(data=[0.7, 0.8, 0.9])


def test_the_tool_is_registered_and_takes_no_arguments():
    assert T.OBJECT_STATES in BENCHMARKS['dexjoco'].robot_tools
    assert T.OBJECT_STATES in Contract(benchmark='dexjoco').robot_names
    assert T.OBJECT_STATES in Contract(observation_profile='rgb-only', benchmark='dexjoco').robot_names
    T.validate_request(T.OBJECT_STATES, {})
    with pytest.raises(RejectedBeforeAction):
        T.validate_request(T.OBJECT_STATES, {'goal': True})
    assert BENCHMARKS['dexjoco'].robot['objects'] == T.OBJECT_STATES


@pytest.mark.parametrize('env_name', ['water_plant', 'fold_glasses', 'bimanual_microwave_cook',
                                      'bimanual_photograph', 'bimanual_hanoi'])
def test_object_states_report_every_label_of_the_trial_tasks(env_name):
    dual = env_name.startswith('bimanual_')
    state = O.object_states(FakeData(), env_name, dual)
    assert state['frame'] == 'world' and state['privileged_level'] == 'objects'
    # The objects level is the original reply: no full-level key.
    assert set(state) == {'privileged_level', 'frame', 'objects', 'arm_joint_positions_rad'}
    assert not any(key in entry for entry in state['objects'].values()
                   for key in ('linear_velocity_mps', 'angular_velocity_radps', 'size'))
    assert set(state['objects']) == set(O.state_sources(env_name)) and state['objects']
    assert not any('error' in entry for entry in state['objects'].values())
    assert set(state['arm_joint_positions_rad']) == ({'right', 'left'} if dual else {'arm'})
    assert all(len(v) == 7 for v in state['arm_joint_positions_rad'].values())


def test_hanoi_reports_the_disks_and_pegs():
    state = O.object_states(FakeData(), 'bimanual_hanoi', True)['objects']
    assert {'large disk', 'medium disk', 'small disk', 'peg A', 'peg B', 'peg C', 'tower base'} <= set(state)
    assert state['small disk'] == {'position_m': [0.1, 0.2, 0.3], 'quaternion_wxyz': [1.0, 0.0, 0.0, 0.0]}
    assert state['peg A'] == {'position_m': [0.5, 0.5, 0.5]}


def test_entries_are_typed_by_source():
    microwave = O.object_states(FakeData(), 'bimanual_microwave_cook', True)['objects']
    assert microwave['door hinge'] == {'angle_rad': 0.1}
    assert microwave['food centre'] == {'position_m': [0.7, 0.8, 0.9]}
    bucket = O.object_states(FakeData(), 'pick_bucket', False)['objects']
    assert len(bucket['bucket corners']['positions_m']) == O.BUCKET_CORNERS


def test_trial_task_extras():
    assert O.object_states(FakeData(), 'water_plant', False)['objects']['trigger'] == {'angle_rad': 0.1}
    glasses = O.object_states(FakeData(), 'fold_glasses', False)['objects']
    assert glasses['first temple hinge'] == glasses['second temple hinge'] == {'angle_rad': 0.1}
    logo = O.object_states(FakeData(), 'bimanual_photograph', True)['objects']['logo']
    assert logo['position_m'] == [0.4, 0.4, 0.4]
    # 90 degrees about +z.
    assert logo['quaternion_wxyz'] == pytest.approx([2 ** -0.5, 0, 0, 2 ** -0.5])


@pytest.mark.parametrize('m', [[1, 0, 0, 0, 1, 0, 0, 0, 1], [1, 0, 0, 0, -1, 0, 0, 0, -1],
                               [-1, 0, 0, 0, -1, 0, 0, 0, 1], [0, 0, 1, 0, 1, 0, -1, 0, 0]])
def test_matrix_to_quaternion(m):
    from scipy.spatial.transform import Rotation
    x, y, z, w = Rotation.from_matrix([m[0:3], m[3:6], m[6:9]]).as_quat()
    expected = [w, x, y, z] if w >= 0 else [-w, -x, -y, -z]
    assert O._quaternion_wxyz(m) == pytest.approx(expected, abs=1e-9)


def test_unreadable_names_are_reported_not_raised():
    class Broken(FakeData):
        def site(self, name):
            raise KeyError(name)

    state = O.object_states(Broken(), 'water_plant', False)['objects']
    assert state['nozzle'] == {'error': "KeyError: 'ref_point'"}


def test_no_success_information_in_the_sources():
    for env_name in O.TAIL:
        for label in O.state_sources(env_name):
            assert 'success' not in label and 'goal' not in label and 'target' not in label


def test_backend_routes_the_tool_to_the_simulator(tmp_path):
    backend = backend_for(tmp_path, task='water_plant')
    backend.start(formal=False, seed=0)
    reply = {'episode_id': backend.sim.episode_id, 'step_id': backend.sim.step_id,
             **O.object_states(FakeData(), 'water_plant', False)}
    backend.sim.request = lambda op, **kwargs: reply if op == 'object_states' else pytest.fail(op)
    value, images = backend.call(T.OBJECT_STATES, {})
    assert images == [] and value['objects']['nozzle'] == {'position_m': [0.5, 0.5, 0.5]}
    # privileged_level names the tool's level; no other private word may appear.
    assert value['privileged_level'] == 'objects'
    text = json.dumps({k: v for k, v in value.items() if k != 'privileged_level'})
    assert not any(word in text for word in PRIVATE_WORDS)


# ------------------------------------------------------------------ full level
import math

import numpy as np

from services.mcp_reply import _agent_safe


def rz(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


class Contacts:
    def __init__(self, rows):
        self.geom1 = [r[0] for r in rows]
        self.geom2 = [r[1] for r in rows]
        self.pos = [np.asarray(r[2], dtype=np.float64) for r in rows]
        self.dist = [r[3] if len(r) > 3 else -0.001 for r in rows]


class Scene:
    """A water_plant-like model: world, table, a single-arm robot with its hand, the watering can
    (free joint spray_root) with its trigger (hinge joint_0), the plant and an unnamed body.

    Geoms: 0 table, 1 arm link, 2 palm, 3 fingertip, 4 can (collision), 5 can (visual only),
    6 trigger, 7 plant, 8 unnamed body."""
    BODIES = ['world', 'table', 'base', 'link7', 'allegro_palm', 'ff_tip', 'link_2', 'link_0', 'plant', '']
    PARENT = [0, 0, 0, 2, 3, 4, 0, 6, 0, 0]
    GEOM_BODY = [1, 3, 4, 5, 6, 6, 7, 8, 9]
    JOINTS = {'spray_root': (6, 0), 'joint_0': (7, 6)}

    def __init__(self):
        n = len(self.BODIES)
        self.nbody, self.ngeom = n, len(self.GEOM_BODY)
        self.body_parentid = np.array(self.PARENT)
        self.geom_bodyid = np.array(self.GEOM_BODY)
        self.geom_contype = np.array([1, 1, 1, 1, 1, 0, 1, 1, 1])
        self.geom_conaffinity = np.array([1, 1, 1, 1, 0, 0, 1, 1, 1])
        self.geom_aabb = np.zeros((self.ngeom, 6))
        self.geom_aabb[:, 3:] = 0.01
        self.geom_aabb[4] = [0, 0, 0, 0.1, 0.2, 0.3]
        self.geom_aabb[5] = [0, 0, 0, 5, 5, 5]  # visual: ignored
        names = list(self.JOINTS)
        self.jnt_bodyid = np.array([self.JOINTS[j][0] for j in names])
        self.jnt_dofadr = np.array([self.JOINTS[j][1] for j in names])
        self._joint_ids = {j: i for i, j in enumerate(names)}

    def body(self, key):
        index = key if isinstance(key, int) else self.BODIES.index(key)
        return SimpleNamespace(id=index, name=self.BODIES[index])

    def joint(self, name):
        return SimpleNamespace(id=self._joint_ids[name])


class SceneData(FakeData):
    def __init__(self, model, contacts=()):
        n = model.nbody
        self.xpos = np.zeros((n, 3))
        self.xmat = np.tile(np.eye(3).reshape(9), (n, 1))
        # The can at (1, 0, 0) turned 90 degrees about z; its trigger 0.5 m above its origin.
        self.xpos[6] = [1, 0, 0]
        self.xmat[6] = rz(90).reshape(9)
        self.xpos[7] = [1, 0, 0.5]
        self.xmat[7] = rz(90).reshape(9)
        self.geom_xpos = np.zeros((model.ngeom, 3))
        self.geom_xmat = np.tile(np.eye(3).reshape(9), (model.ngeom, 1))
        self.geom_xpos[4] = self.xpos[6] + rz(90) @ [0.05, 0, 0]
        self.geom_xpos[5] = self.xpos[6]
        self.geom_xmat[4] = self.geom_xmat[5] = rz(90).reshape(9)
        self.geom_xpos[6] = self.xpos[7]
        self.geom_xmat[6] = rz(90).reshape(9)
        self.qvel = np.array([1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.5])
        self.contact = Contacts(contacts)
        self.ncon = len(contacts)


class FakeMujoco:
    """mj_contactForce gives contact i the normal force forces[i]; mj_objectVelocity a fixed
    [angular, linear]."""
    mjtObj = SimpleNamespace(mjOBJ_BODY=1)

    def __init__(self, forces=()):
        self.forces = list(forces)
        self.velocity_calls = []

    def mj_contactForce(self, model, data, i, out):
        out[:] = [self.forces[i], 0.1, 0.2, 0, 0, 0]

    def mj_objectVelocity(self, model, data, objtype, objid, out, flg_local):
        self.velocity_calls.append((objtype, objid, flg_local))
        out[:] = [0.1, 0.2, 0.3, 1.0, 2.0, 3.0]


def full_state(contacts=(), forces=(), env=None, env_name='water_plant'):
    model = Scene()
    data = SceneData(model, contacts)
    scene = O.scene_index(model, data, env_name)
    mj = FakeMujoco(forces)
    state = O.object_states(data, env_name, False, level='full', model=model, env=env, scene=scene, mj=mj)
    return state, scene, mj


@pytest.mark.parametrize('value, level', [(None, 'full'), ('full', 'full'), ('objects', 'objects')])
def test_privileged_level_reads_the_environment(value, level):
    environ = {} if value is None else {'RAX_PRIVILEGED_STATE': value}
    assert O.privileged_level(environ) == level


@pytest.mark.parametrize('value', ['', 'Full', 'none', 'goal'])
def test_an_invalid_privileged_level_is_an_error(value, monkeypatch):
    with pytest.raises(ValueError, match='RAX_PRIVILEGED_STATE'):
        O.privileged_level({'RAX_PRIVILEGED_STATE': value})
    monkeypatch.setenv('RAX_PRIVILEGED_STATE', value)
    with pytest.raises(ValueError):
        O.privileged_level()


def test_scene_index_labels_robot_hand_and_objects():
    model = Scene()
    scene = O.scene_index(model, SceneData(model), 'water_plant')
    assert scene['objects'] == {'watering can': 6, 'plant': 8}
    # The trigger hinge's body is its own label (nearest labelled ancestor); the rest is unlabelled.
    assert scene['owner'] == {1: None, 2: None, 3: None, 4: None, 5: None, 6: 'watering can', 7: 'trigger',
                              8: 'plant', 9: None}
    assert scene['robot'] == {2, 3, 4, 5} and scene['hand'] == {4, 5}
    assert scene['names'][9] == 'body9'


def test_size_is_the_collision_box_in_the_object_frame():
    model = Scene()
    sizes = O.scene_index(model, SceneData(model), 'water_plant')['sizes']
    can = sizes['watering can']
    # The can's box (offset 0.05 m along its own x) plus the trigger 0.5 m above; the visual geom is ignored.
    assert can['half_extents_m'] == pytest.approx([0.1, 0.2, 0.405])
    assert can['center_offset_m'] == pytest.approx([0.05, 0.0, 0.105])
    assert can['collision_geoms'] == 2 and can['frame'] == 'object'
    assert sizes['plant']['half_extents_m'] == pytest.approx([0.01, 0.01, 0.01])


def test_full_level_adds_velocities_and_sizes():
    state, scene, mj = full_state()
    assert state['privileged_level'] == 'full'
    can = state['objects']['watering can']
    # Free joint: linear from qvel in world, angular rotated from the body frame (90 deg about z).
    assert can['linear_velocity_mps'] == pytest.approx([1.0, 2.0, 3.0])
    assert can['angular_velocity_radps'] == pytest.approx([0.0, 1.0, 0.0], abs=1e-12)
    assert can['size'] == scene['sizes']['watering can']
    plant = state['objects']['plant']
    # Body: mj_objectVelocity in the world frame, [angular, linear].
    assert plant['linear_velocity_mps'] == [1.0, 2.0, 3.0] and plant['angular_velocity_radps'] == [0.1, 0.2, 0.3]
    assert mj.velocity_calls == [(1, 8, 0)]
    # Points and hinges get no velocity or size.
    assert set(state['objects']['nozzle']) == {'position_m'}
    assert set(state['objects']['trigger']) == {'angle_rad'}


def test_contacts_are_summed_per_pair_and_filtered():
    contacts = [
        (4, 0, [1, 0, 0]), (0, 4, [3, 0, 0]),  # can-table twice
        (3, 4, [0, 1, 0]),  # fingertip-can
        (1, 2, [0, 0, 1]),  # arm-palm: robot self contact, dropped
        (0, 8, [0, 0, 2]),  # table-unnamed body: no object or hand, dropped
        (2, 0, [0, 2, 0]),  # palm-table: a hand contact, kept
        (4, 6, [0, 0, 3]),  # can-trigger: two labels, kept
        (4, 5, [0, 0, 4]),  # can-can (inside one object), dropped
    ]
    forces = [2.0, 3.0, 7.0, 100.0, 100.0, 1.0, 0.5, 100.0]
    state, _, _ = full_state(contacts, forces)
    rows = state['contacts']
    assert [(r['a'], r['b']) for r in rows] == [('watering can', 'ff_tip'), ('watering can', 'table'),
                                               ('allegro_palm', 'table'), ('trigger', 'watering can')]
    can_table = rows[1]
    assert can_table['normal_force_n'] == pytest.approx(5.0) and can_table['count'] == 2
    assert can_table['position_m'] == pytest.approx([2.0, 0.0, 0.0])
    assert set(can_table) == {'a', 'b', 'normal_force_n', 'count', 'position_m'}


def test_contacts_are_capped_strongest_first():
    model = Scene()
    data = SceneData(model, [(4, 0, [0, 0, 0])] * 3)
    scene = O.scene_index(model, data, 'water_plant')
    # One pair, summed over its three contacts.
    rows = O.contacts(model, data, scene, FakeMujoco([1.0, 2.0, 3.0]), limit=1)
    assert len(rows) == 1 and rows[0]['count'] == 3 and rows[0]['normal_force_n'] == pytest.approx(6.0)
    names = list(scene['names'])
    many = [(0, 4, [0, 0, 0])] * (O.MAX_CONTACT_PAIRS + 6)
    data = SceneData(model, many)
    owner = dict(scene['owner'])
    # 70 distinct partner bodies, each labelled differently, give 70 pairs.
    model.geom_bodyid = np.concatenate([model.geom_bodyid, np.arange(100, 100 + len(many))])
    data.contact.geom2 = list(range(model.ngeom, model.ngeom + len(many)))
    names += [f'x{i}' for i in range(200)]
    owner.update({100 + i: f'object {i}' for i in range(len(many))})
    rows = O.contacts(model, data, {**scene, 'owner': owner, 'names': names},
                      FakeMujoco([float(i) for i in range(len(many))]))
    assert len(rows) == O.MAX_CONTACT_PAIRS
    forces = [r['normal_force_n'] for r in rows]
    assert forces == sorted(forces, reverse=True) and forces[0] == len(many) - 1


def test_full_level_without_a_scene_or_env_reports_errors():
    model = Scene()
    data = SceneData(model)
    state = O.object_states(data, 'water_plant', False, level='full', model=model, env=None,
                            scene={'error': 'RuntimeError: boom'}, mj=FakeMujoco())
    assert state['contacts'] == {'error': 'RuntimeError: RuntimeError: boom'}
    assert state['objects']['plant']['size'] == {'error': 'no size: the scene index is missing'}
    assert state['goal'] == {'available': False, 'error': 'no environment'}
    other = O.object_states(FakeData(), 'pick_bucket', False, level='full', model=model, scene=None,
                            mj=FakeMujoco())
    assert other['goal'] == {'available': False}


def test_the_full_reply_survives_the_agent_filter_unchanged():
    from local_tests.unit.test_dexjoco_goals import WaterEnv
    model = Scene()
    data = SceneData(model, [(4, 0, [1, 0, 0]), (3, 4, [0, 1, 0])])
    env = WaterEnv(nozzle=[1.0, 2.0, 3.1], angle=0.4, pulled=True, counter=3)
    data.site_xpos = env.data.site_xpos  # FakeData's plant body is at (1, 2, 3)
    state = O.object_states(data, 'water_plant', False, level='full', model=model, env=env,
                            scene=O.scene_index(model, data, 'water_plant'), mj=FakeMujoco([1.0, 2.0]))
    assert state['goal']['satisfied_now'] is True and state['goal']['holding']['steps_satisfied'] == 3
    reply = {'episode_id': 'e', 'step_id': 3, **state}
    assert state['goal']['available'] and state['contacts']
    plain = json.loads(json.dumps(reply))
    assert _agent_safe(plain) == plain


def test_the_tool_text_follows_the_level(monkeypatch):
    full, objects = T.object_states_description('full'), T.object_states_description('objects')
    assert objects.endswith('no physics, no success information.')
    for word in ('linear_velocity_mps', 'angular_velocity_radps', 'half_extents_m', 'contacts', 'normal_force_n',
                 'goal', 'conditions', 'failure_conditions', 'holding', 'satisfied_now', 'privileged_level'):
        assert word in full and word not in objects.replace('privileged', '')
    monkeypatch.setenv('RAX_PRIVILEGED_STATE', 'objects')
    tool = next(t for t in T.tool_definitions() if t['name'] == T.OBJECT_STATES)
    assert tool['description'] == objects
    monkeypatch.delenv('RAX_PRIVILEGED_STATE')
    tool = next(t for t in T.tool_definitions() if t['name'] == T.OBJECT_STATES)
    assert tool['description'] == full


@pytest.mark.parametrize('value, level', [(None, 'full'), ('objects', 'objects')])
def test_the_workspace_robot_words_carry_the_level(monkeypatch, value, level):
    from services.dexjoco.benchmark import task_robot
    if value is None:
        monkeypatch.delenv('RAX_PRIVILEGED_STATE', raising=False)
    else:
        monkeypatch.setenv('RAX_PRIVILEGED_STATE', value)
    assert task_robot('water_plant')['privileged'] == level
    assert BENCHMARKS['dexjoco'].robot_facts('water_plant')['privileged'] == level


def test_exploration_workers_keep_the_level(tmp_path, monkeypatch):
    import subprocess
    from services.exploration import pool
    seen = {}

    def popen(*args, **kwargs):
        seen.update(kwargs['env'])
        return SimpleNamespace(poll=lambda: 0)

    monkeypatch.setattr(subprocess, 'Popen', popen)
    monkeypatch.setenv('RAX_PRIVILEGED_STATE', 'objects')
    monkeypatch.setenv('RAX_SOMETHING_ELSE', '1')
    worker = pool.Worker({'root': str(tmp_path / 'worker'), 'episode_seconds': None})
    try:
        assert seen['RAX_PRIVILEGED_STATE'] == 'objects' and 'RAX_SOMETHING_ELSE' not in seen
    finally:
        worker.stream.close()
        worker.socket.close()
        worker.log.close()
