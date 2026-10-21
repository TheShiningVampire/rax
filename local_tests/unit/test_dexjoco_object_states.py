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
    assert state['frame'] == 'world'
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
    text = json.dumps(value)
    assert not any(word in text for word in PRIVATE_WORDS)
