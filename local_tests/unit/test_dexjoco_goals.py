"""Experiment (exp/add-state-info/object): the trial tasks' goal specifications (goals.py).

No simulator: each fake environment carries the attributes and side-effect-free helpers the
official environment has, and this module stands in for the official environment module, so its
module-level constants below play the official ones (goals._official reads them from the env's
module). A last test checks the literals goals.py copies against the official source.
"""
import json
import math
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from services.dexjoco import goals as G
from services.mcp_reply import _agent_safe

# The official module-level constants (panda_water_plant_env.py, panda_bimanual_photograph_env.py).
_TRIGGER_RELEASE_THRESHOLD = 0.25
_TRIGGER_PULL_THRESHOLD = 0.34
_SUCCESS_STEPS_REQUIRED = 30
_TARGET_REGION_RADIUS = 0.16
_TARGET_REGION_HALF_HEIGHT = 0.10
_TARGET_ANGLE_DEG = 10.0
_WORLD_POS_X = np.asarray([1.0, 0.0, 0.0])

OFFICIAL = Path(__file__).resolve().parents[2] / 'DexJoCo/dexjoco/dexjoco/sim/envs'


def ns(**kwargs):
    return SimpleNamespace(**kwargs)


class Named:
    """name -> namespace lookups, e.g. model.site('ref_point')."""

    def __init__(self, **entries):
        self.entries = entries

    def __call__(self, name):
        return self.entries[name]


def assert_shape(goal):
    assert goal['available'] is True and isinstance(goal['description'], str)
    assert goal['conditions'] and isinstance(goal['failure_conditions'], list)
    for row in goal['conditions'] + goal['failure_conditions']:
        assert set(row) == {'name', 'satisfied', 'measured', 'threshold', 'note'}
        assert isinstance(row['satisfied'], bool)
    assert isinstance(goal['satisfied_now'], bool) and goal['targets']
    plain = json.loads(json.dumps(goal))
    assert _agent_safe(plain) == plain


# ------------------------------------------------------------------ water_plant
class WaterEnv:
    def __init__(self, nozzle, plant=(0.0, 0.0, 1.0), angle=0.0, pulled=False, counter=0):
        self._model = ns(site=Named(ref_point=ns(id=0)))
        self.data = ns(site_xpos=np.array([nozzle], dtype=np.float64),
                       body=Named(plant=ns(xpos=np.array(plant, dtype=np.float64))),
                       sensor=Named(spray_joint_0_pos=ns(data=np.array([angle]))))
        self._trigger_pulled, self._success_counter = pulled, counter


def water(**kwargs):
    env = WaterEnv(**kwargs)
    return G.goal(env, env.data, 'water_plant')


def test_water_plant_at_reset_is_not_satisfied():
    goal = water(nozzle=[0.3, -0.25, 0.95])
    assert_shape(goal)
    assert goal['satisfied_now'] is False
    assert goal['holding'] == {'steps_satisfied': 0, 'steps_required': 30}
    by = {c['name']: c for c in goal['conditions']}
    assert by['nozzle_within_radius_of_plant']['measured'] == pytest.approx(math.hypot(0.3, 0.25))
    assert by['nozzle_within_radius_of_plant']['satisfied'] is False
    assert by['nozzle_within_height_band_of_plant']['measured'] == pytest.approx(-0.05)
    assert by['trigger_pulled']['threshold'] == [0.25, 0.34]
    assert goal['failure_conditions'][0]['satisfied'] is False
    assert goal['targets']['plant']['position_m'] == [0.0, 0.0, 1.0]


def test_water_plant_inside_with_the_trigger_latched():
    goal = water(nozzle=[0.2, 0.0, 1.2], angle=0.3, pulled=True, counter=12)  # both bounds inclusive
    assert goal['satisfied_now'] is True and all(c['satisfied'] for c in goal['conditions'])
    assert goal['holding'] == {'steps_satisfied': 12, 'steps_required': 30}
    assert goal['failure_conditions'][0]['satisfied'] is False
    # The latch, not the angle, decides (0.3 rad is between the thresholds).
    assert water(nozzle=[0.0, 0.0, 1.0], angle=0.3, pulled=False)['satisfied_now'] is False


@pytest.mark.parametrize('nozzle', [[0.21, 0.0, 1.0], [0.0, 0.0, 1.21], [0.0, 0.0, 0.79]])
def test_water_plant_pulling_outside_is_the_failure(nozzle):
    goal = water(nozzle=nozzle, angle=0.4, pulled=True)
    assert goal['satisfied_now'] is False and goal['failure_conditions'][0]['satisfied'] is True


# ------------------------------------------------------------------ fold_glasses
class GlassesEnv:
    def __init__(self, glass, j0=0.2, j1=0.2, case=(0.1, 0.2, 0.97), count=0):
        self._model = ns(body=Named(open_box=ns(id=1)), body_pos=np.array([[0, 0, 0], case], dtype=np.float64))
        self.data = ns(sensor=Named(glass_pos=ns(data=np.asarray(glass, dtype=np.float64)),
                                    glass_joint_0_pos=ns(data=np.array([j0])),
                                    glass_joint_1_pos=ns(data=np.array([j1]))))
        self._success_trigger_count, self._success_trigger_target = count, 50


def glasses(**kwargs):
    env = GlassesEnv(**kwargs)
    return G.goal(env, env.data, 'fold_glasses')


def test_fold_glasses_conditions_and_holding():
    goal = glasses(glass=[0.0, 0.0, 1.0])
    assert_shape(goal)
    assert goal['satisfied_now'] is False and goal['holding'] == {'steps_satisfied': 0, 'steps_required': 50}
    by = {c['name']: c for c in goal['conditions']}
    assert by['glasses_inside_case_x']['measured'] == pytest.approx(-0.1)
    assert by['glasses_inside_case_y']['measured'] == pytest.approx(-0.2) and not by['glasses_inside_case_y']['satisfied']
    assert by['glasses_inside_case_z']['threshold'] == pytest.approx([-0.045, 0.0275])
    done = glasses(glass=[0.1 + 0.14, 0.2 - 0.14, 0.97 + 0.027], j0=1.2, j1=1.11, count=7)
    assert done['satisfied_now'] is True and done['holding']['steps_satisfied'] == 7


@pytest.mark.parametrize('kwargs', [dict(j0=1.1, j1=1.2), dict(j0=1.2, j1=1.1)])
def test_fold_glasses_needs_both_temples_strictly_past(kwargs):
    assert glasses(glass=[0.1, 0.2, 0.97], **kwargs)['satisfied_now'] is False


def test_fold_glasses_top_bound_is_lowered():
    assert glasses(glass=[0.1, 0.2, 0.97 + 0.03], j0=1.2, j1=1.2)['satisfied_now'] is False
    assert glasses(glass=[0.1, 0.2, 0.97 - 0.044], j0=1.2, j1=1.2)['satisfied_now'] is True


# ------------------------------------------------------- bimanual_microwave_cook
def rz(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


class MicrowaveEnv:
    """Interior sites as in robocasa_microwave.xml; the microwave turned 90 deg about z at (1, 2, 1)."""
    SITES = {'int_p0': [-0.258, -0.162, -0.12], 'int_px': [0.156, -0.162, -0.12],
             'int_py': [-0.258, 0.222, -0.12], 'int_pz': [-0.258, -0.162, 0.144]}

    def __init__(self, food_local, door=0.0, contacts=()):
        origin, rotation = np.array([1.0, 2.0, 1.0]), rz(90)
        self._model = ns(site=Named(**{k: ns(pos=np.array(v)) for k, v in self.SITES.items()}),
                         geom=Named(start_button=ns(id=5, size=np.array([0.025, 0.0306, 0.025]))),
                         geom_bodyid=np.array([0, 1, 2, 3, 4, 5, 6]),
                         body=lambda i: ns(name=['world', 'palm_right', 'hot_dog', 'x', 'y', 'microwave_object',
                                                 'z'][i]))
        self.data = ns(joint=Named(microjoint=ns(qpos=np.array([door]))),
                       sensor=Named(hot_dog_pos=ns(data=origin + rotation @ np.asarray(food_local, dtype=np.float64))),
                       body=Named(microwave_object=ns(xpos=origin, xmat=rotation.reshape(9))),
                       geom=Named(start_button=ns(xpos=np.array([0.0, 0.0, 1.0])),
                                  door_handle=ns(xpos=np.array([0.5, 0.0, 1.0]))),
                       ncon=len(contacts), contact=ns(geom1=[c[0] for c in contacts], geom2=[c[1] for c in contacts],
                                                      dist=[c[2] for c in contacts]))

    # The official side-effect-free helpers, as in panda_bimanual_microwave_cook_env.py.
    def hot_dog_inside_microwave(self, margin=0.0):
        body = self.data.body('microwave_object')
        local = np.asarray(body.xmat).reshape(3, 3).T @ (self.data.sensor('hot_dog_pos').data - body.xpos)
        p0, px, py, pz = (np.array(self.SITES[k]) for k in ('int_p0', 'int_px', 'int_py', 'int_pz'))
        return bool((p0[0] - margin <= local[0] <= px[0] + margin) and (p0[1] - margin <= local[1] <= py[1] + margin)
                    and (p0[2] - margin <= local[2] <= pz[2] + margin))

    def _geom_in_contact(self, name, dist_threshold=0.0):
        assert name == 'start_button'
        c = self.data.contact
        return any(5 in (c.geom1[i], c.geom2[i]) and c.dist[i] <= dist_threshold for i in range(self.data.ncon))


def microwave(**kwargs):
    env = MicrowaveEnv(**kwargs)
    return G.goal(env, env.data, 'bimanual_microwave_cook')


def test_microwave_cook_conditions():
    goal = microwave(food_local=[0.0, 0.0, 0.0], door=-0.5, contacts=[(1, 5, 0.001)])
    assert_shape(goal)
    assert 'holding' not in goal and goal['failure_conditions'] == []
    by = {c['name']: c for c in goal['conditions']}
    assert by['door_closed']['satisfied'] is False and by['door_closed']['measured'] == -0.5
    assert by['food_inside_cavity']['satisfied'] is True
    assert by['food_inside_cavity']['measured'] == pytest.approx([0, 0, 0], abs=1e-12)
    assert np.allclose(by['food_inside_cavity']['threshold'], [[-0.268, 0.166], [-0.172, 0.232], [-0.13, 0.154]])
    # A contact within the margin (dist > 0) is not a press.
    assert by['start_button_touched']['satisfied'] is False and by['start_button_touched']['measured'] == 0
    cavity = goal['targets']['microwave cavity']
    centre_local = np.array([(-0.268 + 0.166) / 2, (-0.172 + 0.232) / 2, (-0.13 + 0.154) / 2])
    assert cavity['position_m'] == pytest.approx(list(np.array([1.0, 2.0, 1.0]) + rz(90) @ centre_local))
    assert cavity['half_extents_m'] == pytest.approx([0.217, 0.202, 0.142])


def test_microwave_cook_done_at_one_step():
    goal = microwave(food_local=[0.16, 0.2, 0.15], door=0.005, contacts=[(1, 5, -0.001), (5, 6, 0.0)])
    assert goal['satisfied_now'] is True
    button = next(c for c in goal['conditions'] if c['name'] == 'start_button_touched')
    assert button['measured'] == 2 and 'palm_right' in button['note']
    assert microwave(food_local=[0.17, 0.0, 0.0], contacts=[(1, 5, -0.001)])['satisfied_now'] is False
    assert microwave(food_local=[0.0, 0.0, 0.0], door=0.01, contacts=[(1, 5, -0.001)])['satisfied_now'] is False


# ------------------------------------------------------------ bimanual_photograph
class PhotographEnv:
    def __init__(self, metrics, complete=False):
        self._target_region_geom_id, self._view_site_id, self._shutter_geom_id = 0, 0, 1
        self.metrics, self.complete, self.calls = metrics, complete, 0
        self.data = ns(geom_xpos=np.array([[0.1, 0.2, 1.3], [0.0, 0.0, 1.0]]),
                       geom_xmat=np.array([rz(0).reshape(9), rz(0).reshape(9)]),
                       site_xpos=np.array([[0.0, 0.1, 1.2]]))

    def _compute_success_metrics(self):
        self.calls += 1
        return self.complete, self.metrics


METRICS = {'region_pass': True, 'angle_pass': False, 'shutter_pressed': False, 'radial_dist': 0.05,
           'region_height_dist': 0.02, 'view_angle_deg': 25.0, 'shutter_contact_count': 0}


def test_photograph_maps_the_official_metrics():
    env = PhotographEnv(METRICS)
    goal = G.goal(env, env.data, 'bimanual_photograph')
    assert_shape(goal)
    assert env.calls == 1 and goal['satisfied_now'] is False and 'holding' not in goal
    by = {c['name']: c for c in goal['conditions']}
    assert by['lens_in_target_region']['satisfied'] is True
    assert by['lens_in_target_region']['measured'] == [0.05, 0.02]
    assert by['lens_in_target_region']['threshold'] == [0.16, 0.1]
    assert by['camera_aimed_along_x']['measured'] == 25.0 and by['camera_aimed_along_x']['threshold'] == 10.0
    assert goal['targets']['target region']['axis_world'] == [0.0, 0.0, 1.0]
    assert goal['targets']['aim direction'] == {'direction_world': [1.0, 0.0, 0.0], 'max_angle_deg': 10.0}
    done = PhotographEnv({**METRICS, 'angle_pass': True, 'shutter_pressed': True, 'shutter_contact_count': 2}, True)
    assert G.goal(done, done.data, 'bimanual_photograph')['satisfied_now'] is True


# ------------------------------------------------------------------ bimanual_hanoi
DISKS = ['hanoi_disk_large', 'hanoi_disk_medium', 'hanoi_disk_small']
HALF = {'hanoi_disk_large': 0.025, 'hanoi_disk_medium': 0.02, 'hanoi_disk_small': 0.015}
POSTS = {'A': np.array([0.0, 0.2, 1.1]), 'B': np.array([0.0, 0.0, 1.1]), 'C': np.array([0.0, -0.2, 1.1])}
BASE_TOP = 1.0


class HanoiEnv:
    """The official helpers' logic (panda_bimanual_hanoi_env.py) over fixed disk positions."""

    def __init__(self, positions, qvel=None, success_count=0, illegal_count=0, illegal_reason=''):
        self._disk_names = list(DISKS)
        self._disk_size_rank = {'hanoi_disk_large': 3, 'hanoi_disk_medium': 2, 'hanoi_disk_small': 1}
        self._disk_half_height = dict(HALF)
        self._disk_qvel_adr = {d: 6 * i for i, d in enumerate(DISKS)}
        self._post_assignment_tol, self._placement_height_tol, self._static_qvel_tol = 0.040, 0.012, 0.35
        self._success_trigger_count, self._success_trigger_target = success_count, 20
        self._illegal_trigger_count, self._illegal_trigger_target = illegal_count, 10
        self._illegal_trigger_reason = illegal_reason
        self.positions = {d: np.asarray(p, dtype=np.float64) for d, p in positions.items()}
        self.data = ns(qvel=np.zeros(18) if qvel is None else np.asarray(qvel, dtype=np.float64))

    def _disk_is_static(self, disk):
        adr = self._disk_qvel_adr[disk]
        return float(np.linalg.norm(self.data.qvel[adr:adr + 6])) <= self._static_qvel_tol

    def _reconstruct_tower_state(self):
        assignment, towers = {}, {'A': [], 'B': [], 'C': []}
        for disk in self._disk_names:
            pos = self.positions[disk]
            post = min(POSTS, key=lambda k: np.linalg.norm(pos[:2] - POSTS[k][:2]))
            if np.linalg.norm(pos[:2] - POSTS[post][:2]) <= self._post_assignment_tol:
                assignment[disk] = post
                towers[post].append(disk)
            else:
                assignment[disk] = 'table' if pos[2] <= BASE_TOP + 0.06 else 'floating'
        for post in towers:
            towers[post].sort(key=lambda d: self.positions[d][2])
        return {'post_positions': POSTS, 'base_top_z': BASE_TOP, 'disk_assignment': assignment,
                'disk_positions': self.positions, 'towers': towers,
                'disk_static': {d: self._disk_is_static(d) for d in self._disk_names}}

    def _tower_geometry_errors(self, disks, post, positions, posts, base_top):
        height, xy_max, z_max = base_top, 0.0, 0.0
        for disk in disks:
            expected = height + self._disk_half_height[disk]
            height += 2 * self._disk_half_height[disk]
            xy_max = max(xy_max, float(np.linalg.norm(positions[disk][:2] - posts[post][:2])))
            z_max = max(z_max, abs(float(positions[disk][2]) - expected))
        return xy_max, z_max


def stacked(post, order, dz=0.0):
    height, out = BASE_TOP, {}
    for disk in order:
        out[disk] = [POSTS[post][0], POSTS[post][1], height + HALF[disk] + dz]
        height += 2 * HALF[disk]
    return out


def hanoi(env):
    return G.goal(env, env.data, 'bimanual_hanoi')


def test_hanoi_reset_layout_is_not_satisfied():
    env = HanoiEnv({**stacked('A', ['hanoi_disk_small']), **stacked('B', ['hanoi_disk_medium']),
                    **stacked('C', ['hanoi_disk_large'])})
    goal = hanoi(env)
    assert_shape(goal)
    assert goal['satisfied_now'] is False
    assert goal['tower_state'] == {'A': ['small disk'], 'B': ['medium disk'], 'C': ['large disk']}
    assert goal['holding'] == {'steps_satisfied': 0, 'steps_required': 20}
    assert not any(c['satisfied'] for c in goal['failure_conditions'])
    targets = goal['targets']
    assert targets['large disk']['position_m'] == pytest.approx([0.0, -0.2, 1.025])
    assert targets['medium disk']['position_m'] == pytest.approx([0.0, -0.2, 1.07])
    assert targets['small disk']['position_m'] == pytest.approx([0.0, -0.2, 1.105])


def test_hanoi_solved_stack():
    env = HanoiEnv(stacked('C', DISKS, dz=0.01), success_count=5)
    goal = hanoi(env)
    assert goal['satisfied_now'] is True and all(c['satisfied'] for c in goal['conditions'])
    assert goal['holding']['steps_satisfied'] == 5
    height = next(c for c in goal['conditions'] if c['name'] == 'stack_height_on_peg_C')
    assert height['measured'] == pytest.approx(0.01)
    # Moving disks, too high a stack or a pending illegal state block it.
    moving = HanoiEnv(stacked('C', DISKS), qvel=[0.4] + [0.0] * 17)
    assert hanoi(moving)['satisfied_now'] is False
    assert hanoi(HanoiEnv(stacked('C', DISKS, dz=0.013)))['satisfied_now'] is False
    assert hanoi(HanoiEnv(stacked('C', DISKS), illegal_count=10, illegal_reason='disk_on_table'))['satisfied_now'] is False


def test_hanoi_illegal_stack_and_disk_on_table():
    env = HanoiEnv({**stacked('B', ['hanoi_disk_small', 'hanoi_disk_large']), **stacked('C', ['hanoi_disk_medium'])},
                   illegal_count=4, illegal_reason='illegal_stack_B')
    goal = hanoi(env)
    by = {c['name']: c for c in goal['failure_conditions']}
    assert by['illegal_stack']['satisfied'] is True and by['illegal_stack']['measured'] == 4
    assert by['illegal_stack']['threshold'] == 10 and 'illegal_stack_B' in by['illegal_stack']['note']
    assert by['disk_on_table']['satisfied'] is False and by['disk_on_table']['measured'] == 0
    assert goal['failure_holding'] == {'reason': 'illegal_stack_B', 'steps_satisfied': 4, 'steps_required': 10}
    table = HanoiEnv({**stacked('C', ['hanoi_disk_large', 'hanoi_disk_medium']),
                      'hanoi_disk_small': [0.3, 0.3, BASE_TOP + 0.02]}, illegal_count=2, illegal_reason='disk_on_table')
    by = {c['name']: c for c in hanoi(table)['failure_conditions']}
    assert by['disk_on_table']['satisfied'] is True and by['disk_on_table']['measured'] == 2
    # Held up in a hand (floating), no failure.
    held = HanoiEnv({**stacked('C', ['hanoi_disk_large', 'hanoi_disk_medium']),
                     'hanoi_disk_small': [0.3, 0.3, BASE_TOP + 0.2]})
    goal = hanoi(held)
    assert not any(c['satisfied'] for c in goal['failure_conditions'])
    assert goal['disk_assignment']['small disk'] == 'floating'


# ------------------------------------------------------------------- dispatch
def test_other_tasks_and_errors():
    assert G.goal(object(), None, 'pick_bucket') == {'available': False}
    assert G.goal(None, None, 'water_plant') == {'available': False, 'error': 'no environment'}
    broken = G.goal(ns(), ns(), 'water_plant')
    assert broken['available'] is False and broken['error'].startswith('AttributeError')
    assert set(G.GOALS) == {'water_plant', 'fold_glasses', 'bimanual_microwave_cook', 'bimanual_photograph',
                            'bimanual_hanoi'}


def test_official_constants_come_from_the_env_module():
    assert G._official(WaterEnv(nozzle=[0, 0, 0]), '_SUCCESS_STEPS_REQUIRED') == 30
    with pytest.raises(AttributeError):
        G._official(WaterEnv(nozzle=[0, 0, 0]), '_NOT_A_CONSTANT')


@pytest.mark.skipif(not OFFICIAL.is_dir(), reason='DexJoCo source not checked out')
def test_copied_literals_match_the_official_source():
    def source(name):
        return (OFFICIAL / name).read_text()

    water = source('panda_water_plant_env.py')
    assert re.search(rf'\bR = {G.WATER_RADIUS_M}\b', water) and re.search(rf'HALF_H = {G.WATER_HALF_HEIGHT_M}\b', water)
    assert '(dx * dx + dy * dy <= R * R) and (-HALF_H <= dz <= HALF_H)' in water
    for name, value in (('_TRIGGER_RELEASE_THRESHOLD', 0.25), ('_TRIGGER_PULL_THRESHOLD', 0.34),
                        ('_SUCCESS_STEPS_REQUIRED', 30)):
        assert re.search(rf'^{name} = {value}$', water, re.M)  # this module's stand-ins agree
    glasses = source('panda_fold_glasses_env.py')
    assert f'(j0 > {G.TEMPLE_FOLDED_RAD})' in glasses and f'(j1 > {G.TEMPLE_FOLDED_RAD})' in glasses
    assert re.search(rf'x_half = {G.CASE_HALF_XY_M}\b', glasses) and re.search(rf'y_half = {G.CASE_HALF_XY_M}\b', glasses)
    assert re.search(rf'z_min = {G.CASE_Z_MIN_M}\b', glasses) and re.search(rf'z_max = +{G.CASE_Z_MAX_M}\b', glasses)
    assert f'glass_local[2] <= z_max - {G.CASE_Z_TOP_ALLOWANCE_M})' in glasses
    assert 'glass_local = glass_pos_world - box_pos_world' in glasses
    microwave = source('panda_bimanual_microwave_cook_env.py')
    assert 'abs(micro_qpos) < 1e-2' in microwave and G.DOOR_CLOSED_RAD == 1e-2
    assert f'hot_dog_inside_microwave(margin={G.CAVITY_MARGIN_M})' in microwave
    assert 'self._geom_in_contact("start_button")' in microwave
    photo = source('panda_bimanual_photograph_env.py')
    for name, value in (('_TARGET_REGION_RADIUS', 0.16), ('_TARGET_REGION_HALF_HEIGHT', 0.10),
                        ('_TARGET_ANGLE_DEG', 10.0)):
        match = re.search(rf'^{name} = ([0-9.]+)$', photo, re.M)
        assert match and float(match.group(1)) == value, name
    hanoi_src = source('panda_bimanual_hanoi_env.py')
    assert f'target_stack = {list(G.HANOI_TARGET_STACK)!r}'.replace("'", '"') in hanoi_src
    assert f'base_top_z + {G.HANOI_TABLE_MARGIN_M}' in hanoi_src
