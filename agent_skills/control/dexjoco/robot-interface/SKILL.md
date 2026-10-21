---
name: robot-interface
description: The DexJoCo robot - Franka Panda arm{% if robot.arms == 'dual' %}s{% endif %} with 16-DoF Allegro hand{% if robot.arms == 'dual' %}s{% endif %} - its official action rows (flange pose as a rotation vector plus hand joint targets), its state vector, frames, cameras and control response. Read before composing robot actions.
---

# Robot interface: Panda + Allegro

{% if robot.arms == 'dual' %}
Two Franka Panda arms stand side by side behind the table, each ending in a
16-joint Allegro hand: the right arm's base at y = -0.4 m, the left at y = +0.4 m,
both at x = -0.8 m. There is no gripper: every grasp is made with the fingers.
{% else %}
One Franka Panda arm stands behind the table (its base at x = -0.8 m, y = 0) and ends
in a 16-joint Allegro hand. There is no gripper: every grasp is made with the fingers.
{% endif %}
`{{ robot.status }}` reports the exact action and state layouts; this skill explains them.
They are the official DexJoCo action and state layouts.

## One action row

Each `dexjoco_step` row is one simulation step: it advances 20 ms of physics, and the
environment paces steps at {{ robot.hz }} per second of wall clock. All values are
**absolute targets**, not deltas.

{% if robot.arms == 'dual' %}
| Slice | Meaning |
|---|---|
| `[0:3]` | right flange position (m, world frame) |
| `[3:6]` | right flange orientation as a rotation vector (axis x angle, rad, world frame) |
| `[6:22]` | right hand joint targets (rad), order `ff0-3, mf0-3, rf0-3, th0-3` |
| `[22:25]` | left flange position (m) |
| `[25:28]` | left flange rotation vector (rad) |
| `[28:44]` | left hand joint targets (rad), order **`rf0-3, mf0-3, ff0-3, th0-3`** (mirrored) |

The left hand's joints come in mirrored order: its first four entries are the ring
finger, the next four the middle and then the index finger. The same order holds in
the state vector.
{% else %}
| Slice | Meaning |
|---|---|
| `[0:3]` | flange position (m, world frame) |
| `[3:6]` | flange orientation as a rotation vector (axis x angle, rad, world frame) |
| `[6:22]` | hand joint targets (rad), order `ff0-3, mf0-3, rf0-3, th0-3` |
{% endif %}

The flange is the arm's tool plate; the hand is mounted on it. The rotation vector
is the axis-angle form of the flange orientation. `dexjoco_rotation_math` converts it
to and from the state's quaternion and a rotation matrix with the official row
conversion (`convert_rotation`), composes and relates poses (`compose_pose`,
`relative_pose`), and needs no episode. Your own code can do the same, for example
`scipy.spatial.transform.Rotation.from_quat(q_wxyz, scalar_first=True).as_rotvec()`.
A rotation vector and its negative-angle twin near pi describe the same orientation;
keep consecutive rows on the same side so the target does not jump
(`near_rotation_vector` picks the twin closest to the previous row's).

**Hold still** with `dexjoco_hold(steps=n)`, or by repeating a row built from the
measured state. A row of zeros is not a hold: it sends the flange to the world origin
and opens every finger to zero.

Each row counts once against the episode step limit (`episode_limits`); the episode
also ends at once on success or on a task failure condition, and later rows of the
same call are not run.

## The measured state

`state` is the official state vector:

{% if robot.arms == 'dual' %}
| Slice | Meaning |
|---|---|
| `[0:3]`, `[3:7]` | right flange position (m) and quaternion `[w, x, y, z]` |
| `[7:10]`, `[10:14]` | left flange position and quaternion `[w, x, y, z]` |
| `[14:30]` | right hand joint positions (rad), `ff, mf, rf, th` |
| `[30:46]` | left hand joint positions (rad), mirrored `rf, mf, ff, th` |
{% else %}
| Slice | Meaning |
|---|---|
| `[0:3]` | flange position (m, world frame) |
| `[3:7]` | flange quaternion `[w, x, y, z]` (world frame) |
| `[7:23]` | hand joint positions (rad), `ff0-3, mf0-3, rf0-3, th0-3` |
{% endif %}

Build every row from the latest measured state: convert the quaternion to a rotation
vector, copy the hand joints, then change only what you intend to move
(`dexjoco_rotation_math` `row_from_state` returns that hold row exactly). The state has no
arm joint angles or object positions; `dexjoco_object_states` reports both.

## Object state

`dexjoco_object_states` returns the simulator's exact state at the current `step_id`, in
the world frame: each task object's and fixture's pose (`position_m`, `quaternion_wxyz`),
named object points (`position_m`), articulation angles (`angle_rad`) and the arm joint
positions (`joint1..7` per arm). It runs no physics. Use it for where objects are and how
they moved after each motion; your program can call it too
(`ctx.call("dexjoco_object_states")`). It reports no success information. The cameras remain available, for example to see contacts
and grasps.

## Start pose and frames

- **World frame:** +x points from the robot{% if robot.arms == 'dual' %}s{% endif %} across the table, +z up. The
  table top is at about z = 0.92-0.97 m (its height changes between layouts).
- At the start {% if robot.arms == 'dual' %}each{% else %}the{% endif %} flange is at x = -0.443 m, z = 1.517 m{% if robot.arms == 'dual' %} (y = -0.4 right, +0.4 left){% endif %}, with the
  quaternion about `[0, -1, 0, 0]` (rotation vector about `[-3.14, 0, 0]`): the flange's
  z axis points down. The hand hangs flat, palm down, fingers pointing along +x and
  reaching about 0.18 m beyond the flange, about 3 cm below it.
{% if robot.arms == 'dual' %}
  On each hand the index finger and the thumb face the middle of the table (+y for
  the right hand, -y for the left) and the ring finger the outside; the thumb sits
  slightly lower and nearer the flange.
{% else %}
  The index finger and the thumb are on the +y side, the ring finger on the -y side;
  the thumb sits slightly lower and nearer the flange.
{% endif %}
  A flange target is not a fingertip target: to put a point on the hand somewhere,
  measure its offset in the flange frame (`relative_pose` of a located fingertip
  against the measured flange pose) and get the flange pose that places it with
  `dexjoco_rotation_math` `flange_target`.
- Hand joints start at 0 except the thumb's base rotation `th0` at its lower limit
  0.263 rad (it cannot go below). Joint ranges: `x0` (finger spread) -0.47..0.47;
  `x1`, `x2`, `x3` (flexion, base to tip) about -0.2..1.6-1.7; thumb `th0`
  0.263..1.396, `th1` -0.105..1.163, `th2` -0.189..1.644, `th3` -0.162..1.719.
  Targets outside a range are clipped by the actuators.

## Control response

Measured from the start pose through the official step:

- **Arm:** an operational-space controller pulls the flange toward the target with
  a stiff spring and strong damping, so a large jump is followed slowly. A 10 cm
  target change leaves about 6 cm of error after 10 steps, 2 cm after 30 and 1 cm
  after 45. Send long moves as a sequence of nearby targets (a few
  millimetres to 1-2 cm apart), and hold a few steps before precise contact. The arm
  avoids nothing: plan clearance over the table and objects yourself.
- **Elbow:** only the flange pose is commanded; the rest of the arm drifts toward a
  rest posture. Check the overview image for elbow contact on wide or low moves.
- **Hand:** each joint is a position servo with a weak spring
{%- if robot.arms == 'dual' %} (half as stiff as on the single-arm robot): finger joints settle within about 0.03 rad
  after 20 steps and 0.01 rad after 30.
{%- else %}: finger joints settle within about 0.02 rad after 10 steps.
{%- endif %}
 A finger stopped by an object presses with a force that grows with the
  remaining target error, so command flexion beyond the contact angle to squeeze,
  and verify grasps visually; see [hand-control](../hand-control/SKILL.md).
- **Dry run:** `dexjoco_preview_run` runs rows through the same controller on a
  robot-only simulator and shows the arm from fixed cameras without changing the
  episode ([motion-preview](../motion-preview/SKILL.md)).

## Cameras

{% for camera in robot.cameras %}
- `{{ camera }}`{% if 'wrist' in camera %}: on the {% if robot.arms == 'dual' %}{{ camera.split('_')[1] }} {% endif %}hand's mount, moving with it{% else %}: the third-person overview{% endif %}.
{% endfor %}

Images are 640x640 RGB. Motion replies return the final frame inline and save every
10th step as a frame sequence.
{% if camera_geometry %}
Depth images and `camera_intrinsics` / `camera_poses` (world frame) come with every
observation; see [object-localization](../object-localization/SKILL.md).
{% else %}
For locating objects without depth, see [object-localization](../object-localization/SKILL.md).
{% endif %}

{% if multi_env %}
## Environment selection

- Pass the selected `env_id` on every `dexjoco_*` call except `dexjoco_rotation_math`,
  and on `start_episode`, `evaluate` and `finish`; for example
  `dexjoco_step(env_id=i, actions=...)`.
{% if agent_formal %}
  `dexjoco_status` reports one environment; `status`, `dexjoco_rotation_math` and
  `gemini_generate` take no `env_id`.
{% else %}
  `dexjoco_status` reports one environment; `status`, `dexjoco_rotation_math`,
  `gemini_generate`, `register`, `rehearse` and `submit` take no `env_id`.
{% endif %}
- Keep multi-call sequences in one environment, as
  [multi-env exploration](../multi-env-exploration/SKILL.md) describes.
{% endif %}

## Related skills

- [hand-control](../hand-control/SKILL.md) for finger postures, grasps and releases.
- [motion-preview](../motion-preview/SKILL.md) to watch rows on a robot-only simulator
  before running them in the episode.
- [object-localization](../object-localization/SKILL.md) for finding object points in the
  images; `dexjoco_object_states` gives the objects' poses directly.
{% if expert_demonstrations %}
- [learn-from-experts](../learn-from-experts/SKILL.md) for the recorded human
  demonstrations in `runtime/experts/`.
{% endif %}
{% if not formal_episode %}
- [references/pose-records.md](references/pose-records.md) for how poses and executed
  rows are recorded for reuse.
{% endif %}
