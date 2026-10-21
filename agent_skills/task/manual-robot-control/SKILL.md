---
name: manual-robot-control
{% if formal_episode %}
description: Choose robot actions directly from observations where your carried controller needs help. Use for grounded motion selection, recovery and checking every condition in TASK.md before the episode ends.
{% else %}
description: Solve {{ robot.benchmark }} tasks by directly choosing robot actions from observations during exploration. Use for solving the task by hand, recovery, and physically preparing scenes for script tests.
{% endif %}
---

# Manual robot control

{% if formal_episode %}
Complete the task in this episode through the robot tools and your carried code.
Read `TASK.md` and use current observations to satisfy every condition in it.

## First look
{% else %}
First learn how to complete the task yourself through the robot tools. Read
`TASK.md` and use current observations to satisfy every condition in it. Follow the
plan and budgets in AGENTS.md; confirmed manual success leads to developing an
autonomous controller.

## First exploration
{% endif %}

1. Read `TASK.md` and identify the required object relationships, orientation,
{% if formal_episode %}
   placement and robot end state. Keep a short completion checklist for this episode.
{% else %}
   placement and robot end state. Write a short completion checklist in your notes.
{% endif %}
2. Check `status` and `MCP_SESSION.md`.
{% if multi_env %}
   Use [multi-env exploration](../multi-env-exploration/SKILL.md) for the
   environment you use (a subagent uses the one it was given).
{% endif %}
{% if formal_episode %}
   The episode is already running: inspect it with `{{ robot.status }}` and
   `{{ robot.observe }}`.
{% else %}
   If no episode is active and budget remains, call `start_episode`; otherwise
   inspect the active scene with `{{ robot.status }}` and `{{ robot.observe }}`. Check
   whether that episode has ended before attempting motion.
{% endif %}
{% if demonstrations %}
3. View the current camera images and the supplied demonstration. Identify the
   relevant objects and a short sequence of manipulation stages.
{% else %}
3. View the current camera images. Identify the relevant objects and a short
   sequence of manipulation stages.
{% endif %}
4. Ground the next target in current observations{% if robot.objects %}: read the
   objects' poses with `{{ robot.objects }}`, and use
   [object-localization](../object-localization/SKILL.md) for points it does not name{% else %} with
   [object-localization](../object-localization/SKILL.md){% endif %}, choose a small, inspectable
   action chunk, and pick the motion tool for it as
   [robot-interface](../robot-interface/SKILL.md) describes.
{% if formal_episode %}
5. Inspect feedback and new images and revise the next action. Work through the
   completion checklist until every condition holds.
{% else %}
5. Inspect feedback and new images, revise the next action, and save useful
   parameters and evidence. Work toward the completion checklist, then call
   `evaluate`.
{% endif %}

## Observe and choose actions

{% if formal_episode %}
Start from `{{ robot.observe }}` on the running episode and inspect the returned
observations.
{% elif multi_env %}
Use `start_episode` with the `env_id` of the environment you use to begin exploration
and inspect the returned observations.
{% else %}
Use `start_episode` to begin exploration and inspect the returned observations.
{% endif %}
{% if demonstrations %}
{% if one_demonstration_image %}
`TASK.md` lists the supplied demonstration image; read
[in-context-action-learning](../in-context-action-learning/SKILL.md) before acting.
Infer the task requirements from it, then ground actions in the current scene.
{% else %}
`TASK.md` lists the supplied demonstrations; read
[in-context-action-learning](../in-context-action-learning/SKILL.md) before acting.
Infer the task requirements from them, then ground actions in the current scene.
{% endif %}
{% endif %}

Use these skills as needed rather than loading every reference up front:

- [Robot interface](../robot-interface/SKILL.md) for this robot's motion tools,
  frames and conventions, and its related motion skills. Read it before composing
  actions.
{% if camera_geometry %}
- [Object localization](../object-localization/SKILL.md) for metric positions of
  visible object points from the current images.
{% else %}
- [Object localization](../object-localization/SKILL.md) to locate visible object
  points from the current images.
{% endif %}
{% if wrist_cameras %}
- [Wrist-camera inspection](../wrist-camera-inspection/SKILL.md) to obtain a
  clearer close-up with an empty arm when the {{ robot.overview }} view is
  insufficient; a close wrist view gives more detail than cropping `{{ robot.overview_camera }}`.
{% endif %}
{% if robot.plan %}
- [Free-space motion](../free-space-move/SKILL.md) for planned moves to pose
  targets; inspect goal-reaching feedback, not just planning success.
{% endif %}
{% if formal_episode %}
- [Experience memory](../experience-memory/SKILL.md) to apply the tips and learned
  skills you carried.
{% else %}
- [Experience memory](../experience-memory/SKILL.md) to record and reuse tested
  motion strategies under matching observable conditions.
{% endif %}

{{ block("images.forwarding") }}
Treat previews as hypothetical robot motion, not observations of changed task objects.

## Motion and feedback

[Robot interface](../robot-interface/SKILL.md) says which motion tool fits which
objective on this robot and how its bounded steps behave. The live tool schemas
define argument formats, batch limits and step accounting. Batching reduces calls,
not the number of executed environment steps. Distinguish returned commands from
measured states: the executed prefix is useful replay data, while observations tell
you what physically happened. Read-only inspection and previews do not advance the
episode.

Observe the result of each meaningful action chunk and adapt. When a tool call
fails, read its full error and observe before the next motion. Do not bypass MCP or
blindly repeat the failed
{% if formal_episode %}
motion; keep working with what still functions.
{% else %}
motion; stop and report the tool failure. Do not pause an episode to ask for action
advice; decide from observations and your instructions.
{% endif %}
If optional robot preview fails, continue without it when current evidence is
sufficient.
If localization fails or is uncertain, take a new view; do not replace missing
geometry with guessed coordinates.

{% if formal_episode %}
## Finish

Partial visible progress, a successful plan or a clean tool return is not
completion. Finish as AGENTS.md describes.
{% else %}
## Confirm success and hand off

After strong visible evidence of completion, use `evaluate` for trusted task
confirmation. Native `task_complete: true` also confirms success. An ended episode
without confirmed success is not evidence of completion; do not infer its cause.
Partial visible progress, a successful plan or a clean tool return is insufficient.

After `evaluate` returns false, recover within the remaining episode steps when
possible. Call `evaluate` again after meaningful action and new evidence, not
repeated observation alone. When an episode ends, stop motion.
{% if agent_formal %}
Start the next exploration episode while budget remains.
`exploration_remaining: 0` means no new episodes; it does not remove the current
episode's unused steps.
{% else %}
Start the next exploration episode while budget remains, keeping episodes for
rehearsals. `exploration_remaining: 0` means no new episodes or rehearsals; it does
not remove the current episode's unused steps.
{% endif %}

Record the successful strategy and evidence in `memory/`, then read
[autonomous-control](../autonomous-control/SKILL.md) to implement it, and keep
testing in this episode's remaining steps before you close it with
{% if multi_env %}
`finish` and the same `env_id`{% if not agent_formal %} (`finish` is not a submission){% endif %}.
{% else %}
`finish`{% if not agent_formal %} (it is not a submission){% endif %}.
{% endif %}
{% endif %}
