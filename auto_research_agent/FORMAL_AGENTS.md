# {{ robot.benchmark }} episode

You are the agent that explored this task, continuing it in one live episode on a
layout you have not seen. The episode is already running. Complete the task in it with
your carried code as the controller, and control the robot yourself where it needs
help: satisfy every condition in `TASK.md`. There is no reset, retry or second episode.
{% if partial_credit %}

Make this episode score as high as it can. It scores 1.0 if every goal condition holds
at the end; otherwise it earns partial credit for each goal condition that was false at
the start and holds at the end (`TASK.md`, Scoring). Aim for success, and when the whole
task is out of reach, secure as many conditions as you can and leave them holding.
{% endif %}

## Start here

Call `status`: it reports{% if wall_limit %} this episode's wall limit (`episode_timeout_seconds`, startup
included),{% endif %} elapsed time, the folders you carried (`carried`) and how
much of your earlier conversation you have (`conversation`). Then read `TASK.md` and
what you carried, most useful first: `handoff/README.md` if present,
`memory/PROGRESS.md`, `memory/TIPS.md` and `memory/INDEX.md`, and the code under
`code/`.
{% if demonstrations %}
View the demonstration `TASK.md` lists under `runtime/demonstrations/`.
{% endif %}
Episode IDs, frame paths and coordinates from earlier work are stale. Observe the
current scene before acting.

Automate as much of the episode as you can. Run your carried code for every part it
covers, watch each step, adjust it to this layout, and control the robot by hand only
for what it was not built for, as
[autonomous-control](.agents/skills/autonomous-control/SKILL.md) describes. Your code
acts faster and more consistently than live control{% if wall_limit %}, and the wall limit counts your
own thinking time{% endif %}.

The skill links below are entry points; read only what the next step needs.

- [manual-robot-control](.agents/skills/manual-robot-control/SKILL.md): choose and
  verify robot actions from the current observations.
- [experience-memory](.agents/skills/experience-memory/SKILL.md): use the tips and
  learned skills you carried.
- [robot-interface](.agents/skills/robot-interface/SKILL.md): this robot's motion
  tools, frames and conventions, and the motion skills it links to.
{% if robot.objects %}
- `{{ robot.objects }}` for the objects' exact poses.
{% endif %}
{% if wrist_cameras %}
- [object-localization](.agents/skills/object-localization/SKILL.md) for positions of
  visible object points, and
  [wrist-camera-inspection](.agents/skills/wrist-camera-inspection/SKILL.md) for close-ups.
{% else %}
- [object-localization](.agents/skills/object-localization/SKILL.md) for positions of
  visible object points.
{% endif %}
{% if robot.plan %}
- [free-space-move](.agents/skills/free-space-move/SKILL.md) for moving a hand to a pose.
{% endif %}
{% if method_skills %}
- [keep-trying](.agents/skills/keep-trying/SKILL.md),
  [spend-the-budget](.agents/skills/spend-the-budget/SKILL.md),
  [measure-success](.agents/skills/measure-success/SKILL.md),
  [verify-committal-actions](.agents/skills/verify-committal-actions/SKILL.md) and
  [match-precision-to-error](.agents/skills/match-precision-to-error/SKILL.md): how to use an
  episode well.
{% endif %}
- [gemini](.agents/skills/gemini/SKILL.md) for visual and spatial perception.

## How the episode ends

- A reply with native `task_complete: true` means success, and the episode ends.
- The episode also ends at the task's step limit{% if wall_limit %} or the wall limit{% endif %}.
- Ending your turn ends the episode as it stands. Before you stop, check fresh
  observations against every condition in `TASK.md` at once{% if demonstrations %}, comparing them with
  the demonstration{% endif %}, and fix what is missing while steps remain.
- Once a reply reports `episode_ended`, stop acting; further actions have no effect.

No one reads a report; if a tool fails, carry on with what still works.

## Tools and code

`MCP_SESSION.md` lists every tool this episode exposes. You may run your own code:
ordinary Python uses `from api.runtime import Context`, and
`ctx.call(tool, **arguments)` acts on this episode, as autonomous-control shows.

## Time and budgets

{% if wall_limit %}
The wall limit includes your own thinking and tool time, and the task step limit still
applies.
{% else %}
There is no wall-clock limit; the task step limit applies.
{% endif %}
Let your carried code do what it covers and prefer tested procedures over new
experiments. Every reply reports the remaining episode steps; keep enough of them for
the conditions that must hold at the end. Gemini spending has no limit; each reply
reports its cost in `budget`.

## Workspace

{% if demonstrations %}
This workspace is a fresh, private copy holding these instructions, the skills,
`TASK.md`, the demonstration and the carried folders.
{% else %}
This workspace is a fresh, private copy holding these instructions, the skills,
`TASK.md` and the carried folders.
{% endif %}
Frames of this episode appear under `runtime/autonomous_controller/runs/`. There is no
internet or simulator access, and you can write only in this workspace. Write notes only if they
help you within this episode.
