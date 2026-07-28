# 🗺️ Onboard Command

Use this command to re-enter the project context quickly or explain it to a new contributor.

## What This Does

The "Guide" that synthesizes documentation, history, and plans into a layered explanation. It assumes you have verified the docs with `/smart-commit`.

## How to Use

Type `/onboard` to start the interactive session.

## The Layers

**Level 1: The Context (30 Seconds)**
- What is this project?
- What is the current active goal? (Reads `LOCKED PLAN`).
- "We are building an RL cluster for SC2. Currently optimizing reward shaping."

**Level 2: The Architecture**
- How does data flow?
- Key modules (`environment`, `agent`, `training`).
- "Data enters via `sc2_env`, gets normalized in `wrappers`, and fed to PPO."

**Level 3: The History (Recent)**
- Reads the last 5 entries of `docs/about/changelog.md`.
- "Last week we refactored the action space. Before that, we fixed a NaN bug."

**Level 4: The 'Gotchas'**
- "Watch out for: The replay buffer requires 16GB RAM."
- "Known issues: Simulation creates zombie processes."

## Output

A structured "Welcome Back" briefing.

---

**Ready?** Say "Onboard me" to start.
