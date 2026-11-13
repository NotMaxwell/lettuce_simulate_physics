# Multi-agent workflow

Project-independent. Imported from CLAUDE.md; the project's CLAUDE.md supplies the build/verify
commands, the file-ownership map and anything repo-specific. Agent definitions live in
`.claude/agents/`.

## Roles
- **Lead (the main session)** plans the split, defines interfaces before anyone codes, owns the
  shared files (build files, data contract, CLAUDE.md), integrates, runs final verification,
  commits. The lead writes no code (user rule, 2026-11-12): code changes, including small fixes
  and changes to lead-owned contract files, are specified by the lead and written by an agent;
  review is a separate agent. The lead edits only docs (CLAUDE.md, TODO.md, .claude/).
- **Container agents** are the default workers: headless `claude -p` sessions, one per git
  worktree, each in its own dev container (recipe in CLAUDE.md). In-session subagents are for
  read-only work (Explore, verifier) or when containers are unavailable.
- **Subagents** each own one file set and get a way to verify alone. No two agents edit the same
  file at the same time.

## Which agent (cheapest that can do the job)
| Agent | Model | Effort | Use for |
|---|---|---|---|
| `chore` | Haiku | low | Fully specified mechanical edits, renames, boilerplate, docs |
| `verifier` | Haiku | low | Running build/tests/bench and summarising the logs |
| `Explore` (built-in) | — | — | Broad read-only search when only the conclusion is needed |
| `builder` | Sonnet | medium | Ordinary module/feature work, ports, located bug fixes, tests |
| `optimizer` | Opus | high | SIMD, threads, GPU kernels, spatial structures; must report before/after numbers |
| `architect` | Fable | high | Read-only: interface design before a split, unreproducible bugs, hard trade-offs |

Start low and escalate once on failure (chore → builder → optimizer); do not start high "to be
safe". Override a definition's model only with a reason.

## Spawning
- Parallel only when owned file sets are disjoint. Otherwise run sequentially, or give each agent
  `isolation: "worktree"` and merge afterwards.
- Every prompt states: goal · owned files · interface/contract to code against · files not to
  touch · verify command · report format. Pass facts the lead already knows instead of making the
  agent rediscover them.
- Follow-ups go to the same agent (SendMessage), not a fresh spawn.
- Reports are short (the agent files cap them) and contain results, not file dumps. The lead
  trusts the report for what it covers and spends its own reads on verification and integration.
- After integrating, the lead (or a `verifier`) runs the full build, tests and benchmarks and
  reports real output.

## Efficiency
- Keep CLAUDE.md and this file short: every agent loads them on every turn.
- Push long logs and broad searches into `verifier`/`Explore` so they do not sit in the lead's
  context.
- Measure performance changes with the project's benchmark before and after; never estimate.

## Parallel sessions in containers
For independent long-running sessions (not subagents), give each its own git worktree and its own
dev container, so builds, caches and permissive permission modes stay isolated from the host and
from each other. The project's CLAUDE.md has the exact commands. The container isolates the
filesystem only; the network is open, so do not put secrets in it.
