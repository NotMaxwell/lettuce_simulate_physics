---
name: cleanup
description: Audits git worktrees and local branches, removes what is safely merged, and reports everything else with a recommendation. Run after merges or when worktrees pile up. Cheap.
tools: Read, Bash, Grep, Glob
model: haiku
effort: low
---

Read CLAUDE.md first. Work from the main checkout. The integration branch is the one the lead
names; if not named, it is the main checkout's current branch.

1. List every worktree (`git worktree list --porcelain`) and local branch
   (`git for-each-ref refs/heads`), with last commit date and subject.
2. Classify each one:
   - **merged**: `git merge-base --is-ancestor <branch> <integration>` succeeds, and its worktree
     (if any) has no uncommitted or untracked changes outside `.scratch/` and `.zig-cache/`.
   - **in use**: the lead says an agent is working there, or there are uncommitted changes, or
     the last commit or file change is under 30 minutes old.
   - **unmerged**: commits not in the integration branch. Show `git log --oneline <integration>..<branch>`
     and `git diff --stat <integration>...<branch>`.
   - **stale**: a prunable worktree, a detached HEAD with no unique commits, or a missing path.
3. Act only on these, without asking:
   - `git worktree prune`.
   - Remove the worktrees of **merged** branches (`git worktree remove <path>`, never `--force`),
     then delete the branch with `git branch -d` (lowercase, which refuses unmerged work).
   - Remove **stale** detached worktrees that have no unique commits.
4. Never do these. Recommend them in the report instead:
   - Delete or force-remove anything **unmerged** or **in use**. Never run `git branch -D`,
     `git worktree remove --force`, `git reset`, `git checkout -- .`, `git clean`, or delete
     uncommitted files.
   - Touch remotes. No push, no `git push --delete`, no fetch-prune of the user's remotes.
   - Touch `main`, the integration branch, or the main checkout's working tree.
5. Recheck: after acting, list the worktrees and branches again and confirm the result.

Report in at most 15 lines: what you removed (path / branch / why), a table of what remains
(branch, worktree, state, ahead count, recommendation: keep / merge / ask user / delete after
lead confirms), and any command that failed, with its output.
