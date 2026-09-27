# Contributing

This document covers two things: how to contribute changes, and how this
project tracks **contribution evidence** — a clear, verifiable record of who
did what and when, which matters for coursework/capstone evaluation as much
as for a normal open-source project.

## Development workflow

```bash
git checkout -b feature/<short-description>   # e.g. feature/fedprox-strategy
# make your changes
git add <files>
git commit -m "type: short description"
git push origin feature/<short-description>
# open a pull request against main
```

### Commit message convention

Use a `type: description` prefix so history is scannable at a glance:

| type | for |
|---|---|
| `feat` | a new capability (e.g. `feat: add FedProx strategy option`) |
| `fix` | a bug fix (e.g. `fix: correct DP epsilon formula overflow`) |
| `docs` | documentation only |
| `refactor` | code change with no behavior change |
| `test` | adding or fixing tests |
| `chore` | tooling, dependencies, config |

Write commits at the size of one reviewable idea — not one file, not one day.
A commit that mixes an unrelated formatting pass with a real bug fix makes
both harder to review and harder to attribute later.

### Pull requests

Every PR description should state, in a sentence or two: what changed, why,
and how you verified it (a test run, a screenshot, sample output). This is
what turns a diff into evidence someone actually exercised the change rather
than just writing code that compiles.

## Contribution evidence

If you need to demonstrate individual contribution (e.g., for a course
submission or team project evaluation), this repo's git history is the
primary artifact. A few ways to generate a clean summary of it:

**Per-author commit counts:**
```bash
git shortlog -sn --all
```

**A specific contributor's full commit history with diffs:**
```bash
git log --author="Your Name" -p
```

**Lines added/removed per author:**
```bash
git log --author="Your Name" --pretty=tformat: --numstat \
  | awk '{ add += $1; del += $2 } END { print "added:", add, "deleted:", del }'
```

**A changelog-style summary for a report:**
```bash
git log --pretty=format:"%h %ad %s" --date=short --author="Your Name"
```

When submitting this project for evaluation, it's worth including a short
`CONTRIBUTORS.md` (see the template below) alongside one of the git log
exports above as supporting evidence, rather than relying on the reviewer to
run these commands themselves.

### CONTRIBUTORS.md template

```markdown
# Contributors

## <Your Name>
- Designed and implemented the federated learning pipeline (src/, data/)
- Implemented the differential-privacy mechanism (src/privacy.py)
- Built the role-based dashboard (dashboard/)
- Commits: see `git shortlog -sn --author="<Your Name>"`
```

## Code style

- Python: standard library + the packages already in `requirements.txt`;
  keep functions small and each module focused on one concern (data,
  model, privacy, client, server are already separated for this reason).
- Prefer clear names over comments explaining unclear ones.
- Any change to `src/privacy.py` should come with a note in its own
  docstring about what guarantee changed and why — that file is the whole
  point of the "privacy-preserving" half of this project's name.
