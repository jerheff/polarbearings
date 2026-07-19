# Contributing

## Commit messages — Conventional Commits

Releases are automated from commit history, so commits merged to `main` must
follow [Conventional Commits](https://www.conventionalcommits.org/). The prefix
determines the next version bump:

| Prefix | Example | Effect (pre-1.0) |
| --- | --- | --- |
| `fix:` | `fix(split): stable id hash` | patch (`0.1.0` → `0.1.1`) |
| `feat:` | `feat(regression): add huber loss` | minor (`0.1.0` → `0.2.0`) |
| `feat!:` / `BREAKING CHANGE:` footer | `feat!: drop Polars 1.0 support` | minor while < 1.0 (`bump-minor-pre-major`) |
| `docs:` `test:` `chore:` `refactor:` `build:` `ci:` `perf:` | `docs(readme): fix example` | no release |

A scope in parentheses (e.g. `fix(ranking):`) is encouraged but optional.

## How releases work

Automation lives in [`.github/workflows/release-please.yml`](.github/workflows/release-please.yml):

1. **release-please** maintains a rolling "release X.Y.Z" PR that bumps the
   version in `pyproject.toml` and updates `CHANGELOG.md` from the commits above.
2. While that PR is open, every update publishes an **ephemeral alpha**
   (`X.Y.Za<run_number>`) of the pending release to **TestPyPI** — install it with
   `pip install --pre -i https://test.pypi.org/simple/ polarbearings`.
3. **Merging** the release PR tags `vX.Y.Z`, cuts a GitHub Release, and publishes
   the exact version to TestPyPI as a smoke test.

> **PyPI is not wired up yet.** The `publish-pypi` job is a placeholder that only
> logs what a real publish would do. Promoting it to a real publish is a
> follow-up (add a `pypi` trusted publisher + a `pypi` environment with a required
> reviewer, and swap the placeholder for `uv publish`).

### Known follow-ups

- **`uv.lock` self-version lag:** release-please bumps `pyproject.toml` but not
  `uv.lock` (which pins `polarbearings`'s own version). CI uses plain `uv sync`
  (not `--locked`), so this is not fatal, but the committed lock trails by one
  version until the next `uv lock`. A re-lock step on the release PR branch can
  close this when PyPI publishing is promoted out of placeholder status.
