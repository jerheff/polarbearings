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
   A `relock` job then runs `uv lock` and commits the refreshed `uv.lock` onto
   that same PR branch, so merging lands a lockfile whose `polarbearings` version
   matches the bump.
2. While that PR is open, every update publishes an **ephemeral alpha**
   (`X.Y.Za<run_number>`) of the pending release to **TestPyPI** — install it with
   `pip install --pre -i https://test.pypi.org/simple/ polarbearings`.
3. **Merging** the release PR tags `vX.Y.Z` and cuts a GitHub Release. Then, built
   once and published twice: the sdist + wheel are attached to the Release, then
   published to **TestPyPI** as a smoke test, and then to **PyPI** — gated behind
   the `pypi` environment's required reviewer (manual approval).

### One-time setup for real PyPI publishing

- A **PyPI trusted publisher** (pypi.org → the project → Publishing): repo
  `jerheff/polarbearings`, workflow `release-please.yml`, environment `pypi`.
- A **`pypi` GitHub Environment** with a **required reviewer**, so the
  `publish-pypi` job pauses for manual approval before uploading to PyPI. Its
  deployment-branch policy must allow **`main`** (the workflow runs on the
  release-PR merge to `main`, not on the tag).
- The repo's **"immutable releases" setting OFF** (Settings → General) so
  `build-release` can attach the sdist + wheel to each Release. The immutable
  **tags** ruleset (`version-tags`) stays on — that's the important guarantee.
