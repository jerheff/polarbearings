# Changelog

## [0.2.2](https://github.com/jerheff/polarbearings/compare/v0.2.1...v0.2.2) (2026-07-19)


### Continuous Integration

* **release:** attach release artifacts best-effort (immutable releases off) ([#23](https://github.com/jerheff/polarbearings/issues/23)) ([2c2afa5](https://github.com/jerheff/polarbearings/commit/2c2afa55cdc7aa3c1551be2bdc9debf5aabf3a7d))

## [0.2.1](https://github.com/jerheff/polarbearings/compare/v0.2.0...v0.2.1) (2026-07-19)


### Continuous Integration

* **release:** real PyPI publish (build-once, approval-gated) + release assets ([#21](https://github.com/jerheff/polarbearings/issues/21)) ([2c69222](https://github.com/jerheff/polarbearings/commit/2c69222893814c43311388e5f7ffb455e72a949e))

## [0.2.0](https://github.com/jerheff/polarbearings/compare/v0.1.0...v0.2.0) (2026-07-19)


### ⚠ BREAKING CHANGES

* **api:** make tuning params keyword-only across all metric families

### Features

* accept weight as an expression on every metric ([bebb068](https://github.com/jerheff/polarbearings/commit/bebb068d906b306688a598bb34ac655dc03e732b))
* Add log loss, Brier score metrics and CI/CD ([9f9963a](https://github.com/jerheff/polarbearings/commit/9f9963a16bece3ca48f2f5054bec40e0d7965d64))
* add normalized Gini coefficient metric; reorganize tests per-metric ([5a27be7](https://github.com/jerheff/polarbearings/commit/5a27be7fbe59c6996acbc6cbef3951edaa45e3be))
* **api:** make tuning params keyword-only across all metric families ([1e2ce54](https://github.com/jerheff/polarbearings/commit/1e2ce54cdef17a50196cad24bc514509315b8e19))
* bootstrap confidence intervals for any metric ([8934e7a](https://github.com/jerheff/polarbearings/commit/8934e7a437cf4b951b038de5160cef8c138b6c58))
* bootstrap_weight — composable replicate weights ([7479d3c](https://github.com/jerheff/polarbearings/commit/7479d3ca5220e330a275092927053f26dd94e02e))
* deterministic, id-keyed data splitting ([140befe](https://github.com/jerheff/polarbearings/commit/140befed3e8f0197a19406f3f718e4d4d132c312))
* diagnostic curves, thresholds= grid, calibration by= ([c969a5c](https://github.com/jerheff/polarbearings/commit/c969a5c9931fcc962e5a28a15154420eeceb9d73))
* ECE / MCE calibration metrics ([270ffea](https://github.com/jerheff/polarbearings/commit/270ffea8cd245b3f06a1c74262bc8a7cbf34b30c))
* metric expansion, curves, threshold specs, IntoExpr, lazy frames, missing-value policy ([b40220e](https://github.com/jerheff/polarbearings/commit/b40220e1fec0685aada6f738879d9f3906024a58))
* **regression:** add 9 regression and quantile-loss metrics ([c8c8a1e](https://github.com/jerheff/polarbearings/commit/c8c8a1e30943f01ef23b1823c715a3c9c59021ec))
* support arbitrary positive class via pos_label on binary metrics ([8770431](https://github.com/jerheff/polarbearings/commit/87704319066b5dc7967f6ebdd6586cd708aead60))


### Bug Fixes

* **bootstrap,ranking:** friendly errors for audit findings 2, 3, 6 ([#9](https://github.com/jerheff/polarbearings/issues/9)) ([d9e8c8c](https://github.com/jerheff/polarbearings/commit/d9e8c8c8e2d3a3061fdc081a1b01f46b1ce6b9ab))
* **ci:** drop --check-url from uv publish (conflicts with --index) ([#16](https://github.com/jerheff/polarbearings/issues/16)) ([7375a81](https://github.com/jerheff/polarbearings/commit/7375a81783262274a37e89a23ee440ed48300bc8))
* make the cooldown canary all-core (Polars sort, not single-thread) ([27fd3cb](https://github.com/jerheff/polarbearings/commit/27fd3cb9cf68c374b3972a54d474b65879854347))
* **notebook:** emit the Plotly mimetype so figures render in VS Code ([d2c0f03](https://github.com/jerheff/polarbearings/commit/d2c0f03efe560527cefb11f55fbb409bb44279fe))
* **regression:** d2_pinball_score group_by panic on Polars 1.24.0 ([cd6767b](https://github.com/jerheff/polarbearings/commit/cd6767bafc1586427016b91d47fbcc657f48f8a8))
* **roc_auc:** detect score ties with max==min, not var==0 (underflow) ([0173f1f](https://github.com/jerheff/polarbearings/commit/0173f1f10bb77e55e07a35c45dc91c1694f14d93))
* **roc_auc:** prevent UInt32 overflow in Mann-Whitney U above ~131k rows ([5217ef8](https://github.com/jerheff/polarbearings/commit/5217ef8580d7e68298e7f2a6b477eef7d0c7135b))
* **split:** version-stable id hash via SplitMix64 ([#7](https://github.com/jerheff/polarbearings/issues/7)) ([cb879e3](https://github.com/jerheff/polarbearings/commit/cb879e39703a5f0a2f65bd9bae678f86877991d3))


### Performance Improvements

* **average_precision:** drop redundant (1-target) sort + cum_sum ([9b76e9a](https://github.com/jerheff/polarbearings/commit/9b76e9ae06ac69f9d13dc5b13063f10457117f63))
* **calibration,curves:** version-gated single-pass fast paths ([34fbd1a](https://github.com/jerheff/polarbearings/commit/34fbd1acd950f1653dae475eb7822463c287b071))
* **classification:** boolean-mask confusion cells; add confusion_matrix struct ([a99a3f5](https://github.com/jerheff/polarbearings/commit/a99a3f57281aa8a03fed5d29a0be4f7d75dd3ea5))


### Documentation

* Add comprehensive README and development tooling ([8a57062](https://github.com/jerheff/polarbearings/commit/8a57062490b9f78edc468d1378437b78939a95cb))
* add polars release analysis, extended through 1.41 ([a1d93b7](https://github.com/jerheff/polarbearings/commit/a1d93b7b6940e444dd41b85e9815f06953abce80))
* correct performance claims; document strengths, weights, and pos_label ([3fb5be4](https://github.com/jerheff/polarbearings/commit/3fb5be4f0cbfcd76b08213d8e8db44cccf9f8580))
* document curves, splitting, bootstrap; refresh diagnostics notebook ([655235b](https://github.com/jerheff/polarbearings/commit/655235b0ec4a93dd0b5271235169b7e13c1e1208))
* fix metric count, document MAPE divergence, drop stale docs ([5081411](https://github.com/jerheff/polarbearings/commit/5081411cc40fb0e093ed4f86f5e33528c176ae29))
* mark audit finding 14 resolved (branch coverage via combine) ([5995998](https://github.com/jerheff/polarbearings/commit/5995998ca34d7bc27ac22623715d3c8ececf4ef9))
* **notebook:** use plotly.express for the diagnostic curves ([617e4c0](https://github.com/jerheff/polarbearings/commit/617e4c098c7de8d46681e9b197aa75c91565f2e2))
* **readme:** PyPI-ready install + absolute links ([#18](https://github.com/jerheff/polarbearings/issues/18)) ([4da0b31](https://github.com/jerheff/polarbearings/commit/4da0b316aa9023f2360eecb6e3efdca57a87bb08))
* **readme:** split metrics reference into docs/, refresh roadmap ([#6](https://github.com/jerheff/polarbearings/issues/6)) ([f889bb0](https://github.com/jerheff/polarbearings/commit/f889bb02b986ec519e9aaff3dabca2aed5ae0dbf))
* reconcile metric counts, weight claims, and version-gating notes ([42e914e](https://github.com/jerheff/polarbearings/commit/42e914e0c9af65628a8eb2e2ac100a7e7a9b577a))
* refresh PERFORMANCE.md benchmark tables ([e13b003](https://github.com/jerheff/polarbearings/commit/e13b003c55c86ae8f7726557d31ea6ddb47ed5d5))
* show uv install commands by default (pip in a comment) ([639359c](https://github.com/jerheff/polarbearings/commit/639359cd8f6eb6f2e2c01de48afa299fb85c695d))
* showcase computing a whole metric suite in one select ([bc36884](https://github.com/jerheff/polarbearings/commit/bc36884441c4d60b32d8eb54c8289b87ee7da41e))
* Update README and add example script ([c8d26a3](https://github.com/jerheff/polarbearings/commit/c8d26a31011f05a962d4f890824bfddbd00cd193))


### Code Refactoring

* drop unused __version__ from the package ([59dd527](https://github.com/jerheff/polarbearings/commit/59dd527a3b5b569f35fae1c445285ef9e127a495))
* Restructure project and improve ROC AUC implementation ([aa8d7f1](https://github.com/jerheff/polarbearings/commit/aa8d7f129fb2e90aa35bb78e57dc9147a8326e11))
* **typing:** Literal methods, BootstrapCI TypedDict, shared PosLabel ([8990845](https://github.com/jerheff/polarbearings/commit/8990845cd196f3a6c1c9d8e6b2459e06a5ba86f9))


### Build System

* add uv-lock, case-conflict, private-key hooks + ruff T10 ([26f8e25](https://github.com/jerheff/polarbearings/commit/26f8e255e04e2fea50f2466436177dee6842d6b0))
* bound the hatchling build backend to &gt;=1.30,&lt;2 ([e84e8c7](https://github.com/jerheff/polarbearings/commit/e84e8c71761ab27543ff55af5b21b6a76e2cb8eb))
* commit uv.lock for reproducible environments ([18e4112](https://github.com/jerheff/polarbearings/commit/18e41121cb95b9ea6c45b652686ac10f14d2bd3f))
* **deps:** bump dev tooling + notebooks; clarify numpy/sklearn are dev-only ([95a6bf4](https://github.com/jerheff/polarbearings/commit/95a6bf4029b27de63e0d68691bd1b445dbc36497))
* **deps:** support Polars 1.42.0; gate explode empty_as_null ([6e1ae61](https://github.com/jerheff/polarbearings/commit/6e1ae61c26b3a2ebcfd25ab049525d0b0aad5c9a))
* develop against lower-bound dependencies (lowest-direct) ([22efcd7](https://github.com/jerheff/polarbearings/commit/22efcd782179ac3791623006dce6f188bbd2af60))
* drop python-lsp-server from dev deps ([a5c88ba](https://github.com/jerheff/polarbearings/commit/a5c88ba4334473a26f53331a5dc5147d216faa68))
* fill in [project] metadata for publishing ([a7dc32a](https://github.com/jerheff/polarbearings/commit/a7dc32afa9e45f203b8b74f2350cc8975b21a4c6))
* replace pyright with ty, bump ruff and pre-commit hooks ([b325578](https://github.com/jerheff/polarbearings/commit/b32557855ee042a19f8a40e460326f2e93f24168))
* require uv &gt;=0.11 ([5b977fa](https://github.com/jerheff/polarbearings/commit/5b977faf91ccc2e7dc447f5f766f337113340c53))
* scope the sdist to source + tests + metadata ([90772ea](https://github.com/jerheff/polarbearings/commit/90772eab75817f4c1aff503d2fc05ef308da3c12))
* switch to the uv build backend and configure Test PyPI publishing ([3cae958](https://github.com/jerheff/polarbearings/commit/3cae958ef84d21dc02aefcbb38348bde6498b395))
* tighten ruff — enforce annotations, docstrings, and more ([6722834](https://github.com/jerheff/polarbearings/commit/6722834d6871bb0e54c3e925b41a09602e32d1b3))


### Continuous Integration

* add per-workflow rollup gate jobs for branch protection ([f083488](https://github.com/jerheff/polarbearings/commit/f0834882c29d61dd51a2cf4043e4f876e7654ec5))
* **benchmark:** run benchmarks manually, not on every PR ([#14](https://github.com/jerheff/polarbearings/issues/14)) ([59db1f1](https://github.com/jerheff/polarbearings/commit/59db1f1b1c781817469347e228534314a58567a3))
* bump latest Polars compatibility target to 1.41.2 ([57157c6](https://github.com/jerheff/polarbearings/commit/57157c6cfe657f5a482056bfe117bd497f48ffcc))
* bump release-please-action v4 -&gt; v5 (Node 24); pins now current ([#17](https://github.com/jerheff/polarbearings/issues/17)) ([5d0c9ca](https://github.com/jerheff/polarbearings/commit/5d0c9ca524c199ef7077e3533ccaca032c3bd7c1))
* comment pinned TestPyPI install command on the release PR ([#19](https://github.com/jerheff/polarbearings/issues/19)) ([e923b26](https://github.com/jerheff/polarbearings/commit/e923b26b6c944a5e4dd3783f0af890b65e163017))
* diff-scoped prek hooks, fixed hypothesis job, centralized coverage gate ([1ad447a](https://github.com/jerheff/polarbearings/commit/1ad447ac79da6d035c5acf809bd6b3068373b3be))
* drop dependabot.yml — bump action pins deliberately, not on a schedule ([90a3a2d](https://github.com/jerheff/polarbearings/commit/90a3a2d70b1c4843cddbe277cfb9acda52353369))
* harden workflow — SHA-pin actions, bump setup-uv v8, least-privilege ([f0c6a31](https://github.com/jerheff/polarbearings/commit/f0c6a3109d968f9ffde242adb94d334b3aaa9a29))
* move deep Hypothesis fuzzing from every PR to a nightly job ([#12](https://github.com/jerheff/polarbearings/issues/12)) ([bee4618](https://github.com/jerheff/polarbearings/commit/bee461853bb627fb9005e6cc000a35d906a9eb33))
* release-please + TestPyPI publishing (PyPI placeholder) ([#11](https://github.com/jerheff/polarbearings/issues/11)) ([8319e3b](https://github.com/jerheff/polarbearings/commit/8319e3b2476b93bd4f1f291099db736f9d24d3a5))
* **release-please:** show all commit types in the changelog ([#20](https://github.com/jerheff/polarbearings/issues/20)) ([94ec334](https://github.com/jerheff/polarbearings/commit/94ec334e30717ef74468ad39d0be688b82913701))
* split the monolithic CI workflow into Tests/Lint/Build/Benchmark ([891ec83](https://github.com/jerheff/polarbearings/commit/891ec83fae992381734d8b2fd9adbae0189f9350))


### Tests

* Add comprehensive edge case tests and benchmarks ([d3165cc](https://github.com/jerheff/polarbearings/commit/d3165cc648e30790c69acf2edad657794a69d496))
* add wheel/packaging test (just test-wheel + CI package job) ([1ca5675](https://github.com/jerheff/polarbearings/commit/1ca56754e30af956750e08eb33ffc9ce92f7af14))
* **bench:** benchmark computing many metrics in one select ([4163785](https://github.com/jerheff/polarbearings/commit/41637852fb1700c750b53253936a66b6dfd9bc54))
* **bench:** rigorous config, shared fixtures, new metrics, version comparison ([4af6df2](https://github.com/jerheff/polarbearings/commit/4af6df24986d3574dfd6e20db1f4e43502587d43))
* close mutation-testing gaps in fbeta/mcc/cohens_kappa ([9f7360c](https://github.com/jerheff/polarbearings/commit/9f7360c6be45fdee6a59bf43961bf36aa99e78d5))
* **coverage:** branch coverage via cross-version combine; drop pragmas ([e40c676](https://github.com/jerheff/polarbearings/commit/e40c67627415231f0970dcbdb141cd4d109df85f))
* deepen thorough profile, add local multi-version sweep ([dac4070](https://github.com/jerheff/polarbearings/commit/dac40707a0cc4e5b9c84e65afaca2cb04e131b23))
* document mutmut's fork/Polars unreliability; simplify config ([587c686](https://github.com/jerheff/polarbearings/commit/587c6861d762b23d7a41b0aebabb32eccbefba73))
* fix mutmut so mutations are actually exercised ([3315db8](https://github.com/jerheff/polarbearings/commit/3315db81e9215d4e6e968081d8e83adcbd8e54d8))
* fix mutmut timeouts at the source (thread churn), not the budget ([23f3b09](https://github.com/jerheff/polarbearings/commit/23f3b09f856211d9911fae47e601ac5ab8fcb032))
* per-metric null-policy contract and DCG/NDCG tie behavior ([6c77368](https://github.com/jerheff/polarbearings/commit/6c77368a00d744abb90f63d9b68ab75486c9ac28))
* raise mutmut time budget to cut spurious timeouts ([ac7ab34](https://github.com/jerheff/polarbearings/commit/ac7ab345f2ba41c3274dd1017f829c2062d8262e))
* track conftest.py and harden Hypothesis test config ([b915dd9](https://github.com/jerheff/polarbearings/commit/b915dd99fb9c944ed3bd897189d29e94f25b27dd))


### Miscellaneous Chores

* **deps:** bump dev tooling, prek hooks, and pinned actions; add `just outdated` ([9b82703](https://github.com/jerheff/polarbearings/commit/9b8270317ddaa64ca8ba686e2822d57e14dd4cc0))
* gitignore Claude Code local agent state ([58eb728](https://github.com/jerheff/polarbearings/commit/58eb7281c041a61cc513be9a65b28b3a70ad3190))
* rename package polarbear -&gt; polarbearings ([2232a67](https://github.com/jerheff/polarbearings/commit/2232a67fbb85266672fc29067f87fb4a2a674547))
* use native HTTP transport for the ask_polars MCP server ([9c2fb96](https://github.com/jerheff/polarbearings/commit/9c2fb963b17bedc601d1027648cc1fc96326c1ad))

## Changelog

All notable changes to this project are documented in this file.

This file is maintained automatically by
[release-please](https://github.com/googleapis/release-please) from
[Conventional Commit](https://www.conventionalcommits.org/) messages; do not
edit it by hand.
