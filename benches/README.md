# benches

This folder holds reproducible performance benchmarks and plots.

- `configs/` — YAML configs for runs (`quick_compare.yml` for tonight, `smoke_small.yml` for CI).
- `workloads/` — request stream generators and payloads.
- `runners/` — bench runner, policy factory, metrics.
- `results/` — raw and aggregated CSVs (git-ignored).
- `plots/` — plotting scripts and final images (`plots/out/`).

**Next step:** implement minimal generators + runner to produce CSVs for `quick_compare.yml`.
