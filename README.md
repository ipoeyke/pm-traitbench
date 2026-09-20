# pm-traitbench

This repo builds the pipeline that generates PM-TraitBench, a synthetic dataset of portfolio manager
behavioural traits for evaluating a copilot's behavioural memory.

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```sh
uv sync
```

## Usage

```sh
uv run pm-traitbench sample --config configs/demo.yaml --data-dir data
```

This runs the `sample` stage, which writes four tables to `data`: `personas`,
`traits`, `rules` and `drift_events`. Pass `--force` to overwrite a table that
already exists. Run `uv run pm-traitbench --help` for the full command list.

## Development

```sh
uv run pytest
uv run ruff check
uv run ruff format
```
