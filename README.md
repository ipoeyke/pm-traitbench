# pm-traitbench

This repo builds the pipeline that generates PM-TraitBench, a synthetic dataset of portfolio manager
behavioural traits for evaluating a copilot's behavioural memory.

The dataset itself is described in [`docs/`](docs/). Start with
[`docs/pm-dataset-plan.md`](docs/pm-dataset-plan.md).

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```sh
uv sync
```

## Usage

```sh
uv run pm-traitbench --help
```

## Development

```sh
uv run pytest
uv run ruff check
uv run ruff format
```
