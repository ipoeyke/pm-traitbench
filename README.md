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
uv run pm-traitbench --help
```

## Development

```sh
uv run pytest
uv run ruff check
uv run ruff format
```
