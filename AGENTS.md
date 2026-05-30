# AGENTS.md

## Project Goal

Implement experiments to test whether LoRA B-matrix similarity and SVD-LoRA `B|E|` similarity can distinguish heterogeneous client tasks.

The repository should support independent client training runs, extraction of adapter similarity features, and standalone similarity evaluation. Do not implement federated aggregation in this project yet.

## Core Rules

- Use Python and PyTorch.
- Use Hugging Face `transformers`, `datasets`, and `peft` for standard LoRA.
- Implement SVD-LoRA manually as a custom wrapper around selected `torch.nn.Linear` layers.
- Keep experiment scripts modular and reusable.
- Save every generated artifact under `outputs/`.
- Use fixed random seeds for data sampling, model initialization, adapter initialization, and training.
- Train each client independently from the same pretrained base model and the same adapter initialization.
- Keep similarity evaluation independent from training code.
- Add clear CLI arguments for scripts.
- Keep README usage examples current when adding or changing experiment entry points.

## Expected Repository Structure

Prefer this structure as the codebase grows:

```text
.
|-- README.md
|-- AGENTS.md
|-- requirements.txt
|-- configs/
|-- src/
|   |-- data/
|   |-- models/
|   |-- training/
|   |-- similarity/
|   `-- utils/
|-- scripts/
`-- outputs/
```

Use `src/` for reusable implementation code and `scripts/` for command-line experiment entry points. Keep notebooks, one-off analysis, and generated files out of the core library path unless there is a strong reason.

## Experiment Design Constraints

- Each client task should be trained as a separate run.
- All clients in a comparison must start from:
  - the same pretrained model checkpoint,
  - the same frozen base model weights,
  - the same adapter initialization,
  - comparable training hyperparameters unless the experiment explicitly varies them.
- Heterogeneity should come from task/data differences, not uncontrolled initialization drift.
- Do not mix training and similarity computation in the same implementation module. A training script may optionally call a separate extraction or evaluation entry point, but the similarity logic must live independently.
- Standard LoRA and SVD-LoRA experiments should expose comparable outputs so downstream similarity code can consume both.

## Standard LoRA Implementation

Use PEFT for standard LoRA.

When adding or editing standard LoRA code:

- Configure adapters through `peft.LoraConfig`.
- Make target modules explicit through CLI/config arguments.
- Save adapter weights and metadata under `outputs/`.
- Ensure B matrices can be extracted consistently from saved adapters.
- Avoid custom PEFT internals unless the public API cannot support the experiment.

## SVD-LoRA Implementation

Implement SVD-LoRA manually as a custom wrapper for selected `torch.nn.Linear` layers.

The wrapper should:

- freeze the original linear weight and bias unless an experiment explicitly requires otherwise,
- add trainable low-rank parameters for the SVD-LoRA path,
- expose or save the matrices needed to compute `B|E|` similarity,
- support deterministic initialization from a provided seed,
- preserve the original module interface as much as possible,
- be limited to explicitly selected linear layers.

Keep SVD-LoRA code isolated from PEFT-specific standard LoRA code so the two implementations can be compared clearly.

## Similarity Evaluation

Similarity evaluation must be independent from training.

Similarity code should:

- load trained adapter artifacts from `outputs/`,
- extract standard LoRA B matrices or SVD-LoRA `B|E|` representations,
- compute pairwise similarities across clients,
- save matrices, tables, and plots under `outputs/`,
- include enough metadata to trace each similarity result back to model, dataset, seed, client, adapter type, and target modules.

Do not require a training script import to run similarity evaluation.

## Output Management

All generated files must be saved under `outputs/`.

Recommended layout:

```text
outputs/
|-- runs/
|   `-- <experiment_name>/<client_id>/
|-- adapters/
|   `-- <experiment_name>/<client_id>/
|-- similarity/
|   `-- <experiment_name>/
`-- logs/
```

Every run should save:

- CLI/config arguments,
- random seed values,
- dataset/task identifiers,
- model checkpoint name,
- adapter type and adapter hyperparameters,
- target module list,
- training metrics,
- paths to saved adapter weights.

## CLI Expectations

Experiment scripts should use clear `argparse` or equivalent CLI arguments.

Common arguments should include:

- `--experiment-name`
- `--model-name-or-path`
- `--dataset-name`
- `--task-name` or `--client-task`
- `--client-id`
- `--adapter-type` with values such as `lora` and `svd_lora`
- `--target-modules`
- `--rank`
- `--alpha`
- `--learning-rate`
- `--num-train-epochs`
- `--per-device-train-batch-size`
- `--seed`
- `--output-dir`, defaulting to a path under `outputs/`

Similarity scripts should include:

- `--experiment-name`
- `--adapter-type`
- `--adapter-dirs` or `--runs-dir`
- `--similarity-method`
- `--output-dir`, defaulting to `outputs/similarity/<experiment_name>/`

## Reproducibility

Set fixed seeds in Python, NumPy, PyTorch, Hugging Face, and dataset sampling code where applicable.

Prefer a shared utility such as `src/utils/seed.py` for:

- `random.seed(seed)`,
- `numpy.random.seed(seed)`,
- `torch.manual_seed(seed)`,
- `torch.cuda.manual_seed_all(seed)`,
- Hugging Face seed helpers when used.

If enabling deterministic PyTorch behavior, document any performance or operator limitations.

## Documentation Requirements

When adding scripts or changing workflows, update `README.md` with:

- installation instructions,
- training examples for standard LoRA,
- training examples for SVD-LoRA,
- similarity evaluation examples,
- explanation of where outputs are written,
- any expected dataset preprocessing steps.

README examples should be directly runnable or clearly marked as templates.

## Testing and Validation

Add focused tests for reusable logic when possible, especially:

- SVD-LoRA linear wrapper forward-pass shape compatibility,
- deterministic adapter initialization,
- adapter matrix extraction,
- similarity computation,
- output path handling.

For experiment scripts, at minimum verify CLI parsing and a small smoke run when practical.

## Things Not To Do Yet

- Do not implement federated aggregation.
- Do not average, merge, or aggregate client adapters.
- Do not let clients continue from each other's checkpoints.
- Do not write outputs outside `outputs/`.
- Do not make similarity evaluation depend on training-only modules.
- Do not rely on implicit global state for experiment configuration.

