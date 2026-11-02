# Reproducibility and visual sources

## Scope

This guide describes what is available in the committed project and what is required to reproduce it. The documentation update inspected source and saved outputs; it did not rerun model training or hardware/simulation experiments.

## Required inputs

The recordings and trained weights are not committed. The memory-efficient loader handles Sleep-EDF-style `sleep-cassette` and `sleep-telemetry` naming and expects matching hypnograms. Check annotation/channel availability. Cached preprocessing goes under `preprocessed_data/`; checkpoints use `autoencoder.pt` and `classifier.pt`.

## Entry point

Create a Python environment and install the imported dependencies, selecting PyTorch for your CPU/CUDA platform:

```bash
python -m pip install numpy pandas mne torch scikit-learn scipy matplotlib tqdm
```

Arrange paired PSG/hypnogram EDF recordings under `dataset/` and, if used, a `dataset/RECORDS` file. Review the configuration inside `main()` for data limits, batch size, training epochs, and output paths.

```bash
python main_memory_efficient.py
```

No command-line argument parser is implemented; change configuration values in the script.

## Evidence and interpretation

The source computes metrics and writes a confusion matrix, but no completed evaluation results are committed. The implementation uses an epoch-level train/validation split; subject-independent generalization is not established. Cache memory savings apply mainly to the signal loader, while the later latent encodings are materialized in memory.

## Visual provenance

All screenshots and experiment figures are tied to repository material. Workflow SVGs are explanatory illustrations. The full image inventory is in [assets/sources.json](assets/sources.json).

| Visual | Source |
| --- | --- |
| [overview.svg](assets/overview.svg) | New explanatory project-workflow diagram based on the committed source and documentation; not a screenshot or measured result. |

## Validation

```bash
python tools/check_docs.py
```

The standard-library checker validates README/docs local link destinations and SVG/PNG/JPEG headers. It makes no external requests and installs no project runtime dependencies. External web links, model accuracy, image semantics, and full application execution need separate review.

## Reproduction record

For a new experiment or demo, record the source commit, environment versions, data provenance and split, configuration, command/notebook order, and resulting artifacts. Only label a measurement as reproduced after completing that run. Keep credentials and restricted datasets outside version control.
