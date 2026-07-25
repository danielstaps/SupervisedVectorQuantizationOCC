# prototorch_oneclass — Supervised Vector Quantization One-Class Classifier (SVQ-OCC)

Reference / maintained implementation of the SVQ-OCC model from:

> D. Staps, R. Schubert, M. Kaden, A. Lampe, W. Hermann, T. Villmann,
> **"Prototype-based one-class-classification learning using local representations"**,
> *2022 International Joint Conference on Neural Networks (IJCNN)*, IEEE, 2022, pp. 1–8.
> DOI: [10.1109/IJCNN55064.2022.9892912](https://doi.org/10.1109/IJCNN55064.2022.9892912)

## What this is

**SVQ-OCC** is a prototype-based **one-class classifier** for outlier / novelty detection built in
the spirit of Generalized Learning Vector Quantization (GLVQ). Each prototype acts as the center of
a flexible enclosing ball; the centers are learned from the data density and the per-ball radius is
adapted via the local underlying distribution, optimized end-to-end by stochastic gradient descent.
The "local representations" give an interpretable decision boundary (a point is an outlier if it
falls outside every ball).

Implemented as the Python package **`prototorch_oneclass`**, built on
[`prototorch`](https://github.com/si-cim/prototorch).

## Models & options

- Models: `OneClassGLVQ`, `OneClassGMLVQ`, `SVQ_OCC` (Euclidean and kernelized variants).
- Classification costs: `brier`, `pcs`, `ce`. Representation cost: neural-gas (`NG`).
- Key parameter: `K` (number of prototypes/balls).

## Install & run

```bash
pip install -e .        # or: pip install -r requirements.txt
python3 examples/...    # see examples/ for runnable scripts
```

## Branches (see repo-cleanup convention)

| Branch / tag | Meaning |
|---|---|
| `published` / `v-paper-2022b` | **frozen** as-published state (IJCNN 2022 submission-deadline commit, grafted from the original `si-cim/SVQ-OCC` history) |
| `published-refactored` | paper code **cleaned only**, function verified identical to `published` |
| `main` | maintained best state (post-paper fixes: GPU support, bug fixes) |

This repository is the **canonical, self-contained home** of `prototorch_oneclass`; it no longer
depends on private si-cim research repositories. It is also the package used as the SVQ-OCC baseline
by the K-MEB work (companion experiments repo, kept private).

## How to cite

See [`CITATION.cff`](CITATION.cff) and cite the paper above.

## Acknowledgment

By **Daniel Staps** ([0009-0002-4459-4544](https://orcid.org/0009-0002-4459-4544)) with R. Schubert,
M. Kaden, A. Lampe, W. Hermann and **Thomas Villmann**
([0000-0001-6725-0141](https://orcid.org/0000-0001-6725-0141)). Repository cleanup with assistance
from Claude (Anthropic).

## License

MIT — see [LICENSE](LICENSE).
