# Data audit — 2026-10-16

LV cm: author main.py evaluation comment; feature meanings unknown unless separately documented

See data_audit.json for every array checksum, shape, dtype, distribution and archive member.

Exact repeated varying-geometry groups: 0. Fixed-geometry splits all share one geometry.

| Array | Shape | dtype | nonfinite |
|---|---|---|---|
| cardiac-lv-surrogate/data/raw/lvData/rawData/test/global-features.npy | [567, 4] | float64 | 0 |
| cardiac-lv-surrogate/data/raw/lvData/rawData/test/real-node-coords.npy | [567, 6784, 3] | float64 | 0 |
| cardiac-lv-surrogate/data/raw/lvData/rawData/test/real-node-displacement.npy | [567, 6784, 3] | float64 | 0 |
| cardiac-lv-surrogate/data/raw/lvData/rawData/test/real-node-features.npy | [567, 6784, 10] | float64 | 0 |
| cardiac-lv-surrogate/data/raw/lvData/rawData/test/shape-coeffs.npy | [567, 40] | float64 | 0 |
| cardiac-lv-surrogate/data/raw/lvData/rawData/test/vol.npy | [567, 3] | float64 | 0 |
| cardiac-lv-surrogate/data/raw/lvData/rawData/train/global-features.npy | [2250, 4] | float64 | 0 |
| cardiac-lv-surrogate/data/raw/lvData/rawData/train/real-node-coords.npy | [2250, 6784, 3] | float64 | 0 |
| cardiac-lv-surrogate/data/raw/lvData/rawData/train/real-node-displacement.npy | [2250, 6784, 3] | float64 | 0 |
| cardiac-lv-surrogate/data/raw/lvData/rawData/train/real-node-features.npy | [2250, 6784, 10] | float64 | 0 |
| cardiac-lv-surrogate/data/raw/lvData/rawData/train/shape-coeffs.npy | [2250, 40] | float64 | 0 |
| cardiac-lv-surrogate/data/raw/lvData/rawData/train/vol.npy | [2250, 3] | float64 | 0 |
| cardiac-lv-surrogate/data/raw/lvData/rawData/validation/global-features.npy | [150, 4] | float64 | 0 |
| cardiac-lv-surrogate/data/raw/lvData/rawData/validation/real-node-coords.npy | [150, 6784, 3] | float64 | 0 |
| cardiac-lv-surrogate/data/raw/lvData/rawData/validation/real-node-displacement.npy | [150, 6784, 3] | float64 | 0 |
| cardiac-lv-surrogate/data/raw/lvData/rawData/validation/real-node-features.npy | [150, 6784, 10] | float64 | 0 |
| cardiac-lv-surrogate/data/raw/lvData/rawData/validation/shape-coeffs.npy | [150, 40] | float64 | 0 |
| cardiac-lv-surrogate/data/raw/lvData/rawData/validation/vol.npy | [150, 3] | float64 | 0 |
| cardiac-lv-surrogate/data/raw/lvData/topologyData/real-node-topology.npy | [37056, 2] | int32 | 0 |
| passive-lv-gnn-emul/data/lvDataFixedGeom/rawData/test/augmented-node-coords.npy | [1, 7751, 3] | float32 | 0 |
| passive-lv-gnn-emul/data/lvDataFixedGeom/rawData/test/augmented-node-features.npy | [1, 7751, 10] | float32 | 0 |
| passive-lv-gnn-emul/data/lvDataFixedGeom/rawData/test/edge-features.npy | [1, 29085, 4] | float32 | 0 |
| passive-lv-gnn-emul/data/lvDataFixedGeom/rawData/test/global-features.npy | [246, 4] | float64 | 0 |
| passive-lv-gnn-emul/data/lvDataFixedGeom/rawData/test/real-node-coords.npy | [1, 6784, 3] | float64 | 0 |
| passive-lv-gnn-emul/data/lvDataFixedGeom/rawData/test/real-node-displacement.npy | [246, 6784, 3] | float64 | 0 |
| passive-lv-gnn-emul/data/lvDataFixedGeom/rawData/test/real-node-features.npy | [1, 6784, 10] | float64 | 0 |
| passive-lv-gnn-emul/data/lvDataFixedGeom/rawData/test/shape-coeffs.npy | [1, 40] | float64 | 0 |
| passive-lv-gnn-emul/data/lvDataFixedGeom/rawData/test/vol.npy | [246, 3] | float64 | 0 |
| passive-lv-gnn-emul/data/lvDataFixedGeom/rawData/train/augmented-node-coords.npy | [1, 7751, 3] | float32 | 0 |
| passive-lv-gnn-emul/data/lvDataFixedGeom/rawData/train/augmented-node-features.npy | [1, 7751, 10] | float32 | 0 |
| passive-lv-gnn-emul/data/lvDataFixedGeom/rawData/train/edge-features.npy | [1, 29085, 4] | float32 | 0 |
| passive-lv-gnn-emul/data/lvDataFixedGeom/rawData/train/global-features.npy | [85, 4] | float64 | 0 |
| passive-lv-gnn-emul/data/lvDataFixedGeom/rawData/train/real-node-coords.npy | [1, 6784, 3] | float64 | 0 |
| passive-lv-gnn-emul/data/lvDataFixedGeom/rawData/train/real-node-displacement.npy | [85, 6784, 3] | float64 | 0 |
| passive-lv-gnn-emul/data/lvDataFixedGeom/rawData/train/real-node-features.npy | [1, 6784, 10] | float64 | 0 |
| passive-lv-gnn-emul/data/lvDataFixedGeom/rawData/train/shape-coeffs.npy | [1, 40] | float64 | 0 |
| passive-lv-gnn-emul/data/lvDataFixedGeom/rawData/train/vol.npy | [85, 3] | float64 | 0 |
| passive-lv-gnn-emul/data/lvDataFixedGeom/rawData/validation/augmented-node-coords.npy | [1, 7751, 3] | float32 | 0 |
| passive-lv-gnn-emul/data/lvDataFixedGeom/rawData/validation/augmented-node-features.npy | [1, 7751, 10] | float32 | 0 |
| passive-lv-gnn-emul/data/lvDataFixedGeom/rawData/validation/edge-features.npy | [1, 29085, 4] | float32 | 0 |
| passive-lv-gnn-emul/data/lvDataFixedGeom/rawData/validation/global-features.npy | [15, 4] | float64 | 0 |
| passive-lv-gnn-emul/data/lvDataFixedGeom/rawData/validation/real-node-coords.npy | [1, 6784, 3] | float64 | 0 |
| passive-lv-gnn-emul/data/lvDataFixedGeom/rawData/validation/real-node-displacement.npy | [15, 6784, 3] | float64 | 0 |
| passive-lv-gnn-emul/data/lvDataFixedGeom/rawData/validation/real-node-features.npy | [1, 6784, 10] | float64 | 0 |
| passive-lv-gnn-emul/data/lvDataFixedGeom/rawData/validation/shape-coeffs.npy | [1, 40] | float64 | 0 |
| passive-lv-gnn-emul/data/lvDataFixedGeom/rawData/validation/vol.npy | [15, 3] | float64 | 0 |
| passive-lv-gnn-emul/data/lvDataFixedGeom/topologyData/augmented-topology.npy | [58170, 2] | int64 | 0 |
| passive-lv-gnn-emul/data/lvDataFixedGeom/topologyData/real-node-topology.npy | [37056, 2] | int32 | 0 |
| passive-lv-gnn-emul/data/lvDataFixedGeom/topologyData/sparse-topology.npy | [29085, 2] | int64 | 0 |

Unknown: per-column global/node feature physical definitions, complete Dalton constitutive law and parameter mapping, pressure/traction and boundary labels, volumetric connectivity, original solver reproduction and timings

Track A: displacement emulation only. Track B: independent dynamic mechanics benchmark, solver unavailable in current environment; do not conflate.