# AV-DnR

This package contains the AVDnR generation workflow.

Supported public entrypoints:

- `bin/build_source_manifest.py`
- `bin/make_split_manifests.py`
- `bin/validate_split_manifests.py`
- `bin/generate_dataset.py`

The intended workflow is manifest-driven:

1. Enumerate source clips from the upstream datasets.
2. Deterministically partition source IDs into train and test.
3. Validate zero source-ID overlap across splits.
4. Generate mixtures from those explicit manifests.

`build_DnRv3_fixed.py` is included as the historical mixing implementation used by the manifest wrapper. The supported public interface is still `bin/generate_dataset.py`, not direct invocation of the legacy generator.
