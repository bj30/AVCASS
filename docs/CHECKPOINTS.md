# Checkpoints

This release package does not bundle model weights.

## AV-CASS checkpoints

You should supply these paths at runtime:

- `CKPT`: the AV-CASS model checkpoint to train from or evaluate.
- `INIT_CKPT`: the stage-1 audio-only checkpoint used to initialize stage-2 AV training.

## Visual backbone checkpoints

AV workflows also require external visual encoder checkpoints:

- `CAVP_CKPT`: required for AV inference and for AV training when `visual_encoder_type` uses CAVP.
- `TALKNET_CKPT`: required for stage-2 AV training with `visual_encoder_type=both` or `talknet`.

## Metric checkpoints

The evaluation code uses `audioldm_eval`, which downloads its own metric checkpoints on first use unless they are already cached.

## Metadata files

The JSON files under `av_cass/checkpoints/` describe the intended released AO and AV model families. They are metadata only and should be updated once public checkpoint URLs are available.
