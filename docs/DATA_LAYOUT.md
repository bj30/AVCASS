# Data Layout

## Corrected AVDnR layout

The public wrappers expect the corrected dataset root to look like this:

```text
AVDnR-corrected/
  file_list_train.txt
  file_list_test.txt
  train/
    parts_0/
      0/
        mixture.wav
        speech.wav
        sfx.wav
        music.wav
        annots.json
  test/
    parts_0/
      0/
        mixture.wav
        speech.wav
        sfx.wav
        music.wav
        annots.json
```

The distributed inference code also works when `DATASET_ROOT` points at the dataset root and the `test/` split lives underneath it.

## Source-catalog config

`av_dnr/configs/source_roots.example.json` is only a template. Replace every placeholder path with your local dataset roots before generating manifests.
