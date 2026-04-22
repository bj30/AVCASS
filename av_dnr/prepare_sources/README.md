# Data preprocessing for FMA and VGGSound
Preparation code is based on [TVSM-dataset](https://github.com/biboamy/TVSM-dataset/).
We use SMAD to filter speech from music (FMA dataset), and filter speech & music from sfx (VGGSound).



## Step 0. Download SMAD model
Download the `TVSM-pseudo` directory from [Google Drive](https://drive.google.com/drive/folders/1THtEHYUh1lueUFH37n2VAhVy8n2QfNpp?usp=sharing) and put it under `models/`.


## Step 1. Prepare Music
### 1-1. Download FMA medium set
Download the medium set of FMA dataset from [here](https://os.unil.cloud.switch.ch/fma/fma_medium.zip).
Convert the audios to 16kHz & mono, and place at `FMA_ROOT`.

`prepare_music.py` expects the usual FMA-medium directory structure with MP3 files under numbered subdirectories, for example:

```text
FMA_ROOT/
  000/
    000002.mp3
    000005.mp3
  001/
    001486.mp3
  ...
```

### 1-2. Filter speech from FMA

Running this command will save audios and csv files under `FMA_PREPROCESSED_ROOT`.
```
python prepare_music.py --music_dir ${FMA_ROOT} --output_dir ${FMA_PREPROCESSED_ROOT}
```

The output layout is:

```text
FMA_PREPROCESSED_ROOT/
  audio_16k/
    000/
      000002_ch1_chunk1.wav
      000002_ch2_chunk1.wav
    001/
      ...
  csv_16k/
    000/
      000002_ch1.csv
      000002_ch2.csv
    001/
      ...
```

For manifest generation, point `music.roots` in `av_dnr/configs/source_roots.json` at `FMA_PREPROCESSED_ROOT/audio_16k`.

## Step 2. Prepare SFX
### 2-1. Download VGGSound dataset
We expect the VGGSound directory to be:

```
VGGSOUND_ROOT/
    L audios/
        L 5bHACxteut8_000242.wav
        L 5kPVERHWpVI_000023.wav
        L ....
    L frames/
        L 5bHACxteut8_000242/
            L 000.jpg
            L 001.jpg
            ....
        L ...
    L metadata/
        L train.csv
        L test.csv
```

### 2-2. (Optional) Filter videos based on categories 
We manually deleted speech and music related categories from VGGSound metadata: `vgg_stat_filtered.csv`.
Using these filtered categories, prepare the total list of videos to your `VGGSOUND_PREPROCESSED_ROOT`.

```
python write_filtered_csvs.py \
  --metadata_dir ${VGGSOUND_ROOT}/metadata \
  --output_dir ${VGGSOUND_PREPROCESSED_ROOT} \
  --filtered_data vgg_stat_filtered.csv
```

Output csv will look like this:
```
LDoXsip0BEQ_000177.mp4,parrot talking
cRlp5v9BHeE_000011.mp4,car passing by
...
```

You can skip this step by directly using csv files at `vggsound_filtered` for the next step.

### 2-3. Filter speech and music from VGGSound
If you manually generated csv files with Step 2-2., set the `--filtered_category_csv_dir` to your `VGGSOUND_PREPROCESSED_ROOT`.

```
python prepare_vggsound.py \
  --audios_dir ${VGGSOUND_ROOT}/audios \
  --frames_dir ${VGGSOUND_ROOT}/frames \
  --filtered_category_csv_dir ${VGGSOUND_PREPROCESSED_ROOT} \
  --output_dir ${VGGSOUND_PREPROCESSED_ROOT}
```

If you skip Step 2-2, omit `--filtered_category_csv_dir` and the script will use the bundled `vggsound_filtered/{train,test}.csv`.

The output layout is:

```text
VGGSOUND_PREPROCESSED_ROOT/
  audio/
    train/
      LDoXsip0BEQ_000177_chunk1.wav
      ...
    test/
      ...
  csv/
    train/
      LDoXsip0BEQ_000177.csv
      ...
    test/
      ...
  video_json/
    train/
      LDoXsip0BEQ_000177.json
      ...
    test/
      ...
```

For manifest generation, point `sfx.roots` in `av_dnr/configs/source_roots.json` at `VGGSOUND_PREPROCESSED_ROOT/audio`.

`video_json/` is not used by manifest generation, but it is needed later by the AV data loader to map each retained SFX chunk back to frame paths.


## 3. Set the filtered source paths for mixing pipeline
Following `av_dnr/configs/source_roots.example.json`, set the filtered source paths for FMA and VGGSound.
