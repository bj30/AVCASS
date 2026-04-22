import os
import json
import argparse
import soundfile as sf
from glob import glob

import numpy as np
import librosa
import tqdm
import torch
import torchaudio
import warnings

from itertools import repeat
from scipy import stats
import soundfile as sf
import main.misc.audio_utils as audio_utils
import pyloudnorm as pyln
import subprocess
import shlex
from ffmpeg_normalize import FFmpegNormalize, MediaFile


def lufs_adjust(data, input_loudness, target_loudness):
    """ Loudness normalize a signal.
    
    Normalize an input signal to a user loudness in dB LKFS.   

    Params
    -------
    data : ndarray
        Input multichannel audio data.
    input_loudness : float
        Loudness of the input in dB LUFS. 
    target_loudness : float
        Target loudness of the output in dB LUFS.
        
    Returns
    -------
    output : ndarray
        Loudness normalized output data.
    """    
    # calculate the gain needed to scale to the desired loudness level
    delta_loudness = target_loudness - input_loudness
    gain = np.power(10.0, delta_loudness/20.0)

    output = gain * data

    # check for potentially clipped samples
    if np.max(np.abs(output)) >= 1.0:
        warnings.warn("Possible clipped samples in output.")

    return output

def aug_gain(audio, low=0.0, high=1.0):
    gain = np.random.uniform(low=low, high=high)
    return audio * gain

def gen_poisson(mu):
    """Adapted from https://github.com/rmalouf/learning/blob/master/zt.py"""
    r = np.random.uniform(low=stats.poisson.pmf(0, mu))
    return int(stats.poisson.ppf(r, mu))

def gen_norm(mu, sig):
    return np.random.normal(loc=mu, scale=sig)

def gen_skewnorm(skew, mu, sig):
    # negative a means skewed left while positive means skewed right; a=0 -> normal
    return stats.skewnorm(a=skew, loc=mu, scale=sig).rvs()

def get_some_noise(shape):
    return np.random.randn(shape).astype(np.float32)

class MixtureGeneratorDnRv3():
    def __init__(self, 
        mixture_lengeth=60.0, 
        sampling_rate=16000, 
        partition='test', 
        peak_norm_db=-2.0,
        source_roots=None,
        # wavfiles=None,
        without_replacement=False,
        rank=0,
        enable_background_sfx=True,
        ):

        self.seq_dur = mixture_lengeth # sec
        self.sr = sampling_rate
        self.rank = rank
        self.background = enable_background_sfx
        print("Initialized generator at Rank: ", rank)

        
        wavfiles = {}
        if source_roots is None:
            raise ValueError(
                "source_roots must be provided (e.g. loaded from configs/source_roots.json). "
                "This script no longer embeds machine-specific absolute paths."
            )
        wavfiles['music'] = _load_sources_for_stem(
            source_roots=source_roots,
            stem='music',
            partition=partition,
        )
        wavfiles['speech'] = _load_sources_for_stem(
            source_roots=source_roots,
            stem='speech',
            partition=partition,
        )
        wavfiles['sfx'] = _load_sources_for_stem(
            source_roots=source_roots,
            stem='sfx',
            partition=partition,
        )
        if self.background:
            wavfiles['background'] = _load_sources_for_stem(
                source_roots=source_roots,
                stem='background',
                partition=partition,
            )
        self.files = wavfiles
        # wavfiles is a dictionary containing the file paths, audio duration, length, sr, channels, loudness of each audio file in each class
        # e.g. wavfiles['music'] = [{'file':file_path, 'duration':duration, 'length':length, 'sr':sr, 'channels':channels, 'loudness':loudness}, ...]
        
        # self.peak_norm = audio_utils.db_to_gain(peak_norm_db)
        self.true_peak_db = peak_norm_db
        self.partition = partition
        self.without_replacement = without_replacement
        
        self.source_usage_tracker = {'music':0, 'speech':0, 'sfx':0}

        # track loudness
        self.mu_ref = -27
        self.mu_mix = -27
        self.sigma_mix = 1
        # track loudness delta
        self.delta_mu_track = {
            'music'     : -5,
            'sfx'       : -5,
            'speech'    : 0,
            'background': -13,
            }
        # track loudness sigma
        self.sigma_track = {
            'music'     : 6,
            'sfx'       : 6,
            'speech'    : 4,
            'background': 6,
            }
        # event loudness sigma
        self.sigma_event = {
            'music'     : 10,
            'sfx'       : 10,
            'speech'    : 6,
            'background': 10,
            }
        
        self.lamdb_event = {
            'music'     : 7,
            'sfx'       : 12 if partition=='train' else 24, #100, # <- to get SFX for every frame # 12,
            'speech'    : 12 if partition=='train' else 24,
            'background': 24,
            }
        self.overlap_gamma = {
            'music'     : 1.0,
            'sfx'       : 0.5 if partition=='train' else 1.0,
            'speech'    : 0.75 if partition=='train' else 1.0,
            'background': 0.0
            }
        # beta_min is the minimum proportional duration
        self.beta_min = {
            'music'     : 0.3,
            'sfx'       : 0.3,
            'speech'    : 1.0,
            'background': 0.3,
            }
        self.T_min = {
            'music'     : 0,
            'sfx'       : 0.5,
            'speech'    : 0,
            'background': 1.0,
            }
        
    
    def _set_submix(self, c, meta_info=None):
        
        # initialize buffer to zeros of the track length
        track_length = int(self.seq_dur * self.sr)
        submix = np.zeros(track_length, dtype=np.float32)
        t_cur = 0
        
        # sample track loudness
        track_loudness = np.random.normal(self.mu_ref + self.delta_mu_track[c], self.sigma_track[c])
        
        # sample number of events
        num_seg = gen_poisson(self.lamdb_event[c])
        put_seg = 0
        seg_info_dict = []
        
        for i in range(num_seg):
            # if c=='speech':
            #     print(t_cur)
            # up to 10 attempts to get a valid audio file
            for trial in range(10):
                event_start = None
                event_duration = None
                # random select a audio file in the class and load the audio
                audio_data, file_path = self.get_one_source_sample(c) # should give the audio data as numpy array
                
                ##############################################################################################
                # Check and get the timing parameters for the audio
                ##############################################################################################
                audio_length = len(audio_data)
                
                # absolute minimum duration
                T_min = int(self.T_min[c] * self.sr) # minimum duration for a sample
                S_min = max(T_min, self.beta_min[c] * audio_length) # minimum duration for a audio segment
                               
                # the remaining time in the track should be at least S_min
                if track_length - t_cur < S_min:
                    continue
                
                t_max = track_length - S_min # maximum start time for the audio segment
                if self.beta_min[c] < 1.0:
                    t_max = min(t_max, track_length - 2 * self.sr) # 2 sec margin same as in DnRv2, 
                
                    if t_max < t_cur:
                        continue
                
                # sample event start time in track
                # if c != 'sfx':
                event_start = gen_skewnorm(skew = 5, mu = t_cur / self.sr, sig = 2)
                # else:
                #     # to ensure that one frame is present (in load_fps=4)
                #     event_start = t_cur / self.sr + random.uniform(0, 0.25)
                # if c=='speech':
                #     print(t_cur, event_start*self.sr, t_max)
                event_start = int(event_start * self.sr)
                
                # for speech (beta_min is 1.0), or if the audio is shorter than S_min, we take the whole audio
                if self.beta_min[c] == 1.0 or audio_length <= S_min:
                    # clamp event_start to [t_cur, t_max]
                    # event_start =  int(min(max(event_start, t_cur), t_max))
                    event_start = max(event_start, t_cur)
                    if t_max < event_start:
                        continue
                    event_duration = audio_length
                    # if c=='speech':
                    #     print(event_start)
                else:
                    S_max = min(audio_length, track_length - event_start) # maximum duration for the audio segment
                    if S_max < 0:
                        continue
                    event_start = max(event_start, 0)
                    # truncated normal distribution
                    
                    v_dur = 0.5
                    rho_dur = 0.1
                    event_duration = int(max(min(gen_norm(mu=v_dur*audio_length, sig=rho_dur*audio_length), S_max), S_min))
                
                if c == 'music':
                    # for music the event_start time offset is a uniform distribution between 0 and audio length - event duration
                    audio_clip_start_time = np.random.randint(0, audio_length - event_duration)
                else:
                    audio_clip_start_time = 0
                    
                if c=='speech':
                    assert event_duration == audio_length
                    assert audio_clip_start_time == 0
                    
                # final check 
                if event_start + event_duration > track_length:
                    event_start = None
                    event_duration = None
                    continue
                ##############################################################################################
                
                # if successfully get the event_start and event duration, break the loop of sampling valid audio
                break
            
            if event_start is None or event_duration is None:
                continue
            
            # print(f"c: {c}", "Event Start: ", event_start, "Event Duration: ", event_duration, "Audio Length: ", audio_length, "Track Length: ", track_length, "T_cur: ", t_cur, "T_max: ", t_max)
            
            # sample event loudness
            event_loudness = np.random.normal(track_loudness, self.sigma_event[c])
            audio_clip = audio_data[audio_clip_start_time : audio_clip_start_time + event_duration]
            # normalize the audio clip to the target loudness
            audio_clip_norm, gain = audio_utils.lufs_norm(data=audio_clip, sr=self.sr, norm=event_loudness)
            
            assert event_duration == len(audio_clip_norm)
            assert track_length >= event_start + event_duration, f"Track Length: {track_length}, Event Start: {event_start}, Event Duration: {event_duration}, Residual: {track_length - event_start - event_duration}"
            
            clip_info = {
                "file_path": file_path, # the file path of the audio
                "event_start": event_start, # the start time of the audio in the submix
                "event_duration": event_duration, # the length of the audio in the submix
                "audio_clip_start_time": audio_clip_start_time, # the start time of the sampled raw audio in the audio file
                }
            seg_info_dict.append(clip_info)
            
            # set submix
            submix[event_start : event_start + event_duration] += audio_clip_norm
            # sample delta t_cur
            if self.overlap_gamma[c] == 1.0:
                # for music, the delta_t_cur is the event duration
                delta_t_cur = event_duration
            else:
                delta_t_cur = np.random.randint(self.overlap_gamma[c] * event_duration, event_duration)
            # t_cur += delta_t_cur
            t_cur = event_start + delta_t_cur
            put_seg += 1
            
        # normalize the submix to the target loudness
        submix, gain = audio_utils.lufs_norm(data=submix, sr=self.sr, norm=track_loudness)

        meta_info[c] = {
            "track_loudness": track_loudness,
            "num_seg": num_seg,
            "put_seg": put_seg,
            "seg_info_dict": seg_info_dict,
            "gain": float(gain),
        }
        
        return submix, meta_info
    
    def get_one_source_sample(self, c):
        """get one source sample from the source pool
        """
        last_error = None
        while self.files[c]:
            idx = np.random.randint(0, len(self.files[c]))
            f = self.files[c][idx]['files']
            try:
                with sf.SoundFile(f) as audio:
                    audio_data = audio.read()
                    src_sr = audio.samplerate
                # make sure the audio sample rate is the same as the target sample rate
                if src_sr != self.sr:
                    audio_data = torchaudio.transforms.Resample(src_sr, self.sr)(torch.tensor(audio_data).unsqueeze(0)).numpy()
                return audio_data, f
            except Exception as exc:
                last_error = exc
                print(f"[build_DnRv3_fixed] skipping unreadable source for {c}: {f} ({exc})")
                self.files[c].pop(idx)
        raise RuntimeError(f"No readable source files left for stem '{c}'. Last error: {last_error}")
        # return f

    def __call__(self,):

        track_list = {}
        
        # the annots is the information of the each submix. 
        # It contains the file path, the gain, the start and end time of the audio in the submix
        # The annots is used to save the information of the audio in the submix
        # annots['sfx'] = [{'file':file_path, 'gain':gain, 'start':start_time, 'end':end_time, 'clip_length':clip_length}, ...]
        annots = {
            'sfx': {},
            'music': {},
            'speech': {},
        }
        if self.background:
            annots['background'] = {}

        for c in annots.keys():
            submix, annots = self._set_submix(c=c, meta_info=annots)
            track_list[c] = submix
            
        # mix the submixes
        output = self.track_list_to_mixture(track_list)
        if output is None:
            return output

        mixture, track_list, track_gain = output

        # NOTE: This might be unnecessary
        # measure the final loudness of the mixture
        # meter = pyln.Meter(self.sr)
        # track_loudness = meter.integrated_loudness(mixture)
        # track_gain = np.power(10.0, (self.mu_mix - track_loudness)/20.0)
        # print(f"Final Mixture Loudness: {track_loudness}")
        
        annots['final_gain'] = track_gain
        
        track_list['mixture'] = mixture
        
        # NOTE: manually add the background to sfx, need to change this in the future
        if self.background:
            track_list['sfx'] += track_list['background']
            # and remove the track_list['background']
            del track_list['background']
        
        return track_list, mixture, annots


    def track_list_to_mixture(self, track_list):
        """
        MASTERING
        1. compute intermediate mixture of stems
        2. using the intermediate mixture, compute the necessary gain adjustment
           to get the mixture to the target loudness
        3. linearly scale each stem using the gain adjustment values using pyloudnorm
        4. using loudnorm filter of ffmpeg in double-pass manner,
          [1] compute filter parameters
          [2] jointly apply EBU R 128 loudness normalization & peak limiting to -2.0 dBFS
        5. mix the stems to obtain mixture
        
        1. - 3. : done in self.set_target_loudness()
        4. : done in self.renormalize()
        5. : done in here
        """
        rank = self.rank
        
        # Compute master loundess
        master_lufs = np.random.normal(self.mu_mix, self.sigma_mix)

        ##### NEW CODE #####
        output = self.set_target_loudness(track_list, master_lufs)
        if output is None:
            return None
        loudness_dict, target_loudness_dict, normalized_sources, gain, mixture = output
        
        
        # check the target loudness, should be in the range [-70, -5]
        for key, target_loudness in target_loudness_dict.items():
            if target_loudness < -70 or target_loudness > -5:
                print(f"Target Loudness: {target_loudness}, Key: {key}")
                return None
            
        
        os.makedirs('./temp_dnr', exist_ok=True)
        sf.write(f'./temp_dnr/mixture_woffmpeg_{rank}.wav', mixture, self.sr)
        
        # Using FFmpeg to normalize the audio
        
        mixture = np.zeros(int(self.seq_dur*self.sr), dtype=np.float32)
        final_sources = {}
        for key, clip in normalized_sources.items():
        
            save_key = f'./temp_dnr/{key}'
            sf.write(f'{save_key}_{rank}.wav', clip, self.sr)
            ffmpeg_normalizer = FFmpegNormalize(
                normalization_type = 'ebu',
                target_level = target_loudness_dict[key],
                print_stats = True,
                keep_loudness_range_target = True, # Keep the input loudness range target to allow for linear normalization.
            )
            ffmpeg_file = MediaFile(ffmpeg_normalizer, f'{save_key}_{rank}.wav', f'{save_key}_{rank}_out.wav')
            ffmpeg_file.run_normalization()
            with sf.SoundFile(f'{save_key}_{rank}_out.wav', 'r') as f:
                # check the sample rate
                if f.samplerate != self.sr:
                    # resample the audio
                    final_src = librosa.resample(f.read(), orig_sr=f.samplerate, target_sr=self.sr)
                else:
                    final_src = f.read()
            final_sources[key] = final_src
            
            mixture += final_src
        
        sf.write(f'./temp_dnr/mixture_final_{rank}.wav', mixture, self.sr)
        
        return mixture, final_sources, gain    
    
    def set_target_loudness(self, sources_dict, mixture_target_loudness):
        # MAX_AMP=0.9

        # Initialize loudness
        loudness_dict = {}
        # (1) compute intermediate mixture first
        meter = pyln.Meter(self.sr)
        mixture = np.zeros_like(sources_dict['speech'])
        for submix_key, submix_clip in sources_dict.items():
            loudness = meter.integrated_loudness(submix_clip)
            if np.isinf(loudness):
                return None
            loudness_dict[submix_key] = loudness
            mixture += submix_clip

        # (2) compute the gain adjustment for each stem
        # to fit the target loudness of the mixture
        mixture_loudness = meter.integrated_loudness(mixture)
        if np.isinf(mixture_loudness):
            return None
        
        delta_loudness = mixture_target_loudness - mixture_loudness
        gain = np.power(10.0, delta_loudness/20.0)
        normalized_sources = {}
        target_loudness_dict = {}
        mixture = np.zeros_like(sources_dict['speech'])
        for submix_key, submix_clip in sources_dict.items():
            # 3) linearly scale each stem (same computation with pyln.normalize.loudness)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                src = gain * submix_clip
                target_loudness = loudness_dict[submix_key] + delta_loudness
                
            # Save scaled source and loudness.
            normalized_sources[submix_key] = src
            target_loudness_dict[submix_key] = target_loudness
            mixture += src
            
        return loudness_dict, target_loudness_dict, normalized_sources, gain, mixture


def _get_filepaths(dir, c='', return_dict=True):
    files = glob(dir+'/**/*.wav', recursive=True)
    if return_dict:
        output = []
        for file in files:
            output.append({'files':file})
        return output
    else:
        return files

def _get_filepaths_from_csv(dir, csv_path):
    path_list = []
    with open(csv_path) as f:
        csv_data = csv.reader(f)
        for row in csv_data:
            path_list.append(os.path.join(dir, row[0]).replace('.mp4', '.wav'))
    return path_list


def _read_source_roots_config(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
    return {
        stem: spec.get("roots", [])
        for stem, spec in config.get("sources", {}).items()
    }


def _resolve_partition_root(root, partition):
    root = root.replace("{split}", partition)
    split_dir = os.path.join(root, partition)
    if os.path.isdir(split_dir):
        return split_dir
    return root


def _speech_root_matches_partition(root, partition):
    root_norm = root.replace("\\", "/").lower()
    if partition == "train":
        return "/test" not in root_norm
    return "/test" in root_norm


def _load_sources_for_stem(source_roots, stem, partition):
    roots = source_roots.get(stem, [])
    if not roots:
        raise ValueError(
            f"Missing source roots for stem '{stem}'. "
            "Please check your source_roots config."
        )

    output = []
    for root in roots:
        resolved_root = _resolve_partition_root(root, partition)
        if stem == "speech" and not _speech_root_matches_partition(resolved_root, partition):
            continue
        output.extend(_get_filepaths(resolved_root, return_dict=True))

    if not output:
        raise ValueError(
            f"No wav files found for stem '{stem}' and partition '{partition}'. "
            "Please verify your source roots."
        )
    return output



def main(
    root_dir,
    saving_dir_name='DnRv3-vgg',
    split='test',
    num_samples=20000,
    source_roots_config=None,
):
    # get the rank and size from torch.distributed
    import torch.distributed as dist
    # initialize the process group
    dist.init_process_group(backend='nccl', init_method='env://')
    
    rank = dist.get_rank()
    size = dist.get_world_size()
    
    # print(f"rank: {rank}, size: {size}")
    
    # set numpy seed
    # np.random.seed(rank)
    
    partition = split
    source_roots = _read_source_roots_config(source_roots_config)

    generator = MixtureGeneratorDnRv3(
        partition=partition,
        rank=rank,
        enable_background_sfx=False,
        source_roots=source_roots,
    )
    stems=['speech', 'sfx', 'music']

    saving_dir = f"{root_dir}/{saving_dir_name}/{split}/parts_{rank}"
    
    print(f"rank: {rank}, saving_dir: {saving_dir}")
    generated_samples = 0
    none = 0
    
    # scan the generated samples and reset the generated_samples
    # set the number to the largest number in the folder
    if os.path.exists(saving_dir):
        for dirs in os.listdir(saving_dir):
            if int(dirs) > generated_samples:
                generated_samples = int(dirs)

        if generated_samples > 0:
            print(f"rank: {rank}, Find generated_samples: {generated_samples}")
    
    while generated_samples < num_samples:
        # output = dataset.mixture_generator()
        output = generator()
        if output is None:
            none+=1
            continue
        normalized_sources, mixture, annots = output

        
        # check if ther infs or nans in normalized_sources and mixture, if so, skip this iteration
        skip = False
        for k, v in normalized_sources.items():
            if (v == float('inf')).any() or (v == float('nan')).any():
                skip = True
                print(f"inf or nan in {k}")
                break
        # print(i)

        # save the audio into four files, also the annots as json
        os.makedirs(os.path.join(saving_dir, f"{generated_samples}"), exist_ok=True)
        # the size is 1.1M for each audio folder
        
        for k, v in normalized_sources.items():
            sf.write(f"{saving_dir}/{generated_samples}/{k}.wav", v, 16000)
        sf.write(f"{saving_dir}/{generated_samples}/mixture.wav", mixture, 16000)
        with open(f"{saving_dir}/{generated_samples}/annots.json", 'w') as f:
            json.dump(annots, f, indent=4)
        
        generated_samples += 1
        
        if generated_samples % 200 == 0:
            print(f"rank: {rank}, generated_samples: [{generated_samples}]/[{num_samples}]")
            print(f"rank: {rank}, # of None: [{none}]")
            
    # wait for all the processes to finish
    print(f"rank: {rank}, finished")
    # dist.barrier()

def filelist_generate(root_dir, saving_dir_name, split):
    # generate the file_list_train.txt for total
    audio_files_dir = f"{root_dir}/{saving_dir_name}/{split}"
    file_list_dir = f'{root_dir}/{saving_dir_name}/file_list_{split}.txt'
    number_of_files = 0
    sr = 16000

    file_list_all = []
    # search for all the files in the audio_files_dir
    for dirs in os.listdir(audio_files_dir):
        
        for file in tqdm.tqdm(os.listdir(os.path.join(audio_files_dir, dirs))):
            path = os.path.join(audio_files_dir, dirs, file)
            
            try:
       
                with sf.SoundFile(os.path.join(path, 'speech.wav')) as f:
                    f.read()
                with sf.SoundFile(os.path.join(path, 'sfx.wav')) as f:
                    f.read()
                with sf.SoundFile(os.path.join(path, 'music.wav')) as f:
                    f.read()
                with sf.SoundFile(os.path.join(path, 'mixture.wav')) as f:
                    f.read()

            except:
                print(f"error in loading {path}")
                continue
            
            file_list_all.append(path)
            number_of_files += 1
        print(f"path: {os.path.join(audio_files_dir, dirs)} finished, number of total files now: {number_of_files}")
    
    with open(file_list_dir, 'w') as f:
        for file in file_list_all:
            f.write(f"{file}\n")
    
    with open(f'{root_dir}/{saving_dir_name}/number_of_files_{split}.txt', 'w') as f:
        f.write(f"{number_of_files}")


def overlap_of(x, y): # two ranges
    # x: (x_min, x_max)
    # y: (y_min, y_max)
    return (max(x[0], y[0]), min(x[-1], y[-1]))


def video_info_generate(root_dir, saving_dir_name, split):
    # some hp need to be set
    sr = 16000
    load_fps=4
    num_vid_frames = 60 * load_fps
    frame_length = 0.25 * sr
    
    file_list_dir = f'{root_dir}/{saving_dir_name}/file_list_{split}.txt'
    video_info_dir = f'{root_dir}/{saving_dir_name}/video_info_{split}.json'
    
    with open(file_list_dir, 'r') as f:
        file_list = f.readlines()
    file_list = [x.strip() for x in file_list] # remove the \n
    
    for file in file_list:
        with open(os.path.join(file, 'annots.json'), 'r') as f:
            annots = json.load(f)
            
        # audio_path=annots['sfx']['seg_info_dict'][i]['file_path']
        
        assert len(annots['sfx']) > 0, f"no sfx in {file}"
        
        sfx_vid_frames = [None] * num_vid_frames
        
        for i in range(len(annots['sfx'])):
            mix_start_time = annots['sfx']['seg_info_dict'][i]['event_start']
            clip_duration = annots['sfx']['seg_info_dict'][i]['event_duration']
            mix_end_time = mix_start_time + clip_duration
            
            # the dura should be longer than 0.25s
            if clip_duration < frame_length:
                print(f"clip_duration is too short: {clip_duration}")
                continue
            
            frame_start = mix_start_time // frame_length
            frame_end = mix_end_time // frame_length
            
            # total number of frames
            num_frames = frame_end - frame_start
            
        

    # with open(video_info_dir, 'w') as f:
    #     json.dump(video_info, f, indent=4)

if __name__ == "__main__":  
    parser = argparse.ArgumentParser(description="Legacy AVDnR builder without hard-coded source paths.")
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--root-dir", required=True, help="Output root directory for generated dataset.")
    parser.add_argument("--saving-dir-name", default="AVDnR")
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument(
        "--source-roots-config",
        required=True,
        help="Path to source_roots JSON (e.g. configs/source_roots.json).",
    )
    args = parser.parse_args()

    split = args.split
    root_dir = args.root_dir
    saving_dir_name = args.saving_dir_name
    num_samples = args.num_samples if args.num_samples is not None else (100 if split == "test" else 1000)
    # num_samples = 100

    # Step 1: generate the audio files
    # running the script with torchrun --standalone --master_port=26666 --nproc_per_node=10 build_DnRv3_fixed.py
    # main(root_dir=root_dir, saving_dir_name=saving_dir_name, split=split, num_samples=num_samples, source_roots_config=args.source_roots_config)
    # exit()
    
    
    # NOTE: the following steps do not need to be run in parallel
    # check if running with torchrun, if so, the world_size should be 1
    # import torch.distributed as dist
    # world_size = dist.get_world_size()
    # if world_size > 1:
    #     print("The world_size is larger than 1, please run the following steps in a single process")
    #     exit()
    
    # Step 2: generate the file list
    filelist_generate(root_dir, saving_dir_name, split)
    
    # Step 3: generate the video information
    
    
