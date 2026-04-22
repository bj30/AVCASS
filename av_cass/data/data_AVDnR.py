import abc
import functools
import itertools
import math
import os
import warnings
from abc import ABC
from pathlib import Path
from typing import *
import random
import pandas as pd
import json

import av
import librosa
import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset
from glob import glob
import soundfile as sf
import cv2
from PIL import Image
from time import time
from .spec_utils import stft_kwargs, istft_kwargs, spec_fwd, spec_fwd_numpy


def overlap_of(x, y): # two ranges
    # x: (x_min, x_max)
    # y: (y_min, y_max)
    return (max(x[0], y[0]), min(x[-1], y[-1]))

class MultiSourceDataset(Dataset):
    def __init__(self, 
            sr, 
            sample_length, 
            audio_files_dir, 
            stems, 
            mixture_name,
            audio_feat_type='waveform', # waveform, spec
            visual_encoder_type=None,
            enlarge_dataset=1,
            limit_samples=-1,
            eval_mode=False,
            load_whole=False,
            vid_stem_type=['sfx'],
            vid_stem_mode='random_select', # switch, random_select
            rank=0,
            world_size=1,
            exclude_list=[],
            root_dnrv3_dataset_path=None,
        ):
        super().__init__()
        self.sr = sr
        self.sample_length = sample_length
        self.vid_stem_type = vid_stem_type
        self.vid_stem_mode = vid_stem_mode
        self.audio_feat_type = audio_feat_type
        self.load_whole = load_whole
        self.av_mode = True if visual_encoder_type is not None else False
        self.random_visual_input = False
        if 'dnr_v2' in audio_files_dir:
            if eval_mode and not load_whole:
                enlarge_dataset = 1
                self.audio_files_dir = os.path.join(audio_files_dir, "cv")
            elif eval_mode and load_whole:
                enlarge_dataset = 1
                self.audio_files_dir = os.path.join(audio_files_dir, "tt")
            else:
                self.audio_files_dir = os.path.join(audio_files_dir, "tr")
            self.audio_dir_list = glob(f'{self.audio_files_dir}/*/')
        elif 'DnRv3' in audio_files_dir:
            if eval_mode and load_whole:
                self.audio_files_dir = root_dnrv3_dataset_path
                self.audio_dir_list = glob(f'{self.audio_files_dir}/*/')
            else:
                raise ValueError("dnrv3 is currently only for testing")
        else:
            if eval_mode:
                enlarge_dataset = 1
                self.audio_files_dir = os.path.join(audio_files_dir, "test")
                file_list_path = f'{audio_files_dir}/file_list_test.txt'
                if not os.path.exists(self.audio_files_dir):
                    self.audio_files_dir = os.path.join(audio_files_dir, "eval")
                    file_list_path = f'{audio_files_dir}/file_list_eval.txt'
            else:
                self.audio_files_dir = os.path.join(audio_files_dir, "train")
                file_list_path = f'{audio_files_dir}/file_list_train.txt'

            if os.path.exists(file_list_path):
                with open(file_list_path, 'r') as f:
                    audio_dirs = f.readlines()
                self.audio_dir_list = [d.strip() for d in audio_dirs if d.strip()]
            else:
                self.audio_dir_list = glob(f'{self.audio_files_dir}/*/') 

            self.true_dataset_samples = len(self.audio_dir_list)
        
        self.eval_mode = eval_mode
        if limit_samples > 0:
            self.audio_dir_list = self.audio_dir_list[:limit_samples]
        else:
            self.audio_dir_list = self.audio_dir_list * enlarge_dataset # enlarge the dataset

        if len(exclude_list)>0:
            print(self.audio_dir_list[:3])
            print(exclude_list[:3])
            self.audio_dir_list = [x for x in self.audio_dir_list if x not in exclude_list]

        # distribute the dataset into each rank
        self.audio_dir_list = self.audio_dir_list[rank::world_size]
        self.stems = stems
        self.visual_encoder_type = visual_encoder_type
        
        self.init_dataset()
        self.mixture_name = mixture_name
        
        self.stft_kwargs = stft_kwargs
        self.spec_transform = spec_fwd_numpy # spec_fwd
	

    def sf_load_audio(self, path, frame_start, frame_length):
        try:
            with sf.SoundFile(path) as f:
                # check the sampling rate
                # assert f.samplerate == self.sr
                f.seek(frame_start)
                aud = f.read(frame_length, dtype='float32')
        except:
            raise ValueError(path)
        return aud
    
    def to_spec(self, aud):
        # spec = torch.stft(torch.from_numpy(aud), **self.stft_kwargs) # [F, T=?]
        spec = librosa.stft(aud, **self.stft_kwargs) # [256, 512]
        # print(spec.shape)
        spec = spec.transpose() # [T=512, F=256]
        # print(spec.shape)
        return spec
    
    
    def get_dnr(self, idx):
        audio_dir = self.audio_dir_list[idx]

        if self.eval_mode:
            # for eval mode, start idx is fixed
            if self.load_whole:
                start_idx = 0
                # stacked_sources, mixture, case_num, file_num = self._get_whole_audio(idx)
                speech = self.sf_load_audio(os.path.join(audio_dir, 'speech.wav'), start_idx, -1)
                sfx = self.sf_load_audio(os.path.join(audio_dir, 'sfx.wav'), start_idx, -1)
                music = self.sf_load_audio(os.path.join(audio_dir, 'music.wav'), start_idx, -1)
                mixture = self.sf_load_audio(os.path.join(audio_dir, f'{self.mixture_name}.wav'), start_idx, -1)
                sources_list = [speech, sfx, music]
                
                if self.audio_feat_type=='spec':
                    sources_list = [np.array(self.spec_transform(self.to_spec(source))) for source in sources_list] # [(T=?, F=256)*4]
                    # change complex spectrogram to 2-channel real-valued spectrogram
                    sources_list = [np.stack([source.real, source.imag], axis=0) for source in sources_list] # [(2, T=?, F=256)*4]
                    mixture = np.array(self.spec_transform(self.to_spec(mixture)))
                    mixture = np.stack([mixture.real, mixture.imag], axis=0)
                    stacked_sources = np.concatenate(sources_list, axis=0)
                else:
                    stacked_sources = np.stack(sources_list, axis=0)
                if self.av_mode:
                    if self.random_visual_input:
                        vid = self.get_random_vid(random_type='noise')
                    else:
                        if self.visual_encoder_type == 'cavp':
                            vid_dir = os.path.join(audio_dir, 'video_full_sfx.npy')
                        elif self.visual_encoder_type =='talknet':
                            vid_dir = os.path.join(audio_dir, 'video_full_25.npy')
                        elif self.visual_encoder_type == "SSLAlignment":
                            vid_dir = os.path.join(audio_dir, f'video_full_{self.vid_stem_type[0]}.npy')
                        else:
                            vid_dir = os.path.join(audio_dir, 'video_full.npy')
                        
                        if not os.path.exists(vid_dir):
                            vid = get_vid_cavp_general(self.visual_encoder_type, audio_dir, self.sr*60, self.sr, 0, self.vid_stem_type, 'json')
                            # np.save(vid_dir, vid)
                            # print(f"Saved video: {vid_dir}")
                        else:
                            vid = np.load(vid_dir)
                        # vid = self.get_vid_cavp(audio_dir,)
                    vid = np.float16(vid)
                    output = (torch.from_numpy(stacked_sources), torch.from_numpy(mixture), torch.from_numpy(vid), audio_dir)
                else:
                    output = (stacked_sources, mixture, audio_dir)
                return output
            else:
                start_idx = 16000 # for validation
                case_num, file_num = audio_dir.split('/')[-2:]
                aud_sources = []
                end_idx = start_idx + self.sample_length

                if self.av_mode:
                    if self.random_visual_input:
                        vid = self.get_random_vid(audio_dir, random_type='noise')
                    else:
                        # vid = self.get_vid_cavp(audio_dir, start_idx, end_idx)
                        vid = get_vid_cavp_general(self.visual_encoder_type, audio_dir, self.sample_length, self.sr, start_idx, self.vid_stem_type, 'json')
                
                speech = self.sf_load_audio(os.path.join(audio_dir, 'speech.wav'), start_idx, self.sample_length)
                sfx = self.sf_load_audio(os.path.join(audio_dir, 'sfx.wav'), start_idx, self.sample_length)
                music = self.sf_load_audio(os.path.join(audio_dir, 'music.wav'), start_idx, self.sample_length)
                sources_list = [speech, sfx, music]
                mixture = self.sf_load_audio(os.path.join(audio_dir, f'{self.mixture_name}.wav'), start_idx, self.sample_length)
                
                if self.audio_feat_type=='spec':
                    sources_list = [np.array(self.spec_transform(self.to_spec(source))) for source in sources_list] # [(T=?, F=256)*4]
                    # change complex spectrogram to 2-channel real-valued spectrogram
                    sources_list = [np.stack([source.real, source.imag], axis=0) for source in sources_list] # [(2, T=?, F=256)*4]
                    mixture = np.array(self.spec_transform(self.to_spec(mixture)))
                    mixture = np.stack([mixture.real, mixture.imag], axis=0)
                    stacked_sources = np.concatenate(sources_list, axis=0)
                else:
                    stacked_sources = np.stack(sources_list, axis=0)

                if self.av_mode:
                    vid = np.float16(vid)
                    vid_tensor = torch.from_numpy(vid)
                    return stacked_sources, mixture, vid_tensor, audio_dir
                else:
                    return stacked_sources, mixture, audio_dir
                
        else:
            start_idx = random.randrange(0, 60*self.sr -self.sample_length)

        speech = self.sf_load_audio(os.path.join(audio_dir, 'speech.wav'), start_idx, self.sample_length)
        sfx = self.sf_load_audio(os.path.join(audio_dir, 'sfx.wav'), start_idx, self.sample_length)
        music = self.sf_load_audio(os.path.join(audio_dir, 'music.wav'), start_idx, self.sample_length)
        mixture = self.sf_load_audio(os.path.join(audio_dir, f'{self.mixture_name}.wav'), start_idx, self.sample_length)
        
        sources_list = [speech, sfx, music, mixture]
        if self.audio_feat_type=='spec':
            sources_list = [np.array(self.spec_transform(self.to_spec(source))) for source in sources_list] # [(T=?, F=256)*4]
            # change complex spectrogram to 2-channel real-valued spectrogram
            sources_list = [np.stack([source.real, source.imag], axis=0) for source in sources_list] # [(2, T=?, F=256)*4]
            stacked_sources = np.concatenate(sources_list, axis=0)
        else:
            stacked_sources = np.stack(sources_list, axis=0)
        if self.visual_encoder_type is None:
            return stacked_sources
        else:
            if self.visual_encoder_type == 'cavp' or self.visual_encoder_type == 'talknet' or self.visual_encoder_type == 'SSLAlignment':
                vid = self.get_vid_cavp(audio_dir, start_idx)
            elif 'dino' in self.visual_encoder_type:
                vid = self.get_vid(idx)
            else:
                raise ValueError(f"Visual encoder {self.visual_encoder_type} not implemented")
            return stacked_sources, vid
    
    def get_vid_cavp(self, audio_dir, start_idx, end_idx=None, info_type='json'):
        
        if self.visual_encoder_type in ['cavp', 'SSLAlignment']:
            load_fps = 4
            colorscale = 'RGB'
            H = 224
            
            video_len = int(self.sample_length / self.sr * load_fps) # 32
            video_shape = (video_len, 3, H, H)
            
            
        elif self.visual_encoder_type =='talknet':
            load_fps = 25
            colorscale = 'gray'
            H = 112
            
            video_len = int(self.sample_length / self.sr * load_fps) # 32
            video_shape = (video_len, H, H)
            
        vid_fps = 25
        
        start_time = start_idx / self.sr
        if end_idx is None:
            end_time = (start_idx + self.sample_length) / self.sr
        else:
            end_time = end_idx / self.sr

        if info_type!='json':
            raise ValueError("implemented only for json")
        
        json_path = os.path.join(audio_dir, 'annots.json')
        with open(json_path) as f:
            data_dict = json.load(f)
        if len(data_dict['speech'])==0 and len(data_dict['sfx'])==0:
            # no video available
            return np.zeros(video_shape)

        existence = {'speech':np.zeros((video_len,)), 'sfx':np.zeros((video_len,))}
        frames = {'speech':[], 'sfx':[]}

        frames['speech'] = np.zeros(video_shape)
        frames['sfx'] = np.zeros(video_shape)
        
        if len(self.vid_stem_type)>1 and self.vid_stem_mode=='switch':
            if random.random() < 0.5:
                vid_stem_type = ['sfx']
            else:
                vid_stem_type = ['speech']
        else:
            vid_stem_type = self.vid_stem_type
        
        for stem in vid_stem_type:
            if len(data_dict[stem])==0:
                continue
            for i in range(len(data_dict[stem]['seg_info_dict'])):
                mix_start_time = data_dict[stem]['seg_info_dict'][i]['event_start'] / self.sr
                clip_duration = data_dict[stem]['seg_info_dict'][i]['event_duration'] / self.sr
                mix_end_time = mix_start_time + clip_duration
                overlap_start, overlap_end = overlap_of((start_time, end_time), (mix_start_time, mix_end_time))
                if overlap_start >= (overlap_end-0.04): # no overlap of video 
                    continue
                # else: at least 1 video frame is included

                # convert 60s-mixture-timestamp -> stem-side timestamp
                clip_start_time = data_dict[stem]['seg_info_dict'][i]['audio_clip_start_time'] / self.sr
                offset = mix_start_time - clip_start_time
                # by subtracting "offset" value from 60s-side timestamps,
                # it converts the 60s-side timestamp to the stem-side timestamp
                source_start_time = overlap_start - offset
                source_end_time = overlap_end - offset

                # source-audio-time -> source-video-idx
                start_fidx = int(source_start_time * vid_fps)
                end_fidx = int(source_end_time * vid_fps)
                if end_fidx - start_fidx <= int(vid_fps / load_fps):
                    continue

                # convert 60s-side timestamp -> chunk-side timestamp
                chunk_start_time = overlap_start - start_time
                start_chunk_idx = int(chunk_start_time * load_fps)
                frame_list = extract_frames(
                    fidx_range=(start_fidx, end_fidx), 
                    audio_path=data_dict[stem]['seg_info_dict'][i]['file_path'], 
                    frames_per_second = load_fps,
                    H = H,
                    colorscale = colorscale,
                    )
                end_chunk_idx = start_chunk_idx + len(frame_list)
                frames[stem][start_chunk_idx:end_chunk_idx] = frame_list
                existence[stem][start_chunk_idx:end_chunk_idx] = np.ones((end_chunk_idx - start_chunk_idx,))

                    
        vid_stems = ['' for i in range(video_len)]
        speech_index = np.where(existence['speech']==1)[0]
        for sidx in speech_index:
            vid_stems[sidx] = 'speech'
        sfx_index = np.where(existence['sfx']==1)[0]
        for sfdx in sfx_index:
            vid_stems[sfdx] = 'sfx'
        # randomly select the stem for overlapping range
        overlap = (existence['speech'] + existence['sfx'])==2
        
        if overlap.any():
            overlap_index = np.where(overlap==True)[0]
            for oidx in overlap_index:
                vid_stems[oidx] = random.choice(['speech', 'sfx'])
        
        final_frames = []
        for idx, vid_stem in enumerate(vid_stems):
            if vid_stem=='':
                final_frames.append(np.zeros(video_shape[1:]))
            else:
                final_frames.append(frames[vid_stem][idx])
        return np.array(final_frames, dtype=np.float32)

    def init_dataset(self):
        # Load list of tracks and starts/durations
        print(f"Found {len(self.audio_dir_list)} tracks from {self.audio_files_dir}")


    def __len__(self):
        return len(self.audio_dir_list) 

    def __getitem__(self, idx):
        dnr_item = self.get_dnr(idx)
        if self.eval_mode:
            return dnr_item
        if self.visual_encoder_type is None:
            stacked_sources = dnr_item
            vid_tensor = None
        else:
            stacked_sources, vid = dnr_item
            vid_tensor = torch.from_numpy(vid).unsqueeze(0) if self.visual_encoder_type=='dinov2_noT' else torch.from_numpy(vid)
            

        if self.visual_encoder_type is None:
            return torch.from_numpy(stacked_sources)
        else:
            return torch.from_numpy(stacked_sources), vid_tensor

# Datasets for evaluation --------------------------------------------------------------

class SeparationDataset(Dataset, ABC):
    @abc.abstractmethod
    def __getitem__(self, item) -> Tuple[torch.Tensor, ...]:
        ...

    @abc.abstractmethod
    def __len__(self) -> int:
        ...

    @property
    @abc.abstractmethod
    def sample_rate(self) -> int:
        ...

class DnRv3Dataset(SeparationDataset):
    def __init__(
        self,
        dnr_dir,
        stems = ['speech', 'sfx', 'music', 'mixture'],
        sample_rate = 16000,
        sample_eps_in_sec = 0.1,
        samples_case = 0,
        sample_length = 131072,
        start_idx = 0,
        mixture_name = 'mixture', # mixture for DnRv3, mix for DnRv1
        visual_encoder_type=None,
        load_whole = False,
        random_visual_input = False,
        vid_stem_type = ['sfx'],
        limited_samples = 0,
    ):
        super().__init__()
        self.sr = sample_rate
        self.sample_eps = round(sample_eps_in_sec * sample_rate)
        self.stems = stems
        self.vid_stem_type = vid_stem_type
        self.test_list = []

        
        for root, dirs, files in os.walk(dnr_dir):
            if not dirs:
                self.test_list.append(os.path.join(dnr_dir,root))
        self.dnr_dir = dnr_dir
        self.test_list = self.test_list
        if limited_samples>0:
            self.test_list = self.test_list[:limited_samples]
        
        if isinstance(sample_length, float):
            sample_length = int(sample_length * sample_rate)
        self.sample_length = sample_length
        self.start_idx = start_idx if not load_whole else 0
        self.mixture_name = mixture_name
        self.load_whole = load_whole
        self.av_mode = True if visual_encoder_type is not None else False
        self.visual_encoder_type = visual_encoder_type
        self.random_visual_input = random_visual_input

        print(f"Found {len(self.test_list)} tracks from {self.dnr_dir}")

    def __len__(self):
        return len(self.test_list)

    @property
    def sample_rate(self) -> int:
        return self.sr
    
    def __getitem__(self, idx):
        dirpath = self.test_list[idx]
        
        if self.load_whole:
            stacked_sources, mixture, case_num, file_num = self._get_whole_audio(idx)
            if self.av_mode:
                if self.random_visual_input:
                    vid = self.get_random_vid(random_type='noise')
                else:
                    if self.visual_encoder_type == 'cavp':
                        vid_dir = os.path.join(dirpath, 'video_full_sfx.npy')
                    elif self.visual_encoder_type =='talknet':
                        vid_dir = os.path.join(dirpath, 'video_full_25.npy')
                    elif self.visual_encoder_type == "SSLAlignment":
                        vid_dir = os.path.join(dirpath, f'video_full_{self.vid_stem_type[0]}.npy')
                    else:
                        vid_dir = os.path.join(dirpath, 'video_full.npy')
                    
                        
                    if not os.path.exists(vid_dir):
                        vid = get_vid_cavp_general(self.visual_encoder_type, dirpath, self.sr*60, self.sr, 0, self.vid_stem_type, 'json')
                        np.save(vid_dir, vid)
                        print(f"Saved video: {vid_dir}")
                    else:
                        vid = np.load(vid_dir)
                vid = np.float16(vid)
                output = (torch.from_numpy(stacked_sources), torch.from_numpy(mixture), torch.from_numpy(vid), dirpath)
            else:
                output = (stacked_sources, mixture, dirpath)

            return output
        
        else:
            
            case_num, file_num = dirpath.split('/')[-2:]
            aud_sources = []
            start_idx = self.start_idx 
            end_idx = start_idx + self.sample_length

            if self.av_mode:
                if self.random_visual_input:
                    vid = self.get_random_vid(dirpath, random_type='noise')
                else:
                    vid = get_vid_cavp_general(self.visual_encoder_type, dirpath, self.sample_length, self.sr, start_idx, self.vid_stem_type, 'json')
            
            if len(self.stems) == 4:
                stems = self.stems[:3]
            else:
                stems = self.stems
            for stem in stems:
                with sf.SoundFile(os.path.join(dirpath, f'{stem}.wav')) as f:
                    f.seek(start_idx)
                    aud = f.read(self.sample_length, dtype='float32')
                
                aud = aud
                aud_sources.append(aud)

            stacked_sources = np.stack(aud_sources, axis=0)

            with sf.SoundFile(os.path.join(dirpath, f'{self.mixture_name}.wav')) as f:
                f.seek(start_idx)
                mixture = f.read(self.sample_length, dtype='float32')

            if self.av_mode:
                vid = np.float16(vid)
                vid_tensor = torch.from_numpy(vid)
                return stacked_sources, mixture, vid_tensor, dirpath
            else:
                return stacked_sources, mixture, dirpath
    
    def _get_whole_audio(self, idx):
        dirpath = self.test_list[idx]
        
        case_num, file_num = dirpath.split('/')[-2:]
        aud_sources = []
        with sf.SoundFile(os.path.join(dirpath, f'{self.mixture_name}.wav')) as f:
            mixture = f.read(dtype='float32')

        
        for stem in self.stems[:3]:

            with sf.SoundFile(os.path.join(dirpath, f'{stem}.wav')) as f:
                aud = f.read(dtype='float32')
            aud_sources.append(aud)
        stacked_sources = np.stack(aud_sources, axis=0)

        return stacked_sources, mixture, case_num, file_num

    
    def get_random_vid(self, random_type='noise'):
        if self.visual_encoder_type=='cavp':
            load_fps = 4
            if not self.load_whole:
                video_len = int(self.sample_length / self.sr * load_fps)
            else:
                video_len = int(60 * load_fps)
        elif self.visual_encoder_type=='talknet':
            load_fps = 25
            video_len = int(self.sample_length / self.sr * load_fps)
        else:
            video_len = 1
        
        if self.visual_encoder_type=='talknet':
            H = 112
            video_shape = (video_len, H, H)
            padding = ((0,0),(0,0))
            vid_stem = 'speech'
        elif 'dinov2' in self.visual_encoder_type:
            H = 224
            vid_stem = 'speech'
            if self.visual_encoder_type=='dinov2_noT':
                video_len = 1
                vid_stem = 'sfx'
            video_shape = (video_len, 3, H, H)
            padding = ((0,0),(0,0),(0,0))
            if random_type=='noise':
                return np.random.randn(video_len, 3, H, H) * 255.

        elif self.visual_encoder_type == 'cavp':
            H = 224
            video_shape = (video_len, 3, H, H)
            if random_type=='noise':
                return np.random.randn(video_len, 3, H, H) * 255.

    

    
    def get_vid_cavp(self, audio_dir, start_idx=0, end_idx=None, info_type='json'):
        if self.visual_encoder_type in ['cavp', 'SSLAlignment']:
            load_fps = 4
            colorscale = 'RGB'
            H = 224
        elif self.visual_encoder_type =='talknet':
            load_fps = 25
            colorscale = 'gray' # grayscale
            H = 112
        vid_fps = 25
        whole_length = 60

        if self.load_whole:
            video_len = int(whole_length * load_fps)
        else:
            video_len = int(self.sample_length / self.sr * load_fps) # 32
        
        if self.visual_encoder_type in ['cavp', 'SSLAlignment']:
            video_shape = (video_len, 3, H, H)
        elif self.visual_encoder_type == 'talknet':
            video_shape = (video_len, H, H)
        
        start_time = start_idx / self.sr
        if self.load_whole:
            end_time = whole_length
        elif end_idx is None:
            end_time = (start_idx + self.sample_length) / self.sr
        else:
            end_time = end_idx / self.sr

        if info_type!='json':
            raise ValueError("implemented only for json")
        
        json_path = os.path.join(audio_dir, 'annots.json')
        with open(json_path) as f:
            data_dict = json.load(f)
        if len(data_dict['speech'])==0 and len(data_dict['sfx'])==0:
            # no video available
            return np.zeros(video_shape)

        existence = {'speech':np.zeros((video_len,)), 'sfx':np.zeros((video_len,))}
        frames = {'speech':[], 'sfx':[]}
        for i in range(video_len):
            frames['speech'] = np.zeros(video_shape)
            frames['sfx'] = np.zeros(video_shape)
        
        
        for stem in self.vid_stem_type:
            if len(data_dict[stem])==0:
                continue
            for i in range(len(data_dict[stem]['seg_info_dict'])):
                mix_start_time = data_dict[stem]['seg_info_dict'][i]['event_start'] / self.sr
                clip_duration = data_dict[stem]['seg_info_dict'][i]['event_duration'] / self.sr
                mix_end_time = mix_start_time + clip_duration
                overlap_start, overlap_end = overlap_of((start_time, end_time), (mix_start_time, mix_end_time))
                if overlap_start >= (overlap_end-0.04): # no overlap of video 
                    continue
                # else: at least 1 video frame is included

                # convert 60s-mixture-timestamp -> stem-side timestamp
                clip_start_time = data_dict[stem]['seg_info_dict'][i]['audio_clip_start_time'] / self.sr
                offset = mix_start_time - clip_start_time
                # by subtracting "offset" value from 60s-side timestamps,
                # it converts the 60s-side timestamp to the stem-side timestamp
                source_start_time = overlap_start - offset
                source_end_time = overlap_end - offset

                # source-audio-time -> source-video-idx
                start_fidx = int(source_start_time * vid_fps)
                end_fidx = int(source_end_time * vid_fps)
                if end_fidx - start_fidx <= int(vid_fps / load_fps):
                    continue

                # convert 60s-side timestamp -> chunk-side timestamp
                chunk_start_time = overlap_start - start_time
                start_chunk_idx = int(chunk_start_time * load_fps)
            
                frame_list = extract_frames(
                    fidx_range=(start_fidx, end_fidx), 
                    audio_path=data_dict[stem]['seg_info_dict'][i]['file_path'], 
                    frames_per_second = load_fps,
                    H = H,
                    colorscale = colorscale,
                    )
                end_chunk_idx = start_chunk_idx + len(frame_list)
                frames[stem][start_chunk_idx:end_chunk_idx] = frame_list
                existence[stem][start_chunk_idx:end_chunk_idx] = np.ones((end_chunk_idx - start_chunk_idx,))

                    
        vid_stems = ['' for i in range(video_len)]
        speech_index = np.where(existence['speech']==1)[0]
        for sidx in speech_index:
            vid_stems[sidx] = 'speech'
        sfx_index = np.where(existence['sfx']==1)[0]
        for sfdx in sfx_index:
            vid_stems[sfdx] = 'sfx'
        # randomly select the stem for overlapping range
        overlap = (existence['speech'] + existence['sfx'])==2
        
        if overlap.any():
            overlap_index = np.where(overlap==True)[0]
            for oidx in overlap_index:
                vid_stems[oidx] = random.choice(['speech', 'sfx'])
        
        final_frames = []
        for idx, vid_stem in enumerate(vid_stems):
            if vid_stem=='':
                final_frames.append(np.zeros(video_shape[1:]))
            else:
                final_frames.append(frames[vid_stem][idx])
        
        return np.array(final_frames)



class DnRV3(DnRv3Dataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

def get_vid_cavp_general(visual_encoder_type, audio_dir, sample_length, sampling_rate, start_idx=0, vid_stem_type=[], info_type='json', ):
    if visual_encoder_type in ['cavp', 'SSLAlignment']:
        load_fps = 4
        colorscale = 'RGB'
        H, W = 224, 224
        
        video_len = int(sample_length / sampling_rate * load_fps) # 32
        video_shape = (video_len, 3, H, W)
        
        
    elif visual_encoder_type =='talknet':
        load_fps = 25
        colorscale = 'gray'
        H = 112
        
        video_len = int(sample_length / sampling_rate * load_fps) # 32
        video_shape = (video_len, H, H)
        
    vid_fps = 25

    start_time = start_idx / sampling_rate
    end_time = (start_idx + sample_length) / sampling_rate

    if info_type!='json':
        raise ValueError("implemented only for json")
    
    json_path = os.path.join(audio_dir, 'annots.json')
    with open(json_path) as f:
        data_dict = json.load(f)
    if len(data_dict['speech'])==0 and len(data_dict['sfx'])==0:
        # no video available
        return np.zeros(video_shape)

    existence = {'speech':np.zeros((video_len,)), 'sfx':np.zeros((video_len,))}
    frames = {'speech':[], 'sfx':[]}
    for i in range(video_len):
        frames['speech'] = np.zeros(video_shape)
        frames['sfx'] = np.zeros(video_shape)
    
    
    for stem in vid_stem_type:
        if len(data_dict[stem])==0:
            continue
        for i in range(len(data_dict[stem]['seg_info_dict'])):
            mix_start_time = data_dict[stem]['seg_info_dict'][i]['event_start'] / sampling_rate
            clip_duration = data_dict[stem]['seg_info_dict'][i]['event_duration'] / sampling_rate
            mix_end_time = mix_start_time + clip_duration
            overlap_start, overlap_end = overlap_of((start_time, end_time), (mix_start_time, mix_end_time))
            if overlap_start >= (overlap_end-0.04): # no overlap of video 
                continue
            # else: at least 1 video frame is included

            # convert 60s-mixture-timestamp -> stem-side timestamp
            clip_start_time = data_dict[stem]['seg_info_dict'][i]['audio_clip_start_time'] / sampling_rate
            offset = mix_start_time - clip_start_time
            # by subtracting "offset" value from 60s-side timestamps,
            # it converts the 60s-side timestamp to the stem-side timestamp
            source_start_time = overlap_start - offset
            source_end_time = overlap_end - offset

            # source-audio-time -> source-video-idx
            start_fidx = int(source_start_time * vid_fps)
            end_fidx = int(source_end_time * vid_fps)
            if end_fidx - start_fidx <= int(vid_fps / load_fps):
                continue

            # convert 60s-side timestamp -> chunk-side timestamp
            chunk_start_time = overlap_start - start_time
            start_chunk_idx = int(chunk_start_time * load_fps)
        
            frame_list = extract_frames(
                fidx_range=(start_fidx, end_fidx), 
                audio_path=data_dict[stem]['seg_info_dict'][i]['file_path'], 
                frames_per_second = load_fps,
                H = H,
                colorscale = colorscale,
                )
            end_chunk_idx = start_chunk_idx + len(frame_list)
            frames[stem][start_chunk_idx:end_chunk_idx] = frame_list
            existence[stem][start_chunk_idx:end_chunk_idx] = np.ones((end_chunk_idx - start_chunk_idx,))

                
    vid_stems = ['' for i in range(video_len)]
    speech_index = np.where(existence['speech']==1)[0]
    for sidx in speech_index:
        vid_stems[sidx] = 'speech'
    sfx_index = np.where(existence['sfx']==1)[0]
    for sfdx in sfx_index:
        vid_stems[sfdx] = 'sfx'
    # randomly select the stem for overlapping range
    overlap = (existence['speech'] + existence['sfx'])==2
    
    if overlap.any():
        overlap_index = np.where(overlap==True)[0]
        for oidx in overlap_index:
            vid_stems[oidx] = random.choice(['speech', 'sfx'])
    
    final_frames = []
    for idx, vid_stem in enumerate(vid_stems):
        if vid_stem=='':
            final_frames.append(np.zeros(video_shape[1:]))
        else:
            final_frames.append(frames[vid_stem][idx])
    
    return np.array(final_frames)

# Function to extract exactly 4 frames per second from a video
def extract_frames(fidx_range, audio_path, frames_per_second=4, H=224, colorscale='RGB'):
    if colorscale == 'RGB':
        frame_shape = (3, H, H)
        pil_colorscale = 'RGB'
    elif colorscale == 'gray':
        frame_shape = (H,H) # 'L'
        pil_colorscale = 'L'
    frames = []
    if '/vggsound_filtered/' in audio_path:
        
        chunk_name = os.path.basename(audio_path).split('_')[-1].replace('.wav','')
        json_path = audio_path.replace('/audio/', '/video_json/').replace(f'_{chunk_name}.wav', '.json')
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"json_path not found: {json_path}")
            
        with open(json_path, 'r') as f:
            data_dict = json.load(f)
        path_list = data_dict[chunk_name]

        fps = 25
        total_frames = fidx_range[1] - fidx_range[0]
        duration_in_seconds = total_frames / fps
        
        # Calculate the total number of frames we want to extract
        num_frames_to_load = int(duration_in_seconds * frames_per_second)
        if num_frames_to_load == 0:
            import pdb; pdb.set_trace()
        # Get frame indices evenly spaced across the video
        frame_indices = np.linspace(fidx_range[0], fidx_range[1] - 1, num=num_frames_to_load, dtype=int)
        if len(frame_indices) == 0:
            import pdb; pdb.set_trace()

        fpaths = [path_list[x] for x in frame_indices]
        for fpath in fpaths:
            if not os.path.exists(fpath):
                prev_frame = frames[-1]
                frames.append(prev_frame)
                continue
            try:
                frame = Image.open(fpath).convert(pil_colorscale) # cv2.imread(fpath)
            except:
                if len(frames)>0:
                    prev_frame = frames[-1]
                    frames.append(prev_frame)
                    continue
                else:
                    print(fpath)
                    frames.append(np.zeros(frame_shape))
                    continue
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame.resize((H,H)) #cv2.resize(frame, (H,H))
            frame = np.transpose(np.array(frame), (2,0,1))
            assert frame.shape == frame_shape, frame.shape
            frames.append(frame)

    elif '/VGGSound_final/audio/' in audio_path:
        frame_dir = audio_path.replace('/audio/', '/frames/').replace('.wav', '')
        frame_paths = sorted(glob(os.path.join(frame_dir, '*.jpg')))
        fps = 25
        total_frames = fidx_range[1] - fidx_range[0]
        duration_in_seconds = total_frames / fps
        num_frames_to_load = max(int(duration_in_seconds * frames_per_second), 1)
        frame_indices = np.linspace(fidx_range[0], fidx_range[1] - 1, num=num_frames_to_load, dtype=int)

        if len(frame_paths) == 0:
            return np.zeros((num_frames_to_load, *frame_shape), dtype=np.uint8)

        for frame_idx in frame_indices:
            safe_idx = min(max(int(frame_idx), 0), len(frame_paths) - 1)
            fpath = frame_paths[safe_idx]
            try:
                frame = Image.open(fpath).convert(pil_colorscale)
            except Exception:
                if len(frames) > 0:
                    frames.append(frames[-1])
                    continue
                frames.append(np.zeros(frame_shape))
                continue
            frame = frame.resize((H, H))
            frame = np.array(frame)
            if frame.ndim == 3:
                frame = np.transpose(frame, (2, 0, 1))
            assert frame.shape == frame_shape, frame.shape
            frames.append(frame)
    
    elif '/lrs3/' in audio_path:
        if colorscale == 'RGB':
            cv2_colorscale = cv2.COLOR_BGR2RGB
        elif colorscale == 'gray':
            cv2_colorscale = cv2.COLOR_BGR2GRAY
        video_path = audio_path.replace('.wav', '.mp4')
        # Open the video file
        cap = cv2.VideoCapture(video_path)

        fps = cap.get(cv2.CAP_PROP_FPS)
        
        total_frames = fidx_range[1]-fidx_range[0]
        duration_in_seconds = total_frames / fps
        
        # Calculate the total number of frames we want to extract
        num_frames_to_load = int(duration_in_seconds * frames_per_second)

        # Get frame indices evenly spaced across the video
        frame_indices = np.linspace(fidx_range[0], fidx_range[1] - 1, num=num_frames_to_load, dtype=int)
        if len(frame_indices)==0:
            print(fidx_range)
            import pdb; pdb.set_trace()
        # Initialize the current frame index
        current_frame_index = 0

        # Process the video
        frames = []

        while cap.isOpened():
            ret, frame = cap.read()
            if current_frame_index < fidx_range[0]:
                current_frame_index += 1
                continue

            if ret:
                if current_frame_index in frame_indices:
                    frame = cv2.cvtColor(frame, cv2_colorscale)
                    frame = cv2.resize(frame, (H,H))
                    if frame.ndim==3:
                        frame = np.transpose(frame, (2,0,1))
                    assert frame.shape == frame_shape, frame.shape
                    frames.append(frame)
                
                current_frame_index += 1
            else:
                break

            # Stop when all desired frames are processed
            if current_frame_index > frame_indices[-1]:
                break
    
    return np.array(frames)


if __name__=="__main__":
    #### remove "main." from these two imports
    # from main.visual.shared import norm
    # from main.spec_utils import stft_kwargs, istft_kwargs, spec_fwd, spec_back

    dataset = MultiSourceDataset(
                sr=16000, 
                sample_length=131072, 
                audio_files_dir='/mnt/scratch/datasets/AVDnR', 
                stems=['speech', 'sfx', 'music'], 
                mixture_name='mixture',
                audio_feat_type='spec', 
                eval_mode=True
                )
    item = dataset.__getitem__(0)
