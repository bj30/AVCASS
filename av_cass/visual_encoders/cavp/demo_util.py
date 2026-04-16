

import torch
import subprocess
from pathlib import Path
import os
import cv2
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
import librosa
from omegaconf import OmegaConf
import importlib

sr = 16000


def which_ffmpeg() -> str:
    '''Determines the path to ffmpeg library

    Returns:
        str -- path to the library
    '''
    result = subprocess.run(['which', 'ffmpeg'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    ffmpeg_path = result.stdout.decode('utf-8').replace('\n', '')
    return ffmpeg_path


def reencode_video_with_diff_fps(video_path: str, tmp_path: str, extraction_fps: int, start_second, truncate_second) -> str:
    '''Reencodes the video given the path and saves it to the tmp_path folder.

    Args:
        video_path (str): original video
        tmp_path (str): the folder where tmp files are stored (will be appended with a proper filename).
        extraction_fps (int): target fps value

    Returns:
        str: The path where the tmp file is stored. To be used to load the video from
    '''
    assert which_ffmpeg() != '', 'Is ffmpeg installed? Check if the conda environment is activated.'
    # assert video_path.endswith('.mp4'), 'The file does not end with .mp4. Comment this if expected'
    # create tmp dir if doesn't exist
    os.makedirs(tmp_path, exist_ok=True)

    # form the path to tmp directory
    if truncate_second is None:
        new_path = os.path.join(tmp_path, f'{Path(video_path).stem}_new_fps_{str(extraction_fps)}.mp4')
        if os.path.exists(new_path):
            print(f"Skipping {new_path}; already exists")
            return new_path
        cmd = f'{which_ffmpeg()} -hide_banner -loglevel panic '
        cmd += f'-y -i {video_path} -an -filter:v fps=fps={extraction_fps} {new_path}'
        subprocess.call(cmd.split())
    else:
        new_path = os.path.join(tmp_path, f'{Path(video_path).stem}_new_fps_{str(extraction_fps)}_truncate_{start_second}_{truncate_second}.mp4')
        if os.path.exists(new_path):
            print(f"Skipping {new_path}; already exists")
            return new_path
        cmd = f'{which_ffmpeg()} -hide_banner -loglevel panic '
        cmd += f'-y -ss {start_second} -t {truncate_second} -i {video_path} -an -filter:v fps=fps={extraction_fps} {new_path}'
        subprocess.call(cmd.split())
    return new_path


def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def init_from_ckpt(path, model_to_init):
    model = torch.load(path, map_location="cpu")
    if "state_dict" in list(model.keys()):
        model = model["state_dict"]
    # Remove: module prefix
    new_model = {}
    for key in model.keys():
        new_key = key.replace("module.","")
        new_model[new_key] = model[key]
    missing, unexpected = model_to_init.load_state_dict(new_model, strict=False)
    print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
    if len(missing) > 0:
        print(f"Missing Keys: {missing}")
    if len(unexpected) > 0:
        print(f"Unexpected Keys: {unexpected}")
    return model_to_init

class Extract_CAVP_Features(torch.nn.Module):

    def __init__(self, fps=4, batch_size=2, device=None, tmp_path="./", video_shape=(224,224), config_path=None, ckpt_path=None):
        super(Extract_CAVP_Features, self).__init__()
        self.fps = fps
        self.batch_size = batch_size
        self.device = device
        self.tmp_path = tmp_path

        # Initalize Stage1 CAVP model:
        print("Initalize Stage1 CAVP Model")
        config = OmegaConf.load(config_path)
        self.stage1_model = instantiate_from_config(config.model).to(device)

        # Loading Model from:
        assert ckpt_path is not None
        print("Loading Stage1 CAVP Model from: {}".format(ckpt_path))
        self.init_first_from_ckpt(ckpt_path)
        self.stage1_model.eval()
        
        # Transform:
        self.img_transform = transforms.Compose([
            transforms.Resize(video_shape),
            transforms.ToTensor(),
        ])
        from data_preprocess.wav2spec import TRANSFORMS
        self.audio_transform = TRANSFORMS
    
    
    def init_first_from_ckpt(self, path):
        model = torch.load(path, map_location="cpu")
        if "state_dict" in list(model.keys()):
            model = model["state_dict"]
        # Remove: module prefix
        new_model = {}
        for key in model.keys():
            new_key = key.replace("module.","")
            new_model[new_key] = model[key]
        missing, unexpected = self.stage1_model.load_state_dict(new_model, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")
    
    
    @torch.no_grad()
    def forward(self, video_path, start_second=None, truncate_second=None, tmp_path="./tmp_folder"):
        self.tmp_path = tmp_path
        
        # print("video_path", video_path)
        # print("truncate second: ", truncate_second)
        # Load the video, change fps:
        video_path_low_fps = reencode_video_with_diff_fps(video_path, self.tmp_path, self.fps, start_second, truncate_second)
        video_path_high_fps = reencode_video_with_diff_fps(video_path, self.tmp_path, 21.5, start_second, truncate_second)
        
        # read the video:
        cap = cv2.VideoCapture(video_path_low_fps)

        feat_batch_list = []
        video_feats = []
        first_frame = True
        pbar = tqdm(cap.get(7))
        i = 0
        while cap.isOpened():
            i += 1
            # pbar.set_description("Processing Frames: {} Total: {}".format(i, cap.get(7)))
            frames_exists, rgb = cap.read()
            
            if first_frame:
                if not frames_exists:
                    continue
            first_frame = False

            if frames_exists:
                rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                rgb_tensor = self.img_transform(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
                feat_batch_list.append(rgb_tensor)      # 32 x 3 x 224 x 224
                
                # Forward:
                if len(feat_batch_list) == self.batch_size:
                    # Stage1 Model:
                    input_feats = torch.cat(feat_batch_list,0).unsqueeze(0).to(self.device)
                    contrastive_video_feats = self.stage1_model.encode_video(input_feats, normalize=True, pool=False)
                    video_feats.extend(contrastive_video_feats.detach().cpu().numpy())
                    feat_batch_list = []
            else:
                if len(feat_batch_list) != 0:
                    import pdb; pdb.set_trace()
                    input_feats = torch.cat(feat_batch_list,0).unsqueeze(0).to(self.device)
                    contrastive_video_feats = self.stage1_model.encode_video(input_feats, normalize=True, pool=False)
                    video_feats.extend(contrastive_video_feats.detach().cpu().numpy())
                cap.release()
                break
        
        video_contrastive_feats = np.concatenate(video_feats)
        return video_contrastive_feats, video_path_high_fps

    def get_video_feat(self, video_path, start_second=None, truncate_second=None, pool_type='max'):
        #### (1) load video first
        print("video_path", video_path)
        print("truncate second: ", truncate_second)
        # Load the video, change fps:
        video_path_low_fps = reencode_video_with_diff_fps(video_path, self.tmp_path, self.fps, start_second, truncate_second)
        # video_path_high_fps = reencode_video_with_diff_fps(video_path, self.tmp_path, 21.5, start_second, truncate_second)
        
        # read the video:
        cap = cv2.VideoCapture(video_path_low_fps)
        feat_batch_list = []
        video_feats_list = []
        first_frame = True
        pbar = tqdm(cap.get(7))
        i = 0
        while cap.isOpened():
            i += 1
            pbar.set_description("Processing Frames: {} Total: {}".format(i, cap.get(7)))
            frames_exists, rgb = cap.read()
            
            if first_frame:
                if not frames_exists:
                    continue
            first_frame = False

            if frames_exists:
                rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                rgb_tensor = self.img_transform(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
                feat_batch_list.append(rgb_tensor)      # 32 x 3 x 224 x 224
                # Forward:
                if len(feat_batch_list) == self.batch_size:
                    # Stage1 Model:
                    input_feats = torch.cat(feat_batch_list,0).unsqueeze(0).to(self.device)
                    if pool_type=='avg':
                        video_feats = self.stage1_model.encode_video(input_feats, normalize=False, pool=False)
                        video_feats = F.avg_pool1d(video_feats.pernute(0, 2, 1), kernel_size=16).squeeze(2)
                        video_feats = F.normalize(video_feats, dim=-1)
                    elif pool_type=='max':
                        video_feats = self.stage1_model.encode_video(input_feats, normalize=True, pool=True)
                        
                    video_feats_list.extend(video_feats.detach().cpu().numpy())
                    feat_batch_list = []
            else:
                if len(feat_batch_list) != 0:
                    input_feats = torch.cat(feat_batch_list,0).unsqueeze(0).to(self.device)
                    video_feats = self.stage1_model.encode_video(input_feats)
                    video_feats_list.extend(video_feats.detach().cpu().numpy())
                cap.release()
                break
        if len(video_feats_list)==0:
            import pdb; pdb.set_trace()
            return []
        video_contrastive_feats = np.concatenate(video_feats_list)
        return video_contrastive_feats


    @torch.no_grad()
    def compare_video_features(self, video1_path, video2_path, start_second=None, truncate_second=None, tmp_path="./tmp_folder", pool_type='max'):
        self.tmp_path = tmp_path
        vid_feat1 = self.get_video_feat(video1_path, start_second, truncate_second)
        if video1_path == video2_path:
            start_second = 10 - truncate_second
        vid_feat2 = self.get_video_feat(video2_path, start_second, truncate_second)
        import ipdb; ipdb.set_trace()
        if vid_feat1.ndim==2:
            vid_feat1 = torch.Tensor(vid_feat1.transpose())
            vid_feat2 = torch.Tensor(vid_feat2.transpose())
        cos_sim = F.cosine_similarity(vid_feat1, vid_feat2, eps=0)

        return cos_sim


    @torch.no_grad()
    def compare_audio_features(self, audio1_path, audio2_path, start_second=None, truncate_second=None, tmp_path="./tmp_folder", pool_type='max'):        
        self.tmp_path = tmp_path
        
        ### (1) load audio1
        mel_spec1 = self.wav_to_spec(audio1_path, start_second, truncate_second)
        ### (2) load audio2. If two files are identical, change the start second to load other part
        if audio1_path==audio2_path:
            start_second = -1
        mel_spec2 = self.wav_to_spec(audio2_path, start_second, truncate_second)
        if pool_type=='avg':
            spec_feat1 = self.stage1_model.encode_spec(mel_spec1, normalize=False, pool=False)
            spec_feat1 = F.avg_pool1d(spec_feat1.permute(0, 2, 1), kernel_size=16).squeeze(2)
            spec_feat1 = F.normalize(spec_feat1, dim=-1)
            spec_feat2 = self.stage1_model.encode_spec(mel_spec2, normalize=False, pool=False)
            spec_feat2 = F.avg_pool1d(spec_feat2.permute(0, 2, 1), kernel_size=16).squeeze(2)
            spec_feat2 = F.normalize(spec_feat2, dim=-1)
        elif pool_type=='max':
            spec_feat1 = self.stage1_model.encode_spec(mel_spec1, normalize=True, pool=True)
            spec_feat2 = self.stage1_model.encode_spec(mel_spec2, normalize=True, pool=True)
        
        spec_feat1 = spec_feat1.detach().cpu()
        spec_feat2 = spec_feat2.detach().cpu()

        if spec_feat1.ndim==2:
            spec_feat1 = spec_feat1.transpose()
            spec_feat2 = spec_feat2.transpose()
        
        cos_sim = F.cosine_similarity(spec_feat1, spec_feat2, eps=0)
        return cos_sim

    def wav_to_spec(self, audio_path, start_second=0, truncate_second=8.2):
        ### (2) load audio
        wav, sr_new = librosa.load(audio_path, sr=sr)
        wav = wav.reshape(-1)
        length = int(truncate_second * sr)
        if start_second==-1:
            start_idx = max(wav.shape[0] - length, 0)
        else:
            start_idx = int(start_second * sr)
        y = np.zeros(length)
        if wav.shape[0] < length:
            y[:len(wav)] = wav
        else:
            y = wav[start_idx:start_idx+length]
        # # wav:
        # y = y[ : length - 1]        # ensure: 640 spec
        
        mel_spec = np.expand_dims(self.audio_transform(y), axis=0)
        # print('\n',mel_spec.shape) # spec: B x Mel_num x T (1, 128, 513) for 8.2s
        mel_spec = torch.from_numpy(mel_spec).cuda()
        return mel_spec

    @torch.no_grad()
    def compare_av_features(self, video_path, audio_path, start_second=None, truncate_second=None, tmp_path="./tmp_folder"):
        """
        use max pool for extra contrastive
        """
        
        self.tmp_path = tmp_path

        #### (1) load video first
        print("video_path", video_path)
        print("truncate second: ", truncate_second)
        # Load the video, change fps:
        video_path_low_fps = reencode_video_with_diff_fps(video_path, self.tmp_path, self.fps, start_second, truncate_second)
        video_path_high_fps = reencode_video_with_diff_fps(video_path, self.tmp_path, 21.5, start_second, truncate_second)
        
        # read the video:
        cap = cv2.VideoCapture(video_path_low_fps)

        ### (2) load audio
        wav, sr_new = librosa.load(audio_path, sr=sr)
        wav = wav.reshape(-1)
        length = int(truncate_second * sr)
        y = np.zeros(length)
        if wav.shape[0] < length:
            y[:len(wav)] = wav
        else:
            y = wav[:length]
        # # wav:
        # y = y[ : length - 1]        # ensure: 640 spec

        mel_spec = np.expand_dims(self.audio_transform(y), axis=0)
        # print('\n',mel_spec.shape) # spec: B x Mel_num x T (1, 128, 513) for 8.2s
        mel_spec = torch.from_numpy(mel_spec).cuda()

        feat_batch_list = []
        video_feats_list = []
        audio_feats_list = []
        first_frame = True
        pbar = tqdm(cap.get(7))
        i = 0
        while cap.isOpened():
            i += 1
            pbar.set_description("Processing Frames: {} Total: {}".format(i, cap.get(7)))
            frames_exists, rgb = cap.read()
            
            if first_frame:
                if not frames_exists:
                    continue
            first_frame = False

            if frames_exists:
                rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                rgb_tensor = self.img_transform(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
                feat_batch_list.append(rgb_tensor)      # 32 x 3 x 224 x 224
                # Forward:
                if len(feat_batch_list) == self.batch_size:
                    # Stage1 Model:
                    input_feats = torch.cat(feat_batch_list,0).unsqueeze(0).to(self.device)
                    video_feats, spec_feats, logit_scale = self.stage1_model(input_feats, mel_spec, output_dict=False)
                    video_feats_list.extend(video_feats.detach().cpu().numpy())
                    audio_feats_list.extend(spec_feats.detach().cpu().numpy())
                    feat_batch_list = []
            else:
                if len(feat_batch_list) != 0:
                    pool=True
                    input_feats = torch.cat(feat_batch_list,0).unsqueeze(0).to(self.device)
                    video_feats, spec_feats, logit_scale = self.stage1_model(input_feats, mel_spec, pool=pool, output_dict=False)
                    video_feats_list.extend(video_feats.detach().cpu().numpy())
                    audio_feats_list.extend(spec_feats.detach().cpu().numpy())
                    # print(video_feats.shape, spec_feats.shape)
                cap.release()
                break
        if len(video_feats_list)==0:
            import pdb; pdb.set_trace()
            return []
        video_contrastive_feats = np.concatenate(video_feats_list)
        audio_contrastive_feats = np.concatenate(audio_feats_list)
        logit_scale = logit_scale.detach().cpu().numpy()
        if pool:
            # print(video_contrastive_feats.shape, audio_contrastive_feats.shape)
            video_contrastive_feats = np.transpose(video_contrastive_feats, (1,0))
            audio_contrastive_feats = np.transpose(audio_contrastive_feats, (1,0))
        
        # logits_per_video = logit_scale * video_contrastive_feats @ audio_contrastive_feats.T
        # logits_per_spec = logit_scale * audio_contrastive_feats @ video_contrastive_feats.T
        
        min_len = min(len(audio_contrastive_feats),len(video_contrastive_feats))

        aud_feat = torch.Tensor(audio_contrastive_feats[:min_len])
        vid_feat = torch.Tensor(video_contrastive_feats[:min_len])
        cos_sim = F.cosine_similarity(aud_feat, vid_feat, eps=0)
        # labels = torch.arange(min_len, dtype=torch.long)
        # extra_contrast_loss = (
        #                      F.cross_entropy(torch.Tensor(logits_per_video[:min_len]), labels) +
        #                      F.cross_entropy(torch.Tensor(logits_per_spec[:min_len]), labels)
        #              ) / 2


        return cos_sim



def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    # if len(m) > 0 and verbose:
    #     print("missing keys:")
    #     print(m)
    # if len(u) > 0 and verbose:
    #     print("unexpected keys:")
    #     print(u)
    model.cuda()
    model.eval()
    return model


def inverse_op(spec):
    sr = 22050
    n_fft = 1024
    fmin = 125
    fmax = 7600
    nmels = 80
    hoplen = 1024 // 4
    spec_power = 1

    # Inverse Transform
    spec = spec * 100 - 100
    spec = (spec + 20) / 20
    spec = 10 ** spec
    spec_out = librosa.feature.inverse.mel_to_stft(spec, sr = sr, n_fft = n_fft, fmin=fmin, fmax=fmax, power=spec_power)
    wav = librosa.griffinlim(spec_out, hop_length=hoplen)
    return wav


