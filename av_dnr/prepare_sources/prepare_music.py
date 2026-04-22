import os, glob, tqdm

import numpy as np
import librosa
import torch
import torchvision.transforms as T
import torchaudio
from pcen import PCENTransform
import tqdm
import CRNN
import argparse
import csv
import soundfile as sf

from inference import SMDetector, export_result

# sr = 16000
n_fft = 1024
hop_size = 512
n_features = 128
duration = 20

music_threshold = 0.5
speech_threshold = 0.5
here = os.path.dirname(os.path.abspath(__file__))
pseudo_model_path = os.path.join(here, 'models', 'TVSM-pseudo', 'epoch=28-step=67192.ckpt.torch.pt')

        
def save_audio(audio, filename_base, result, min_consecutive_s = 5.0, sr=16000):
    music_ranges = []
    non_speech_start = 0.0
    non_speech_end = 0.0
    consecutive = 0.0 # 1frame = 0.192s
    
    for x in result:
        # start_times.append(x['start_time_s'])
        # end_times.append(x['end_time_s'])
        # music_prob.append(x['music_prob'])
        # speech_prob.append(x['speech_prob'])
        is_speech = (x['speech_prob'] > speech_threshold)
        if not is_speech:
            if consecutive == 0:
                non_speech_start = float(x['start_time_s'])
                non_speech_end = float(x['end_time_s'])
                consecutive += 0.192
            else: # speech_start is set
                non_speech_end = float(x['end_time_s'])
                consecutive += 0.192
        else: # speech detected
            if consecutive > min_consecutive_s:
                music_ranges.append([non_speech_start, non_speech_end])
                # reset params
            non_speech_start = 0.0
            non_speech_end = 0.0
            consecutive = 0.0
            

    if consecutive > min_consecutive_s:
        music_ranges.append([non_speech_start, non_speech_end])
    if len(music_ranges)==0:
        return False
    else:
        # save music chunks
        for chunk_idx, music_range in enumerate(music_ranges):
            start_s, end_s = music_range
            start_frame, end_frame = int(start_s*sr), int(end_s*sr)
            try:
                chunk = audio[start_frame:end_frame]
            except:
                import pdb; pdb.set_trace()
            sf.write(filename_base+f'_chunk{chunk_idx+1}.wav', chunk, sr)
        return True

        



        

def main(sr, music_dir, output_dir):
    music_list = glob.glob(os.path.join(music_dir, '*/*.mp3'))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    smd = SMDetector(pseudo_model_path)

    for mpath in tqdm.tqdm(music_list, dynamic_ncols=True):
        results, y_list = smd.predict_audio_fma(mpath, sr=sr) # get results for each channel
        if results is None:
            continue
        filenum = os.path.basename(mpath).replace('.mp3', '')
        dirname = filenum[:3]

        if sr==44100:
            audio_dir = 'audio_44k'
            csv_dir = 'csv_44k'
        elif sr==16000:
            audio_dir = 'audio_16k'
            csv_dir = 'csv_16k'
        os.makedirs(os.path.join(output_dir, csv_dir, dirname), exist_ok=True)
        os.makedirs(os.path.join(output_dir, audio_dir, dirname), exist_ok=True)
        for i, file_result in enumerate(results):
            audio = y_list[i]
            result_csv_filename = os.path.join(output_dir, csv_dir, dirname, filenum+f'_ch{i+1}' + '.csv')
            export_result(result_csv_filename, file_result, format_type='csv_prob')
            aud_name_base = os.path.join(output_dir, audio_dir, dirname, filenum+f'_ch{i+1}')
            save_audio(audio, aud_name_base, file_result, sr=sr)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sr', default=16000, type=int)
    parser.add_argument('--music_dir', required=True, default='/path/to/FMA/medium' type=str, help='Path to FMA medium dataset')
    parser.add_argument('--output_dir', required=True, default='fma_medium_filtered' type=str, help='Path to save filtered audios')
    args = parser.parse_args()
    print(f"Sampling rate of saved audio: {args.sr}")
    print(f"Path to FMA medium dataset: {args.music_dir}")
    print(f"Path to save filtered audios: {args.output_dir}")
    main(args.sr, args.music_dir, args.output_dir)