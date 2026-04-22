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
import json
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
    sfx_ranges = []
    sfx_start = 0.0
    sfx_end = 0.0
    consecutive = 0.0 # 1frame = 0.192s
    
    for x in result:
        is_speech = (x['speech_prob'] > speech_threshold)
        is_music = (x['music_prob'] > music_threshold)
        if (not is_speech) and (not is_music):
            if consecutive == 0:
                sfx_start = float(x['start_time_s'])
                sfx_end = float(x['end_time_s'])
                consecutive += 0.192
            else: # speech_start is set
                sfx_end = float(x['end_time_s'])
                consecutive += 0.192
        else: # speech detected
            if consecutive > min_consecutive_s:
                sfx_ranges.append([sfx_start, sfx_end])
                # reset params
            sfx_start = 0.0
            sfx_end = 0.0
            consecutive = 0.0
            

    if consecutive > min_consecutive_s:
        sfx_ranges.append([sfx_start, sfx_end])
    if len(sfx_ranges)==0:
        return False
    else:
        # save sfx chunks
        for chunk_idx, sfx_range in enumerate(sfx_ranges):
            start_s, end_s = sfx_range
            start_frame, end_frame = int(start_s*sr), int(end_s*sr)
            try:
                chunk = audio[start_frame:end_frame]
            except:
                import pdb; pdb.set_trace()
            sf.write(filename_base+f'_chunk{chunk_idx+1}.wav', chunk, sr)
        return True

        


def main_video(frames_dir, audio_csv_dir, save_dir, min_consecutive_s):
    directories = os.listdir(audio_csv_dir)
    for dirname in directories:
        os.makedirs(os.path.join(save_dir, dirname), exist_ok=True)
    audio_csv_list = glob.glob(os.path.join(audio_csv_dir, '*/*.csv'))
    for audio_csv in tqdm.tqdm(audio_csv_list):
        filename_base = os.path.basename(audio_csv).replace('.csv', '')
        json_save_dir = os.path.join(save_dir, os.path.dirname(audio_csv).replace(audio_csv_dir+'/', ''))
        
        with open(audio_csv, 'r') as f:
            result = csv.DictReader(f)
            sfx_ranges = []
            sfx_start = 0.0
            sfx_end = 0.0
            consecutive = 0.0 # 1frame = 0.192s

            for x in result:
                is_speech = (float(x['speech_prob']) > speech_threshold)
                is_music = (float(x['music_prob']) > music_threshold)
                if (not is_speech) and (not is_music):
                    if consecutive == 0:
                        sfx_start = float(x['start_time_s'])
                        sfx_end = float(x['end_time_s'])
                        consecutive += 0.192
                    else: # speech_start is set
                        sfx_end = float(x['end_time_s'])
                        consecutive += 0.192
                else: # speech detected
                    if consecutive > min_consecutive_s:
                        sfx_ranges.append([sfx_start, sfx_end])
                        # reset params
                    sfx_start = 0.0
                    sfx_end = 0.0
                    consecutive = 0.0
                    
            if consecutive > min_consecutive_s:
                sfx_ranges.append([sfx_start, sfx_end])
            if len(sfx_ranges)==0:
                continue
            else:
                # save sfx chunks
                save_path = f'{json_save_dir}/{filename_base}.json'
                chunks_dict = {}
                for chunk_idx, sfx_range in enumerate(sfx_ranges):
                    start_s, end_s = sfx_range
                    start_frame, end_frame = int(start_s*25)+1, int(end_s*25)+1
                    chunks_dict[f'chunk{chunk_idx+1}'] = []
                    
                    for frame_num in range(start_frame, end_frame):
                        vid_path = os.path.join(frames_dir,filename_base, f'{frame_num:03d}.jpg')
                        if not os.path.exists(vid_path):
                            with open("./no_frame.txt", 'a+') as noframe_f:
                                noframe_f.write(f"{vid_path}\n")
                        chunks_dict[f'chunk{chunk_idx+1}'].append(vid_path)
                with open(save_path, 'w') as json_f:
                    json.dump(chunks_dict, json_f, indent=4)
        

            
        

        

def main(sr, sfx_dir, output_dir, filtered_csv_dir):
    splits = ['train', 'test']
    

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    smd = SMDetector(pseudo_model_path)

    audio_dir = 'audio'
    csv_dir = 'csv'

    for split in splits:
        csvpath = os.path.join(filtered_csv_dir, f'{split}.csv')
        os.makedirs(os.path.join(output_dir, csv_dir, split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, audio_dir, split), exist_ok=True)
        print(f"Saving results in {os.path.join(output_dir, audio_dir, split)}")
        with open(csvpath, 'r') as csvfile:
            csvdata = csv.reader(csvfile)
            for row in tqdm.tqdm(csvdata):
                dirname = row[0].replace('.mp4', '')
                audio_path = os.path.join(sfx_dir, dirname+'.wav')
                audio, loaded_sr = librosa.load(audio_path, sr=None, mono=True)
                file_result = smd.predict_audio(audio_path=audio_path, sr=sr) # get results for each channel
                if file_result is None:
                    continue
                
                
                result_csv_filename = os.path.join(output_dir, csv_dir, split, dirname+'.csv')
                export_result(result_csv_filename, file_result, format_type='csv_prob')
                aud_name_base = os.path.join(output_dir, audio_dir, split, dirname)
                save_audio(audio, aud_name_base, file_result, sr=sr)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--audios_dir', required=True, default='/path/to/VGGSound/audio' type=str, help='Path to audios directory (16kHz, mono)') 
    parser.add_argument('--frames_dir', required=True, default='/path/to/VGGSound/frames' type=str, help='Path to frames directory') 
    parser.add_argument('--filtered_category_csv_dir', default='vggsound_filtered', type=str, help='Path to directory containing filtered category csv files') 
    ### output directory
    parser.add_argument('--output_dir', required=True, type=str, help='Path to save filtered audios') 
    ### hyperparameters
    parser.add_argument('--sr', default=16000, type=int)
    parser.add_argument('--min_consecutive_s', default=5.0, type=float, help='Minimum consecutive seconds of silence') # default=5.0
    args = parser.parse_args()

    print(f"Path to save filtered audios & video json files: {args.output_dir}")

    ### Step 1. Trim audios and save them to args.output_dir
    main(args.sr, args.audios_dir, args.output_dir, args.filtered_category_csv_dir)
    ### Step 2. Save json files to load frame paths for each chunk
    main_video(args.frames_dir, args.filtered_category_csv_dir, os.path.join(args.output_dir, 'video_json'), args.min_consecutive_s)