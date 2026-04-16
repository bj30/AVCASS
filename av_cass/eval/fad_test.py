
import torchaudio
import numpy as np
import os
import fnmatch
import pathlib
import torch


import soundfile as sf
import tqdm
import glob
import json
import pandas as pd
import torchmetrics
import csv

from collections import defaultdict
torch.multiprocessing.set_sharing_strategy('file_system')




def create_symlinks(source_dir, target_dir, type_of_dataset):
    """
    Search for all WAV files matching a specific name pattern in the source directory 
    recursively and create symbolic links in the target directory.

    :param source_dir: Directory to search for WAV files
    :param target_dir: Directory where symbolic links will be created
    :param file_pattern: Pattern to match file names (e.g., "*.wav" or "sample*.wav")
    """
    # Ensure target directory exists
    pathlib.Path(target_dir).mkdir(parents=True, exist_ok=True)
    
    # target_stems = ['mix.wav', 'speech.wav', 'sfx.wav', 'music.wav']
    target_stems_dict = {
        'speech.wav': 'speech',
        'sfx.wav': 'sfx',
        'music.wav': 'music',
        'effects.wav': 'sfx',
        'speech.flac': 'speech',
        'sfx.flac': 'sfx',
        'music.flac': 'music',
        'pred_speech.wav': 'speech',
        'pred_sfx.wav': 'sfx',
        'pred_music.wav': 'music'
    }

    
    step_count = {stem: 0 for stem in target_stems_dict.keys()}
    if type_of_dataset == "avdnr" or type_of_dataset == "avdnr_test":
        for parts in os.listdir(source_dir):
            # skip if we saved some other files in the source directory
            if not os.path.isdir(os.path.join(source_dir, parts)):
                continue

            for sample_idx in os.listdir(os.path.join(source_dir, parts)):
                
                for file in os.listdir(os.path.join(source_dir, parts, sample_idx)):
                    # skip if file not our target stems
                    if file not in target_stems_dict.keys():
                        continue

                    source_file = os.path.join(source_dir, parts, sample_idx, file)
                    ori_extention = file.split(".")[-1]
                    link_filename = f"{parts}-{sample_idx}.{ori_extention}"

                    link_root = os.path.join(target_dir, target_stems_dict[file])
                    os.makedirs(link_root, exist_ok=True)
                    link_path = os.path.join(link_root, link_filename)
                    
                    os.symlink(source_file, link_path)

                    step_count[file] += 1
    
    elif type_of_dataset in ["dnrv2", "dnrv2_test", "dnrv1", "dnrv1_test", "dnrv3", "dnrv3_test", "dnrv2_16k", "dnrv2_16k_test"]:
        if type_of_dataset=="dnrv1_test":
            import pdb; pdb.set_trace()
        for sample_idx in os.listdir(source_dir):
            # skip if we saved some other files in the source directory
            if not os.path.isdir(os.path.join(source_dir, sample_idx)):
                continue

            for file in os.listdir(os.path.join(source_dir, sample_idx)):
                # skip if file not our target stems
                if file not in target_stems_dict.keys():
                    continue

                source_file = os.path.join(source_dir, sample_idx, file)
                ori_extention = file.split(".")[-1]
                link_filename = f"{sample_idx}.{ori_extention}"

                link_root = os.path.join(target_dir, target_stems_dict[file])
                os.makedirs(link_root, exist_ok=True)
                link_path = os.path.join(link_root, link_filename)
                
                os.symlink(source_file, link_path)

                step_count[file] += 1

    print(f"Created symlinks for {source_dir} to {target_dir}")
    print(step_count)

def prepare_symlinks(testing_folders):
    # Usage
    for source_directory, target_directory, type_of_dataset in testing_folders:
        if os.path.exists(target_directory):
            print(f"Directory {target_directory} already exists. Skip creating symlinks.")
            continue

        assert type_of_dataset in ["dnrv3", "dnrv2", "avdnr", "dnrv1", "dnrv2_16k", "dnrv2_16k_test", "dnrv3_test", "dnrv2_test", "avdnr_test", "dnrv1_test"], f"type_of_dataset must be either 'dnrv3', 'dnrv2' or 'avdnr'. Got '{type_of_dataset}'"
        create_symlinks(source_directory, target_directory, type_of_dataset)



# create a simple dataloader
class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, source_path, target_path, load_mixture, type_of_dataset, stem):
        self.source_path = source_path
        self.target_path = target_path
        self.stem = stem
        self.source_files = os.listdir(source_path)
        self.load_mixture = load_mixture
        self.mixture_name = "mix.wav" if type_of_dataset in ["dnrv2", "dnrv1"] else "mixture.wav"

    def __len__(self):
        return len(self.source_files)
    
    def __getitem__(self, idx):
        source_file = self.source_files[idx]
        file_name = source_file.split(".")[0]

        with sf.SoundFile(os.path.join(self.source_path, source_file)) as f:
            source_audio = f.read(dtype='float32')
            sr = f.samplerate
        # source_audio, sr = torchaudio.load(os.path.join(self.source_path, source_file))
        assert sr == 16000, f"Sample rate of {source_file} is not 16000"
        
        with sf.SoundFile(os.path.join(self.target_path, f"{file_name}.wav")) as f:
            target_audio = f.read(dtype='float32')
            sr = f.samplerate
        # target_audio, sr = torchaudio.load(os.path.join(self.target_path, source_file))
        assert sr == 16000, f"Sample rate of {os.path.join(self.target_path, file_name+'.wav')} is not 16000"

        if self.load_mixture:
            real_path = os.path.realpath(os.path.join(self.target_path, source_file))
            mixture_audio, sr = torchaudio.load(real_path.replace(f"{self.stem}.wav", self.mixture_name))
            return source_audio, target_audio, mixture_audio, source_file
        else:
            return source_audio, target_audio, source_file
    

def main(args):
    if args.where == "mmai":
        target_directory_root = "/mnt/bear2/users/syun/msdm_logs/symlinks_for_fad_iccv"
        # root_avdnr_dataset_path = "/mnt/lynx1/datasets/AVDnR/test"
        root_avdnr_dataset_path = "/mnt/lynx3/users/syun/AVDnR/test" # fixedAVDnR
        root_dnrv2_dataset_path = "/mnt/lynx1/datasets/dnr_v2_official/tt"
        root_dnrv2_16k_dataset_path = "/mnt/lynx1/datasets/dnr_v2_16k/tt"
        root_dnrv3_dataset_path = "/mnt/lynx2/datasets/AV-DnR/DnRv3/eng/wav_16k/val"
        root_dnrv1_dataset_path = "/mnt/lynx2/datasets/DnR-compiled/eval"
    elif args.where == "nax":
        target_directory_root = "symlinks_for_fad_iccv"
        root_avdnr_dataset_path = "/public/home/nax/datasets/Audio/AVDnR/AVDnR/test"

    dnr_v1_test_path = os.path.join(target_directory_root, 'DnR-compiled-eval')
    dnr_v3_vgg_v2_test_path = os.path.join(target_directory_root, 'dnrv3-vgg-v2-test')
    dnr_v3_eng_test_path = os.path.join(target_directory_root, 'dnrv3_eng_val')

    # avdnr_test_path = os.path.join(target_directory_root, 'AV-DnR-test')
    avdnr_test_path = os.path.join(target_directory_root, 'AV-DnR-test-cvpr26')
    dnrv2_test_path = os.path.join(target_directory_root, 'DnRv2-test')
    dnrv3_test_path = os.path.join(target_directory_root, 'DnRv3-test')
    dnrv2_16k_test_path = os.path.join(target_directory_root, 'DnRv2-test-16k')
    dnrv1_test_path = os.path.join(target_directory_root, 'DnRv1-test')

    testing_folders = [
        # example
        # [
        #     source_dir_for_predicted, 
        #     target_dir_for_symlinks, 
        #     type_of_dataset(dnrv3, dnrv2, avdnr) 
        # ],

        # AVDnR test set
        [
            root_avdnr_dataset_path,
            avdnr_test_path,
            "avdnr_test",
        ],
        [
            root_dnrv1_dataset_path,
            dnrv1_test_path,
            "dnrv1_test"    
        ],
        # # DnRv2 test set
        # [
        #     "/mnt/lynx1/datasets/dnr_v2_official/tt",
        #     dnrv2_test_path,
        #     "dnrv2_test",
        # ],
        [
            "/mnt/lynx1/datasets/dnr_v2_16k/tt",
            dnrv2_16k_test_path,
            "dnrv2_16k_test",
        ],
        # # DnRv3 test set
        [
            "/mnt/lynx2/datasets/AV-DnR/DnRv3/eng/wav_16k/val",
            dnrv3_test_path,
            "dnrv3_test",
        ],

        # for BANDIT model
        # [
        #     "",
        #     os.path.join(target_directory_root, 'BANDIT_AVDnR_s16le'),
        #     "avdnr",
        # ],
        # [
        #     "/home/zhang/workspace/MSDM-cocktail-fork-separation/.logs/log_zk/testing/bandit_avdnr",
        #     os.path.join(target_directory_root, 'BANDIT_AVDnR_zk_inference'),
        #     "avdnr",
        # ],

        # # for hdemucs v3
        # [
        #     "/mnt/bear2/users/syun/25CVPR/main/demucs-v3/outputs/fixed_avdnr/epoch11",
        #     os.path.join(target_directory_root, 'DEMUCSV3_AVDnR'),
        #     "avdnr",
        # ],

        # # for hdemucs v4
        # [
        #     "/mnt/bear2/users/syun/25CVPR/main/demucs-v4/outputs/fixed_avdnr/epoch11",
        #     os.path.join(target_directory_root, 'DEMUCSV4_AVDnR'),
        #     "avdnr",
        # ],

        # # for Ours model
        # [
        #     "/home/zhang/workspace/MSDM-cocktail-fork-separation/.logs/log_sy_spec/testing/spec_resume_fixedAVDnR_ep82",
        #     os.path.join(target_directory_root, 'Ours_audio_only_ep82_AVDnR'),
        #     "avdnr",
        # ],
        # [
        #     "/home/zhang/workspace/MSDM-cocktail-fork-separation/.logs/log_sy_spec/testing/spec_resume_fixedAVDnR_ep112",
        #     os.path.join(target_directory_root, 'Ours_audio_only_ep112_AVDnR'),
        #     "avdnr",
        # ],
        # [
        #     "/home/zhang/workspace/MSDM-cocktail-fork-separation/.logs/log_zk/testing/spec_resume_fixedAVDnR-ep115",
        #     os.path.join(target_directory_root, 'Ours_audio_only_ep115_AVDnR_zk_test'),
        #     "avdnr",
        # ],

        # [
        #     "/home/zhang/workspace/MSDM-cocktail-fork-separation/.logs/log_zk/testing/spec_resume_fixedAVDnR-ep24-save_flac-small",
        #     os.path.join(target_directory_root, 'Ours_audio_only_ep24_flac_AVDnR_zk_test'),
        #     "avdnr",
        # ],
    ]

    if os.path.exists(args.testing_dir_csv):
        with open(args.testing_dir_csv, 'r') as f:
            reader = csv.reader(f)
            local_testing_dirs = list(reader)
        print(local_testing_dirs)
        for local_testing_dir in local_testing_dirs:
            testing_folders.append(
                [
                    local_testing_dir[0],
                    os.path.join(target_directory_root, local_testing_dir[1]),
                    local_testing_dir[2],
                ]
            )


    prepare_symlinks(testing_folders)
    print("\nSymlinks created. Start evaluation...\n")

    for _, target_directory, type_of_dataset in testing_folders:
        if type_of_dataset in ["avdnr_test", "dnrv2_test", "dnrv2_16k_test", "dnrv3_test", "dnrv1_test"]:
            continue

        for stem in ['speech', 'sfx', 'music']:
            source_path = os.path.join(target_directory, stem)
            if type_of_dataset == "avdnr":
                gt_path_root = avdnr_test_path
            elif type_of_dataset == "dnrv2":
                gt_path_root = dnrv2_test_path
            elif type_of_dataset == "dnrv2_16k":
                gt_path_root = dnrv2_16k_test_path
            elif type_of_dataset == "dnrv3":
                gt_path_root = dnrv3_test_path
            elif type_of_dataset == "dnrv1":
                gt_path_root = dnrv1_test_path
            else:
                raise ValueError(f"Unknown type_of_dataset: {type_of_dataset}")
            target_path = os.path.join(gt_path_root, stem)
            
            print(f"Source: {source_path}")
            print(f"Target: {target_path}")

            # GPU acceleration is preferred
            device = torch.device(f"cuda:{0}")

            # check if the folder already been evaluated
            # scan the source_path to see if there exists xxx_{stem}.json
            if len(glob.glob(os.path.join(target_directory, f"*{stem}.json"))) > 0:
                print(f"Already evaluated {stem} for FAD at {glob.glob(os.path.join(target_directory, f'*{stem}.json'))}. Skip.")
            else:
                # Initialize a helper instance
                from audioldm_eval import EvaluationHelper
                backbone="cnn14" # `cnn14` refers to PANNs model, `mert` refers to MERT model
                evaluator = EvaluationHelper(
                    16000, 
                    device,
                    backbone=backbone, # `cnn14` refers to PANNs model, `mert` refers to MERT model
                    )
                
                # Perform evaluation, result will be print out and saved as json
                metrics = evaluator.main(
                    source_path,
                    target_path,
                    limit_num=None # If you only intend to evaluate X (int) pairs of data, set limit_num=X
                )

            # frechet = FrechetAudioDistance(
            #     model_name="vggish",
            #     sample_rate=16000,
            #     use_pca=False, 
            #     use_activation=False,
            #     verbose=False
            # )
            # fad_score = frechet.score(
            #     target_path, 
            #     source_path, 
            #     dtype="float32"
            # )

            # with open(os.path.join(target_directory, "fad_vggish_results.txt"), 'a') as f:
            #     f.write(f"fad_vggish: {fad_score}\n")


            # metric calculation for SDR, SNR, SI-SNR, SI-SDR
            source_results_save_path = os.path.join(target_directory, f"{stem}_source_results.json")
            target_results_save_path = os.path.join(gt_path_root, f"{stem}_target_results.json")

            if os.path.exists(source_results_save_path):
                print(f"Already evaluated {stem} for general metrics. Skip.")
            else:
                from torchmetrics.functional.audio import signal_distortion_ratio, signal_noise_ratio, scale_invariant_signal_noise_ratio, scale_invariant_signal_distortion_ratio

                if not os.path.exists(target_results_save_path):
                    load_mixture = True
                else:
                    load_mixture = False

                dataset = SimpleDataset(source_path, target_path, load_mixture, type_of_dataset, stem)
                data_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False, num_workers=16)

                n_samples = 0
                metrics_dict = {
                    "sdr": 0,
                    "snr": 0,
                    "sisnr": 0,
                    "sisdr": 0,
                    "mixture_sdr": 0,
                    "mixture_snr": 0,
                    "mixture_sisnr": 0,
                    "mixture_sisdr": 0,
                }

                sdr_dict = defaultdict(list)

                for batch in tqdm.tqdm(data_loader, desc="Testing general metrics"):
                    if load_mixture:
                        source_audio, target_audio, mixture_audio, source_file = batch
                        mixture_audio = mixture_audio.to(device)
                    else:
                        source_audio, target_audio, source_file = batch
                        mixture_audio = None
                    source_audio = source_audio.to(device)
                    target_audio = target_audio.to(device)

                    sdr = signal_distortion_ratio(source_audio, target_audio)
                    metrics_dict['sdr'] += sdr.sum().cpu().item()
                    snr = signal_noise_ratio(source_audio, target_audio)
                    metrics_dict['snr'] += snr.sum().cpu().item()
                    sisnr = scale_invariant_signal_noise_ratio(source_audio, target_audio)
                    metrics_dict['sisnr'] += sisnr.sum().cpu().item()
                    sisdr = scale_invariant_signal_distortion_ratio(source_audio, target_audio)
                    metrics_dict['sisdr'] += sisdr.sum().cpu().item()

                    if not os.path.exists(target_results_save_path):
                        # print(mixture_audio.shape, target_audio.shape)
                        if mixture_audio.ndim==3:
                            mixture_audio = mixture_audio.squeeze(1)
                        if target_audio.ndim==3:
                            target_audio = target_audio.squeeze(1)
                        sdr_mixture = signal_distortion_ratio(mixture_audio, target_audio)
                        metrics_dict['mixture_sdr'] += sdr_mixture.sum().cpu().item()
                        snr_mixture = signal_noise_ratio(mixture_audio, target_audio)
                        metrics_dict['mixture_snr'] += snr_mixture.sum().cpu().item()
                        sisnr_mixture = scale_invariant_signal_noise_ratio(mixture_audio, target_audio)
                        metrics_dict['mixture_sisnr'] += sisnr_mixture.sum().cpu().item()
                        sisdr_mixture = scale_invariant_signal_distortion_ratio(mixture_audio, target_audio)
                        metrics_dict['mixture_sisdr'] += sisdr_mixture.sum().cpu().item()

                    n_samples += source_audio.size(0)

                    sdr_dict['name'].extend(source_file)
                    sdr_dict['snr'].extend(snr.cpu().tolist())

                assert n_samples == len(dataset), f"n_samples: {n_samples}, len(dataset): {len(dataset)}"

                # save sdr_dict into a csv file
                df = pd.DataFrame(sdr_dict)
                df.to_csv(os.path.join(target_directory, f"{stem}_sdr.csv"))

                metrics_dict["sdr"] /= n_samples
                metrics_dict["snr"] /= n_samples
                metrics_dict["sisnr"] /= n_samples
                metrics_dict["sisdr"] /= n_samples

                if not os.path.exists(target_results_save_path):
                    metrics_dict["mixture_sdr"] /= n_samples
                    metrics_dict["mixture_snr"] /= n_samples
                    metrics_dict["mixture_sisnr"] /= n_samples
                    metrics_dict["mixture_sisdr"] /= n_samples

                    mixture_metrics_dict = {
                        "sdr": metrics_dict["mixture_sdr"],
                        "snr": metrics_dict["mixture_snr"],
                        "sisnr": metrics_dict["mixture_sisnr"],
                        "sisdr": metrics_dict["mixture_sisdr"],
                    }
                    with open(target_results_save_path, 'w') as f:
                        json.dump(mixture_metrics_dict, f, indent=4)
                else:
                    with open(target_results_save_path, 'r') as f:
                        mixture_metrics_dict = json.load(f)
                
                metrics_dict['sdri'] = metrics_dict['sdr'] - mixture_metrics_dict['sdr']
                metrics_dict['snri'] = metrics_dict['snr'] - mixture_metrics_dict['snr']
                metrics_dict['sisnri'] = metrics_dict['sisnr'] - mixture_metrics_dict['sisnr']
                metrics_dict['sisdri'] = metrics_dict['sisdr'] - mixture_metrics_dict['sisdr']

                with open(source_results_save_path, 'w') as f:
                    json.dump(metrics_dict, f, indent=4)


            if stem == "speech" and not os.path.exists(os.path.join(target_directory, "pesq_results.txt")):
                from pesq import pesq, pesq_batch
                # batch mode
                pesq_score_total = 0
                num_files = 0
                batch_size = 64
                workers = 32
                source_files = os.listdir(source_path)

                dataset = SimpleDataset(source_path, target_path, False, type_of_dataset, stem)
                data_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False, num_workers=16)


                for source_audio, target_audio, source_file in tqdm.tqdm(data_loader, desc="Testing PESQ"):
                    pred_audios = source_audio.numpy()
                    gt_audios = target_audio.numpy()

                    pesq_score = pesq_batch(16000, gt_audios, pred_audios, 'wb', n_processor=workers)
                    num_files += pred_audios.shape[0]
                    pesq_score_total += sum(pesq_score)

                # for source_files_batch in tqdm.tqdm([source_files[i:i+batch_size] for i in range(0, len(source_files), batch_size)], desc="Testing PESQ"):
                #     source_aud_paths = [os.path.join(source_path, source_file) for source_file in source_files_batch]
                #     target_aud_paths = [os.path.join(target_path, source_file) for source_file in source_files_batch]
                #     pred_audios_list = []
                #     gt_audios_list = []
                #     for source_aud_path, target_aud_path in zip(source_aud_paths, target_aud_paths):
                #         pred_audio, sr = sf.read(source_aud_path)
                #         assert sr == 16000, f"Sample rate of {source_aud_path} is not 16000"
                #         gt_audio, sr = sf.read(target_aud_path)
                #         assert sr == 16000, f"Sample rate of {target_aud_path} is not 16000"
                #         pred_audios_list.append(pred_audio)
                #         gt_audios_list.append(gt_audio)
                #     pred_audios = np.stack(pred_audios_list)
                #     gt_audios = np.stack(gt_audios_list)
                #     pesq_score = pesq_batch(16000, gt_audios, pred_audios, 'wb', n_processor=workers)
                #     num_files += len(source_files_batch)
                #     pesq_score_total += sum(pesq_score)
                
                print(f"PESQ: {pesq_score_total / num_files}, num_files: {num_files}")
                with open(os.path.join(target_directory, "pesq_results.txt"), 'a') as f:
                    f.write(f"PESQ: {pesq_score_total / num_files}, num_files: {num_files}\n")
    
    # aggregate all results "avdnr_test", "dnrv2_test", "dnrv3_test"
    all_results = {
        "avdnr": defaultdict(list),
        "dnrv2": defaultdict(list),
        "dnrv3": defaultdict(list),
        "dnrv2_16k": defaultdict(list),
        "dnrv1": defaultdict(list),
    }

    metrics_to_show = [
        "speech-frechet_audio_distance",
        "sfx-frechet_audio_distance",
        "music-frechet_audio_distance",
        "speech-pesq",
        "speech-sisdri",
        "sfx-sisdri",
        "music-sisdri",
        "speech-kullback_leibler_divergence_sigmoid",
        "sfx-kullback_leibler_divergence_sigmoid",
        "music-kullback_leibler_divergence_sigmoid",
        "speech-frechet_distance",
        "sfx-frechet_distance",
        "music-frechet_distance",
    ]

    for _, target_directory, type_of_dataset in testing_folders:
        if type_of_dataset in ["avdnr_test", "dnrv2_test", "dnrv2_16k_test", "dnrv3_test", "dnrv1_test"]:
            continue

        model_name = os.path.basename(target_directory)
        all_results[type_of_dataset]['model_name'].append(model_name)
        
        exp_results = {}

        for stem in ['speech', 'sfx', 'music']:
            # load general results
            with open(os.path.join(target_directory, f"{stem}_source_results.json"), 'r') as f:
                general_results = json.load(f)
            
            # add into all_results
            for key, value in general_results.items():
                if f"{stem}-{key}" in metrics_to_show:
                    all_results[type_of_dataset][f"{stem}-{key}"].append(value)
        
            # load FAD results
            try:
                fad_path = glob.glob(os.path.join(target_directory, f"*{stem}.json"))[0]
            except:
                import pdb; pdb.set_trace()
            with open(fad_path, 'r') as f:
                fad_results = json.load(f)
            
            # add into all_results
            for key, value in fad_results.items():
                # if value is None or not isinstance(value, (int, float)):
                #     continue
                if f"{stem}-{key}" in metrics_to_show:
                    all_results[type_of_dataset][f"{stem}-{key}"].append(value)
            
            # load PESQ results
            if stem == "speech":
                with open(os.path.join(target_directory, "pesq_results.txt"), 'r') as f:
                    pesq_results = f.readlines()
                pesq_score = float(pesq_results[0].split(",")[0].split(":")[1].strip())

                if f"{stem}-pesq" in metrics_to_show:
                    all_results[type_of_dataset][f"{stem}-pesq"].append(pesq_score)
    
    # reorder the columns
    for type_of_dataset in ["avdnr", "dnrv2", "dnrv2_16k", "dnrv3", "dnrv1"]:
        if all_results[type_of_dataset] == {}:
            continue
        df = pd.DataFrame(all_results[type_of_dataset])
        df = df[["model_name"] + metrics_to_show]
        # for scalar metrics, round to 4 decimal places
        for col in df.columns:
            if col not in ["model_name"]:
                df[col] = df[col].apply(lambda x: round(x, 4))
        # make model_name has the same width
        df["model_name"] = df["model_name"].apply(lambda x: f"{x:_<50}")

        # sort by model_name
        df = df.sort_values(by="model_name")

        print(df)
        if 'trainDnRv2' in args.testing_dir_csv:
            df.to_csv(f"{type_of_dataset}_evaluation_results_trainDnRv2.csv")
        else:
            df.to_csv(f"{type_of_dataset}_evaluation_results2.csv")

    # # save all results into three csv files
    # for type_of_dataset in ["avdnr", "dnrv2", "dnrv3"]:
    #     df = pd.DataFrame(all_results[type_of_dataset])
    #     df.to_csv(f"{type_of_dataset}_evaluation_results.csv")

            





if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--where", type=str, default="mmai")
    parser.add_argument("--testing_dir_csv", type=str, default="local_testing_dirs.csv")
    args = parser.parse_args()
    
    # assert args.model_name in ["vggish", "pann", "clap", "encodec"], "model_name must be either 'vggish', 'pann', 'clap' or 'encodec'"
    
    # if args.model_name == "vggish":
    #     sample_rate = 16000
    # elif args.model_name == "pann":
    #     sample_rate = 16000
    # elif args.model_name == "clap":
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=4):
        main(args)
    
    
    
    # # sacan all json files end with `cnn14.json`
    # import glob
    # import json
    # import numpy as np
    # import pandas as pd
    # import matplotlib.pyplot as plt
    # import seaborn as sns
    # import os
    # import re
    # import shutil
    # import pathlib
    # import datetime
    # from collections import defaultdict
    # from itertools import product
    
    
    # root_dir = "./"
    # json_files = glob.glob(os.path.join(root_dir, "*cnn14.json"))
    
    # # load all json files and create a table for comparison
    # data = defaultdict(list)
    # for json_file in json_files:
    #     with open(json_file, 'r') as f:
    #         metrics = json.load(f)
    #         data['name'].append(json_file)
            
    #         if "DnR-compiled" in json_file:
    #             data['dataset'].append("DnR-compiled")
    #         elif "dnrv3-vgg-v2" in json_file:
    #             data['dataset'].append("dnrv3-vgg-v2")
    #         else:
    #             data['dataset'].append("unknown")
            
    #         if "speech" in json_file:
    #             data['stem'].append("speech")
    #         elif "sfx" in json_file:
    #             data['stem'].append("sfx")
    #         elif "music" in json_file:
    #             data['stem'].append("music")
    #         else:
    #             data['stem'].append("unknown")
            
    #         for key, value in metrics.items():
    #             data[key].append(value)

    # df = pd.DataFrame(data)
    # # sort the table by dataset first, then by stem
    # df = df.sort_values(by=['dataset', 'stem'])
    
    # # df = df.set_index('dataset')
    # # df = df.sort_index()
    # print(df)
    
    # # save the table into csv file
    # # df.to_csv("evaluation_results.csv")









# # back up previous testing pathes
#     source_directory_list = [
#         # "/home/zhang/workspace/bandit/banditlogs/bandit_dnrours",
#         # "/mnt/lynx2/datasets/DnR-compiled/eval",
#         # "/home/zhang/workspace/bandit/banditlogs/bandit_dnrours-dnrv3-vgg-v2",
#         # "/mnt/lynx2/datasets/AV-DnR/DnRv3-vgg-v2/test",
#         # "/mnt/bear2/users/syun/msdm_logs/log_sy/testing/RFM_ReFlow_logit_normal_DnR_8s_msdm_dnr_mixture_cond_bs64/newstep_4",
#         # "/mnt/bear2/users/syun/msdm_logs/log_sy/testing/AV_RFM_ReFlow_logit_normal_DnRv3-vgg-v2_frozen-avdiffuss-enc_resume_ep19+8_epoch62/newstep_4",
#         # "/mnt/bear2/users/syun/msdm_logs/log_sy/testing/RFM_ReFlow_logit_normal_DnRv3-vgg-v2_last_testset-DnRv3-vgg-v2",
#         # "/mnt/bear2/users/syun/dnr_outputs/pretrained_ckpt/dnrv1-batch1",
#         # "/mnt/bear2/users/syun/dnr_outputs/our_ckpt/DnRv3-vgg-v2-test",
#         # "/mnt/bear2/users/syun/msdm_logs/log_zk/testing/AV_mode_RFM_DnRv3-vgg-v2-SSLAlignment-frozen-feature_embedding/ema_sampler_test/step_4"
#         # "/home/zhang/workspace/MSDM-cocktail-fork-separation/.logs/log_zk/testing/train_post_RFM_denoise_bsrnn_cond/ep_09_step_4_no_scale"
        
#         # DnRv3 eng
#         # "/mnt/lynx2/datasets/AV-DnR/DnRv3/eng/wav_16k/val",
        
#         # for our model
#         # "/mnt/bear2/users/syun/msdm_logs/log_zk/testing/Post_RFM_denoise_bsrnn_cond/step_5_trainours_testours",
#         # "/mnt/bear2/users/syun/msdm_logs/log_zk/testing/Post_RFM_denoise_bsrnn_cond/step_5_trainours_testdnrv1",
#         # "/mnt/bear2/users/syun/msdm_logs/log_zk/testing/Post_RFM_denoise_bsrnn_cond/step_5_trainours_testdnrv3_eng",
#         "/mnt/bear2/users/syun/msdm_logs/log_zk/testing/RFM_post_denoise_bsrnn_cond_dnrv1/dnrv1_model_denoiser_step_5_trainours_testours",
        
#         # for our model, audio-visual facial video
#         # "/mnt/bear2/users/syun/msdm_logs/log_zk/testing/AV_SSLAlignment_Post_RFM_denoise_bsrnn_cond/AV_speech_step_5_trainours_testours",
#         # for our model, audio-visual sfx video
#         # "/mnt/bear2/users/syun/msdm_logs/log_zk/testing/AV_SSLAlignment_Post_RFM_denoise_bsrnn_cond/AV_sfx_step_5_trainours_testours",
        
#         # for our model, audio-visual AVDiffuSS encoder (facial video)
#         # (w/o PD)
#         # "/mnt/bear2/users/syun/msdm_logs/log_sy/testing/AV_RFM_ReFlow_logit_normal_DnRv3-vgg-v2_frozen-avdiffuss-enc_resume_ep19+8_epoch62/newstep_4",
#         # (w/ PD)
#         # "/mnt/bear2/users/syun/msdm_logs/log_zk/testing/AV_avdiffuss_Post_RFM_denoise_bsrnn_cond/AV_speech_step_5_trainours_testours",
        
#         # for bandit model
#         # "/home/zhang/workspace/MSDM-cocktail-fork-separation/.logs/log_zk/testing/bandit_dnrv3-vgg-v2/sec_6",
#         # "/home/zhang/workspace/MSDM-cocktail-fork-separation/.logs/log_zk/testing/bandit_dnrv3-vgg-v2/dnrv3_dataset",
#         # "/home/zhang/workspace/MSDM-cocktail-fork-separation/.logs/log_zk/testing/bandit_dnrv3-vgg-v2/dnrv1_dataset",
#         # "/mnt/bear2/users/syun/msdm_logs/log_zk/testing/bandit_dnrv1/train_dnrv1_test_dnrv1",

#         # for MRX model
#         # "/mnt/bear2/users/syun/dnr_outputs/our_ckpt/dnrv1",
#         # "/mnt/bear2/users/syun/dnr_outputs/our_ckpt/DnRv3-vgg-v2-test",
#         # "/mnt/bear2/users/syun/dnr_outputs/our_ckpt/dnrv3-eng/val",
#         # "/mnt/bear2/users/syun/msdm_logs/log_sy/testing_interest/MRX-testDnRv1",
        
#         # for Demucs-v3
#         # (trained on DnRv1)
#         # "/mnt/bear2/users/syun/25CVPR/main/wavs/demucs/dnrv1",
#         # (trained on AVDnR)
#         # "/mnt/bear2/users/syun/25CVPR/main/demucs-v3/outputs/dnrv1/epoch11",
#         # "/mnt/bear2/users/syun/25CVPR/main/demucs-v3/outputs/avdnr/epoch11",
#         # "/mnt/bear2/users/syun/25CVPR/main/demucs-v3/outputs/dnrv3/epoch11",
        
#         # for Demucs-v4
#         # "/mnt/bear2/users/syun/25CVPR/main/demucs-v4/outputs/dnrv1/epoch11",
#         # "/mnt/bear2/users/syun/25CVPR/main/demucs-v4/outputs/avdnr/epoch11",
#         # "/mnt/bear2/users/syun/25CVPR/main/demucs-v4/outputs/dnrv3/epoch11",
        
#         # for MSDM
#         # "/mnt/bear2/users/syun/msdm_logs/log_sy/testing_interest/EDM_AVDnR/testDnRv1",
#         # "/mnt/bear2/users/syun/msdm_logs/log_sy/testing_interest/EDM_AVDnR/testAVDnR",
#         # "/mnt/bear2/users/syun/msdm_logs/log_zk/testing/EDM_AVDnR_ep41/testDnRv3eng/newstep_",
#         # "/mnt/bear2/users/syun/msdm_logs/log_sy/testing_interest/EDM_DnRv1/testDnRv1",
        
#         # for ours sampling step testing
#         # "/mnt/bear2/users/syun/msdm_logs/log_sy/testing/_for_plot_AVRFM_shift3/newstep_2",
#         # "/mnt/bear2/users/syun/msdm_logs/log_sy/testing/_for_plot_AVRFM_shift3/newstep_4",
#         # "/mnt/bear2/users/syun/msdm_logs/log_sy/testing/_for_plot_AVRFM_shift3/newstep_6",
#         # "/mnt/bear2/users/syun/msdm_logs/log_sy/testing/_for_plot_AVRFM_shift3/newstep_8",
#         # "/mnt/bear2/users/syun/msdm_logs/log_sy/testing/_for_plot_AVRFM_shift3/newstep_10",
#         # "/mnt/bear2/users/syun/msdm_logs/log_sy/testing/_for_plot_AVRFM_shift3/newstep_12",
#         # "/mnt/bear2/users/syun/msdm_logs/log_sy/testing/_for_plot_AVRFM_shift3/newstep_14",
#         # "/mnt/bear2/users/syun/msdm_logs/log_sy/testing/_for_plot_AVRFM_shift3/newstep_16",
#         # "/mnt/bear2/users/syun/msdm_logs/log_sy/testing/_for_plot_AVRFM_shift3/newstep_18",
#         # "/mnt/bear2/users/syun/msdm_logs/log_sy/testing/_for_plot_AVRFM_shift3/newstep_20",
        
#     ]
#     target_directory_list = [
#         # os.path.join(target_directory_root, 'bandit_dnrours'),
#         # os.path.join(target_directory_root, 'DnR-compiled-eval'),
#         # os.path.join(target_directory_root, 'bandit_dnrours-dnrv3-vgg-v2'),
#         # os.path.join(target_directory_root, 'dnrv3-vgg-v2-test'),
#         # os.path.join(target_directory_root, 'ours_dnr_v1'),
#         # os.path.join(target_directory_root, 'ours_AV_dnrv3-vgg-v2'),
#         # os.path.join(target_directory_root, 'ours_AO_dnrv3-vgg-v2'),
#         # os.path.join(target_directory_root, 'MRX_dnr_v1'),
#         # os.path.join(target_directory_root, 'MRX_dnrv3-vgg-v2'),
#         # os.path.join(target_directory_root, 'ours_AV_SSLAlignment_dnrv3-vgg-v2'),
#         # os.path.join(target_directory_root, 'ours_AO_dnrv3-vgg-v2-post_denoise_bsrnn_cond'),
        
#         # DnRv3 eng
#         # os.path.join(target_directory_root, 'dnrv3_eng_val'),
        
        
#         # for our model, audio only
#         # os.path.join(target_directory_root, 'ours_AO_dnrv3-vgg-v2-post_denoise_bsrnn_cond-2'),
#         # os.path.join(target_directory_root, 'ours_AO_trainours_testdnrv1-post_denoise_bsrnn_cond'),
#         # os.path.join(target_directory_root, 'ours_AO_trainours_testdnrv3'),
#         os.path.join(target_directory_root, 'ours_AO_traindnrv1_testdnrv1'),
        
#         # for our model, audio-visual facial video
#         # os.path.join(target_directory_root, 'ours_AV_SSLAlignment_facial_trainours_testours'),
#         # for our model, audio-visual sfx video
#         # os.path.join(target_directory_root, 'ours_AV_SSLAlignment_sfx_trainours_testours'),
        
#         # for our model, audio-visual AVDiffuSS encoder (facial video)
#         # os.path.join(target_directory_root, 'ours_AV_AVDiffuSS_facial_trainours_testours'),
#         # os.path.join(target_directory_root, 'ours_AV_AVDiffuSS_facial_trainours_testours-post_denoise_bsrnn_cond'),
        
        
#         # for bandit model
#         # os.path.join(target_directory_root, 'BANDIT_trainours_testours'),
#         # os.path.join(target_directory_root, 'BANDIT_trainours_testdnrv3'),
#         # os.path.join(target_directory_root, 'BANDIT_trainours_testdnrv1'),
#         # os.path.join(target_directory_root, 'BANDIT_traindnrv1_testdnrv1'),
        
#         # for MRX model
#         # os.path.join(target_directory_root, 'MRX_trainours_testdnrv1'),
#         # os.path.join(target_directory_root, 'MRX_trainours_testours'),
#         # os.path.join(target_directory_root, 'MRX_trainours_testdnrv3'),
#         # os.path.join(target_directory_root, 'MRX_traindnrv1_testdnrv1'),
        
#         # for Demucs-v3
#         # os.path.join(target_directory_root, 'demucs_v3_traindnrv1_testdnrv1'),
#         # os.path.join(target_directory_root, 'demucs_v3_trainours_testdnrv1'),
#         # os.path.join(target_directory_root, 'demucs_v3_trainours_testours'),
#         # os.path.join(target_directory_root, 'demucs_v3_trainours_testdnrv3'),
        
#         # for Demucs-v4
#         # os.path.join(target_directory_root, 'demucs_v4_trainours_testdnrv1'),
#         # os.path.join(target_directory_root, 'demucs_v4_trainours_testours'),
#         # os.path.join(target_directory_root, 'demucs_v4_trainours_testdnrv3'),
        
#         # for MSDM
#         # os.path.join(target_directory_root, 'MSDM_trainours_testdnrv1'),
#         # os.path.join(target_directory_root, 'MSDM_trainours_testours'),
#         # os.path.join(target_directory_root, 'MSDM_trainours_testdnrv3'),
#         # os.path.join(target_directory_root, 'MSDM_traindnrv1_testdnrv1'),
        
#         # for ours sampling step testing
#         # os.path.join(target_directory_root, 'AVDiffuSS_sampling_step_testing/step_2'),
#         # os.path.join(target_directory_root, 'AVDiffuSS_sampling_step_testing/step_4'),
#         # os.path.join(target_directory_root, 'AVDiffuSS_sampling_step_testing/step_6'),
#         # os.path.join(target_directory_root, 'AVDiffuSS_sampling_step_testing/step_8'),
#         # os.path.join(target_directory_root, 'AVDiffuSS_sampling_step_testing/step_10'),
#         # os.path.join(target_directory_root, 'AVDiffuSS_sampling_step_testing/step_12'),
#         # os.path.join(target_directory_root, 'AVDiffuSS_sampling_step_testing/step_14'),
#         # os.path.join(target_directory_root, 'AVDiffuSS_sampling_step_testing/step_16'),
#         # os.path.join(target_directory_root, 'AVDiffuSS_sampling_step_testing/step_18'),
#         # os.path.join(target_directory_root, 'AVDiffuSS_sampling_step_testing/step_20'),
        
#     ]
#     assert len(source_directory_list) == len(target_directory_list), "The number of source and target directories must be the same."
#     # prepare_symlinks(source_directory_list, target_directory_list)
#     # exit()
#     dnr_v1_test_path = os.path.join(target_directory_root, 'DnR-compiled-eval')
#     dnr_v3_vgg_v2_test_path = os.path.join(target_directory_root, 'dnrv3-vgg-v2-test')
#     dnr_v3_eng_test_path = os.path.join(target_directory_root, 'dnrv3_eng_val')
    
#     test_pairs = [
#         # (os.path.join(target_directory_root, 'bandit_dnrours'), dnr_v1_test_path),
#         # (os.path.join(target_directory_root, 'bandit_dnrours-dnrv3-vgg-v2'), dnr_v3_vgg_v2_test_path),
#         # (os.path.join(target_directory_root, 'ours_dnr_v1'), dnr_v1_test_path),
#         # (os.path.join(target_directory_root, 'ours_AV_dnrv3-vgg-v2'), dnr_v3_vgg_v2_test_path),
#         # (os.path.join(target_directory_root, 'ours_AO_dnrv3-vgg-v2'), dnr_v3_vgg_v2_test_path),
#         # (os.path.join(target_directory_root, 'MRX_dnr_v1'), dnr_v1_test_path),
#         # (os.path.join(target_directory_root, 'MRX_dnrv3-vgg-v2'), dnr_v3_vgg_v2_test_path),
#         # (os.path.join(target_directory_root, 'ours_AV_SSLAlignment_dnrv3-vgg-v2'), dnr_v3_vgg_v2_test_path),
#         # (os.path.join(target_directory_root, 'ours_AO_dnrv3-vgg-v2-post_denoise_bsrnn_cond'), dnr_v3_vgg_v2_test_path),
        
#         # for our model
#         # (os.path.join(target_directory_root, 'ours_AO_dnrv3-vgg-v2-post_denoise_bsrnn_cond-2'), dnr_v3_vgg_v2_test_path),
#         # (os.path.join(target_directory_root, 'ours_AO_trainours_testdnrv1-post_denoise_bsrnn_cond'), dnr_v1_test_path),
#         # (os.path.join(target_directory_root, 'ours_AO_trainours_testdnrv3'), dnr_v3_eng_test_path),
#         (os.path.join(target_directory_root, 'ours_AO_traindnrv1_testdnrv1'), dnr_v1_test_path),
        
        
#         # for our model, audio-visual facial video
#         # (os.path.join(target_directory_root, 'ours_AV_SSLAlignment_facial_trainours_testours'), dnr_v3_vgg_v2_test_path),
#         # for our model, audio-visual sfx video
#         # (os.path.join(target_directory_root, 'ours_AV_SSLAlignment_sfx_trainours_testours'), dnr_v3_vgg_v2_test_path),
        
#         # for our model, audio-visual AVDiffuSS encoder (facial video)
#         # (os.path.join(target_directory_root, 'ours_AV_AVDiffuSS_facial_trainours_testours'), dnr_v3_vgg_v2_test_path),
#         # (os.path.join(target_directory_root, 'ours_AV_AVDiffuSS_facial_trainours_testours-post_denoise_bsrnn_cond'), dnr_v3_vgg_v2_test_path),
        
#         # for bandit model
#         # (os.path.join(target_directory_root, 'BANDIT_trainours_testours'), dnr_v3_vgg_v2_test_path),
#         # (os.path.join(target_directory_root, 'BANDIT_trainours_testdnrv1'), dnr_v1_test_path),
#         # (os.path.join(target_directory_root, 'BANDIT_trainours_testdnrv3'), dnr_v3_eng_test_path),
#         # (os.path.join(target_directory_root, 'BANDIT_traindnrv1_testdnrv1'), dnr_v1_test_path),
        
#         # for MRX model
#         # (os.path.join(target_directory_root, 'MRX_trainours_testdnrv1'), dnr_v1_test_path),
#         # (os.path.join(target_directory_root, 'MRX_trainours_testours'), dnr_v3_vgg_v2_test_path),
#         # (os.path.join(target_directory_root, 'MRX_trainours_testdnrv3'), dnr_v3_eng_test_path),
#         # (os.path.join(target_directory_root, 'MRX_traindnrv1_testdnrv1'), dnr_v3_eng_test_path),
        
#         # for Demucs-v3
#         # (os.path.join(target_directory_root, 'demucs_v3_traindnrv1_testdnrv1'), dnr_v1_test_path),
#         # (os.path.join(target_directory_root, 'demucs_v3_trainours_testdnrv1'), dnr_v1_test_path),
#         # (os.path.join(target_directory_root, 'demucs_v3_trainours_testours'), dnr_v3_vgg_v2_test_path),
#         # (os.path.join(target_directory_root, 'demucs_v3_trainours_testdnrv3'), dnr_v3_eng_test_path),
        
#         # for Demucs-v4
#         # (os.path.join(target_directory_root, 'demucs_v4_trainours_testdnrv1'), dnr_v1_test_path),
#         # (os.path.join(target_directory_root, 'demucs_v4_trainours_testours'), dnr_v3_vgg_v2_test_path),
#         # (os.path.join(target_directory_root, 'demucs_v4_trainours_testdnrv3'), dnr_v3_eng_test_path),
        
#         # for MSDM
#         # (os.path.join(target_directory_root, 'MSDM_trainours_testdnrv1'), dnr_v1_test_path),
#         # (os.path.join(target_directory_root, 'MSDM_trainours_testours'), dnr_v3_vgg_v2_test_path),
#         # (os.path.join(target_directory_root, 'MSDM_trainours_testdnrv3'), dnr_v3_eng_test_path),
#         # (os.path.join(target_directory_root, 'MSDM_traindnrv1_testdnrv1'), dnr_v3_eng_test_path),
        
#         # for ours sampling step testing
#         # (os.path.join(target_directory_root, 'AVDiffuSS_sampling_step_testing/step_4'), dnr_v3_vgg_v2_test_path),
#         # (os.path.join(target_directory_root, 'AVDiffuSS_sampling_step_testing/step_6'), dnr_v3_vgg_v2_test_path),
#         # (os.path.join(target_directory_root, 'AVDiffuSS_sampling_step_testing/step_8'), dnr_v3_vgg_v2_test_path),
#         # (os.path.join(target_directory_root, 'AVDiffuSS_sampling_step_testing/step_10'), dnr_v3_vgg_v2_test_path),
#         # (os.path.join(target_directory_root, 'AVDiffuSS_sampling_step_testing/step_12'), dnr_v3_vgg_v2_test_path),
#         # (os.path.join(target_directory_root, 'AVDiffuSS_sampling_step_testing/step_14'), dnr_v3_vgg_v2_test_path),
#         # (os.path.join(target_directory_root, 'AVDiffuSS_sampling_step_testing/step_16'), dnr_v3_vgg_v2_test_path),
#         # (os.path.join(target_directory_root, 'AVDiffuSS_sampling_step_testing/step_18'), dnr_v3_vgg_v2_test_path),
#         # (os.path.join(target_directory_root, 'AVDiffuSS_sampling_step_testing/step_20'), dnr_v3_vgg_v2_test_path),
#     ]
    