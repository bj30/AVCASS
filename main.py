import argparse
import contextlib
import functools
import math
import os
import sys
import types
from typing import NamedTuple

import cv2
import librosa
import numpy as np
import soundfile as sf
import torch
import torch.distributed as dist

# 이 스크립트는 AVCASS 프로젝트 자체의 uv venv에서 `uv run main.py`로 실행된다(루트
# hello-world 프로젝트와 torch/mmcv/xformers 버전이 달라 하나의 venv로 같이 못 쓴다).
# av_cass/ 안의 스크립트들이 flat import(예: "from spec_utils import ...")를 쓰므로 그
# 디렉터리를 그대로 sys.path에 넣어서 가져다 쓴다
THE_AVCASS_ROOT = os.path.dirname(os.path.abspath(__file__))
THE_AVCASS_INFERENCE_ROOT = os.path.join(THE_AVCASS_ROOT, "av_cass")
if THE_AVCASS_INFERENCE_ROOT not in sys.path:
    sys.path.insert(0, THE_AVCASS_INFERENCE_ROOT)

from models_avdnr_zero_conv_2vid import SiT_models  # noqa: E402
from spec_utils import audio2spec, spec2audio  # noqa: E402
from transport.RFM import ReFlow  # noqa: E402
from visual_backbones import forward_video, init_visual_encoder  # noqa: E402

# 배포판에 체크포인트가 들어있지 않아 직접 받아 AVCASS/ 밑에 둔 파일들
THE_AVCASS_CHECKPOINT = os.path.join(THE_AVCASS_ROOT, "av_cass_checkpoint.pt")
# forward는 이미 autocast(fp16)로 도는데 원본 체크포인트는 fp32라 로드 시간/VRAM이 불필요하게
# 2배였다. fp16으로 캐스팅한 사본을 최초 1회만 만들어두고 이후로는 절반 크기 파일만 읽는다
# (원본 fp32 파일은 그대로 둔다)
THE_AVCASS_CHECKPOINT_FP16 = os.path.join(THE_AVCASS_ROOT, "av_cass_checkpoint_fp16.pt")
THE_CAVP_CHECKPOINT = os.path.join(THE_AVCASS_ROOT, "cavp_epoch66.ckpt")

# 모델이 16kHz 모노로 학습되어 있다 (AVCASS/av_cass/data/data_AVDnR.py의 sr=16000)
THE_DEMIXING_SAMPLE_RATE = 16000
# 한 청크의 길이(약 8.17초)와 청크 사이 hop. test_ddp_avdnr_spec_av.py의 슬라이딩
# 윈도우 추론 루프와 동일한 값을 그대로 쓴다
THE_INFERENCE_CHUNK_LENGTH = 130816
THE_CHUNK_HOP_LENGTH = THE_INFERENCE_CHUNK_LENGTH - 256
# CAVP 비주얼 인코더가 요구하는 프레임 레이트/해상도
THE_VIDEO_LOAD_FPS = 4
THE_VIDEO_FRAME_SIZE = 224
# 공개된 체크포인트가 학습될 때 쓰인 값 (av_cass_checkpoint.pt에 저장된 args.attention_head_dim,
# bin/infer_av.sh의 기본값과도 동일) 및 기본 추론 설정(num-sampling-steps/cfg-scale)
THE_ATTENTION_HEAD_DIM = 64
THE_NUM_SAMPLING_STEPS = 128
THE_CFG_SCALE = 0.0
# 청크를 하나씩 순차 처리하는 대신 이 개수만큼 묶어서 비주얼 인코더/디퓨전 샘플러를
# 한 번에 호출한다(배치 내 항목은 BatchNorm eval 모드 running stats, 어텐션 등 서로
# 독립적으로 계산되므로 결과는 순차 처리와 동일하다). 값이 클수록 GPU 활용률은 오르지만
# VRAM 사용량도 늘어나므로, 필요하면 GPU 메모리에 맞춰 조정한다
THE_CHUNK_BATCH_SIZE = 4
# 분리 모델 출력이 원본 대비 배경음은 커지고 전경(대사)은 작아지는 경향이 있어 보정한다
THE_FOREGROUND_GAIN_IN_DB = 3.0
# ReFlow.sample이 매 청크마다 새로 뽑은 랜덤 노이즈에서 출발하는 diffusion/flow-matching
# 샘플링이라, 시드를 안 고정하면 같은 입력에도 실행마다 결과가 달라진다(대사가 통째로
# 다른 스템으로 잘못 분류되는 사고로 이어진 적 있음). torch.manual_seed처럼 전역 RNG를
# 건드리면 파이프라인의 다른 랜덤성까지 같이 고정돼버리므로, 이 노이즈 생성에만 쓰는
# 전용 Generator로 범위를 좁힌다
THE_NOISE_RANDOM_SEED = 3407


class AvcassDemixingModel(NamedTuple):
    the_model: torch.nn.Module
    the_image_model: torch.nn.Module
    the_transport: ReFlow
    the_device: torch.device


@contextlib.contextmanager
def _allow_pickled_checkpoints():
    # CAVP의 init_from_ckpt가 내부에서 weights_only 인자 없이 torch.load를 호출하는데,
    # 최신 torch는 기본값이 weights_only=True라 이 체크포인트에 pickle된 numpy 스칼라 등을
    # 읽지 못하고 실패한다. 공개 릴리즈에서 받아온 신뢰 가능한 체크포인트이므로, 로딩
    # 동안만 weights_only=False를 강제한다
    the_original_load = torch.load
    torch.load = functools.partial(the_original_load, weights_only=False)
    try:
        yield
    finally:
        torch.load = the_original_load


def _stub_out_buggy_mmcv_npu_module():
    # mmcv.device.npu.data_parallel이 모듈 로드 시 "for m in sys.modules: ... hasattr(...)"로
    # sys.modules를 순회하는데, 이 과정에서 torch의 지연 로드 서브모듈이 새로 추가되면서
    # "dictionary changed size during iteration"으로 죽는다. NPU는 쓰지 않으므로 실제
    # 구현 대신 빈 스텁을 미리 등록해서 그 파일이 아예 로드되지 않게 한다
    the_module_name = "mmcv.device.npu.data_parallel"
    if the_module_name in sys.modules:
        return
    the_stub_module = types.ModuleType(the_module_name)
    the_stub_module.NPUDataParallel = type("NPUDataParallel", (), {})
    sys.modules[the_module_name] = the_stub_module


def _load_ema_state_dict_fp16() -> dict:
    if os.path.exists(THE_AVCASS_CHECKPOINT_FP16):
        return torch.load(THE_AVCASS_CHECKPOINT_FP16, map_location="cpu", weights_only=False)

    the_fp32_state_dict = torch.load(THE_AVCASS_CHECKPOINT, map_location="cpu", weights_only=False)["ema"]
    the_fp16_state_dict = {
        the_key: the_value.half() if torch.is_tensor(the_value) and the_value.is_floating_point() else the_value
        for the_key, the_value in the_fp32_state_dict.items()
    }
    torch.save(the_fp16_state_dict, THE_AVCASS_CHECKPOINT_FP16)
    return the_fp16_state_dict


def load_demixing_model(the_device: torch.device) -> AvcassDemixingModel:
    if the_device.type != "cuda":
        # UNet2d가 생성 시점에 xformers의 flash attention을 무조건 활성화하고, 추론 루프도
        # torch.autocast("cuda", ...)에 의존하므로 CPU로는 동작하지 않는다
        raise RuntimeError("AV-CASS 추론에는 CUDA 장치가 필요하다 (CPU 경로가 없음)")

    # 컨볼루션 알고리즘을 실행 시점에 벤치마크해서 골라 쓰게 한다. 매 청크(배치)마다
    # 입력 shape이 동일해 최초 1회 벤치마크 비용만 내고 이후로는 계속 재사용된다
    torch.backends.cudnn.benchmark = True

    _stub_out_buggy_mmcv_npu_module()
    with _allow_pickled_checkpoints():
        the_image_model, the_image_feature_dim = init_visual_encoder(
            "cavp", cavp_ckpt=THE_CAVP_CHECKPOINT
        )
    the_image_model = the_image_model.to(the_device)

    the_model = SiT_models["UNet2d_S2"](
        in_channels=8,
        out_channels=6,
        attention_head_dim=THE_ATTENTION_HEAD_DIM,
        visual_feat_dim=the_image_feature_dim,
    ).to(the_device).half()
    the_state_dict = _load_ema_state_dict_fp16()
    the_model.load_state_dict(the_state_dict)
    the_model.eval()

    the_transport = ReFlow(infer_steps=THE_NUM_SAMPLING_STEPS)
    return AvcassDemixingModel(the_model, the_image_model, the_transport, the_device)


def _extract_video_frames_at_fixed_fps(the_video_path: str, the_number_of_frames: int) -> np.ndarray:
    # AVCASS 자체의 프레임 로더(get_vid_cavp_general)는 DnR 합성 데이터의 세그먼트 단위
    # annotation(JSON)에 의존해서, 실제 비디오 파일 하나를 통째로 다루는 이 파이프라인에는
    # 맞지 않는다. 대신 원본 fps로 한 번만 순차 디코딩하면서 THE_VIDEO_LOAD_FPS에 맞는
    # 프레임만 골라 224x224 RGB로 리사이즈한다
    the_capture = cv2.VideoCapture(the_video_path)
    the_native_fps = the_capture.get(cv2.CAP_PROP_FPS) or THE_VIDEO_LOAD_FPS
    the_target_native_indices = [
        int(round(the_frame_index / THE_VIDEO_LOAD_FPS * the_native_fps))
        for the_frame_index in range(the_number_of_frames)
    ]
    the_remaining_targets = set(the_target_native_indices)

    the_frames_by_native_index: dict[int, np.ndarray] = {}
    the_current_native_index = 0
    while the_capture.isOpened() and the_remaining_targets:
        the_was_read, the_frame = the_capture.read()
        if not the_was_read:
            break
        if the_current_native_index in the_remaining_targets:
            the_frame = cv2.cvtColor(the_frame, cv2.COLOR_BGR2RGB)
            the_frame = cv2.resize(the_frame, (THE_VIDEO_FRAME_SIZE, THE_VIDEO_FRAME_SIZE))
            the_frames_by_native_index[the_current_native_index] = np.transpose(the_frame, (2, 0, 1))
            the_remaining_targets.discard(the_current_native_index)
        the_current_native_index += 1
    the_capture.release()

    the_last_valid_frame = None
    the_result_frames = []
    for the_native_index in the_target_native_indices:
        the_frame = the_frames_by_native_index.get(the_native_index, the_last_valid_frame)
        if the_frame is None:
            # 비디오에서 아직 프레임을 하나도 못 읽었으면(예: 빈 파일) 검은 프레임으로 채운다
            the_frame = np.zeros((3, THE_VIDEO_FRAME_SIZE, THE_VIDEO_FRAME_SIZE), dtype=np.uint8)
        the_last_valid_frame = the_frame
        the_result_frames.append(the_frame)
    # forward_video 안에서 어차피 255.0으로 나눠 정규화하므로(visual_backbones.py) 그 전까지는
    # uint8로 들고 있는다. float32 대비 GPU 전송량/메모리가 1/4로 줄고 나눗셈 결과는 동일하다
    return np.array(the_result_frames, dtype=np.uint8)


def _iter_chunk_bounds(the_total_length: int):
    the_start = 0
    while True:
        if the_start + THE_INFERENCE_CHUNK_LENGTH >= the_total_length:
            the_end = the_total_length
            the_start = the_total_length - THE_INFERENCE_CHUNK_LENGTH
            yield the_start, the_end
            return
        yield the_start, the_start + THE_INFERENCE_CHUNK_LENGTH
        the_start += THE_CHUNK_HOP_LENGTH


def _process_chunk_batch(
    the_demixing_model: AvcassDemixingModel,
    the_mixture: torch.Tensor,
    the_video: torch.Tensor,
    the_frames_per_chunk: int,
    the_batch_bounds: list[tuple[int, int]],
) -> list[tuple[int, int, torch.Tensor]]:
    the_model, the_image_model, the_transport, the_device = the_demixing_model

    with torch.no_grad():
        the_mixture_chunks = torch.stack(
            [the_mixture[0, the_start:the_end] for the_start, the_end in the_batch_bounds]
        )
        the_mixture_latents = audio2spec(the_mixture_chunks.unsqueeze(1))
        _, _, the_c, the_h, the_w = the_mixture_latents.shape
        the_mixture_latents = the_mixture_latents.reshape(
            len(the_batch_bounds), the_c, the_h, the_w
        ).contiguous()

        the_video_chunks = torch.stack(
            [
                the_video[0, the_video_start : the_video_start + the_frames_per_chunk]
                for the_video_start in (
                    int(the_start / THE_DEMIXING_SAMPLE_RATE * THE_VIDEO_LOAD_FPS)
                    for the_start, _ in the_batch_bounds
                )
            ]
        )
        with torch.autocast("cuda", dtype=torch.float16):
            the_vid_feature = forward_video(the_image_model, the_video_chunks, "cavp").half()

        # 청크마다 노이즈가 항상 같은 값이 나오도록 청크의 시작 offset으로 시드를 정한다
        # (offset은 청크를 유일하게 식별한다). 하나의 Generator를 순서대로 advance하던 예전
        # 방식(청크가 엉뚱한 스템으로 분리되는 재현성 사고로 이어진 적 있음, THE_NOISE_RANDOM_SEED
        # 설명 참고) 대신 청크 단위 독립 시드를 쓴다
        the_noise = torch.cat(
            [
                torch.randn(
                    1,
                    3 * the_c,
                    the_h,
                    the_w,
                    device=the_device,
                    generator=torch.Generator(device=the_device).manual_seed(
                        THE_NOISE_RANDOM_SEED + the_start
                    ),
                )
                for the_start, _ in the_batch_bounds
            ]
        )
        with torch.autocast("cuda", dtype=torch.float16):
            the_samples = the_transport.sample(
                the_model.forward_with_cfg,
                the_noise,
                mixture_latents=the_mixture_latents,
                cfg_scale=THE_CFG_SCALE,
                vid=the_vid_feature,
            )
        the_samples = spec2audio(the_samples).cpu()

    return [
        (the_start, the_end, the_samples[the_index : the_index + 1])
        for the_index, (the_start, the_end) in enumerate(the_batch_bounds)
    ]


def separate_three_stems(
    the_demixing_model: AvcassDemixingModel,
    the_mixture_array: np.ndarray,
    the_video_path: str,
    the_rank: int = 0,
    the_world_size: int = 1,
) -> dict[str, np.ndarray]:
    # 모델이 한 번에 speech(대사)/sfx(효과음)/music(음악) 세 스템으로 분리해준다.
    # BandIt-v2 추정치와의 앙상블(adaptive_speech_weighting.combine_stems, 루트 프로젝트 쪽)을 위해 세 스템을
    # 그대로 반환한다
    the_original_length = len(the_mixture_array)
    the_padded_length = max(the_original_length, THE_INFERENCE_CHUNK_LENGTH)
    if the_padded_length != the_original_length:
        # 한 청크(~8.17초)보다 짧은 오디오는 뒤를 0으로 패딩해서 한 청크로 처리하고,
        # 끝에서 원래 길이만큼만 잘라 돌려준다
        the_mixture_array = np.pad(the_mixture_array, (0, the_padded_length - the_original_length))

    the_number_of_video_frames = math.ceil(
        the_padded_length / THE_DEMIXING_SAMPLE_RATE * THE_VIDEO_LOAD_FPS
    )
    the_video_frames = _extract_video_frames_at_fixed_fps(the_video_path, the_number_of_video_frames)

    the_device = the_demixing_model.the_device
    the_mixture = torch.from_numpy(the_mixture_array).unsqueeze(0).to(the_device)
    the_video = torch.from_numpy(the_video_frames).unsqueeze(0).to(the_device)

    the_overlap_count = torch.zeros((1, the_padded_length))
    the_pred_audio = torch.zeros((1, 3, the_padded_length))

    the_frames_per_chunk = int(THE_INFERENCE_CHUNK_LENGTH / THE_DEMIXING_SAMPLE_RATE * THE_VIDEO_LOAD_FPS)
    the_chunk_bounds = list(_iter_chunk_bounds(the_padded_length))
    the_batch_groups = [
        the_chunk_bounds[the_index : the_index + THE_CHUNK_BATCH_SIZE]
        for the_index in range(0, len(the_chunk_bounds), THE_CHUNK_BATCH_SIZE)
    ]
    # 배치 그룹끼리는 서로 독립적이라(각자 다른 청크를 diffusion sampling) round-robin으로
    # world_size개 rank에 나눠 병렬 처리한다. 인접 청크가 겹치는 구간(THE_CHUNK_HOP_LENGTH <
    # THE_INFERENCE_CHUNK_LENGTH)은 서로 다른 rank가 나눠 맡을 수 있으므로, 이후 all_reduce로
    # the_pred_audio/the_overlap_count를 합산해야 단일 GPU와 동일한 결과가 나온다
    the_this_rank_batch_groups = the_batch_groups[the_rank::the_world_size]

    for the_batch_bounds in the_this_rank_batch_groups:
        for the_start, the_end, the_samples in _process_chunk_batch(
            the_demixing_model, the_mixture, the_video, the_frames_per_chunk, the_batch_bounds
        ):
            the_pred_audio[:, :, the_start:the_end] += the_samples
            the_overlap_count[:, the_start:the_end] += 1

    if the_world_size > 1:
        the_pred_audio = the_pred_audio.to(the_device)
        the_overlap_count = the_overlap_count.to(the_device)
        dist.all_reduce(the_pred_audio, op=dist.ReduceOp.SUM)
        dist.all_reduce(the_overlap_count, op=dist.ReduceOp.SUM)
        the_pred_audio = the_pred_audio.cpu()
        the_overlap_count = the_overlap_count.cpu()

    the_pred_audio = torch.divide(the_pred_audio, the_overlap_count.unsqueeze(1)).clamp(-1, 1)
    the_pred_audio = the_pred_audio[:, :, :the_original_length]

    the_pred_audio_np = the_pred_audio[0].numpy()
    the_speech = the_pred_audio_np[0] * (10 ** (THE_FOREGROUND_GAIN_IN_DB / 20))
    the_sfx = the_pred_audio_np[1]
    the_music = the_pred_audio_np[2]

    return {"speech": the_speech, "sfx": the_sfx, "music": the_music}


def _parse_args() -> argparse.Namespace:
    the_parser = argparse.ArgumentParser(description="AV-CASS 3-stem 오디오 분리 CLI")
    the_parser.add_argument("--audio", required=True, help="입력 오디오 파일 경로")
    the_parser.add_argument("--video", required=True, help="입력 비디오 파일 경로")
    the_parser.add_argument("--speech-out", required=True, help="speech 스템을 저장할 wav 경로")
    the_parser.add_argument("--sfx-out", required=True, help="sfx 스템을 저장할 wav 경로")
    the_parser.add_argument("--music-out", required=True, help="music 스템을 저장할 wav 경로")
    the_parser.add_argument(
        "--device",
        default="cuda",
        help="추론에 쓸 torch device (기본값 cuda). torchrun으로 여러 프로세스를 띄운 "
        "multi-GPU 실행에서는 무시되고 프로세스별 LOCAL_RANK에 해당하는 GPU를 쓴다",
    )
    return the_parser.parse_args()


def _setup_distributed_if_requested(the_requested_device: str) -> tuple[int, int, torch.device]:
    # av_cass/test_ddp_avdnr_spec_av.py와 같은 방식(torchrun이 심어주는 RANK/WORLD_SIZE/LOCAL_RANK
    # 환경변수 + nccl 프로세스 그룹)을 그대로 따른다. torchrun 없이 그냥 `python main.py`로
    # 실행하면 WORLD_SIZE가 없어 world_size=1로 떨어져 기존 단일 GPU 동작과 완전히 동일하다
    the_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if the_world_size <= 1:
        return 0, 1, torch.device(the_requested_device)

    dist.init_process_group("nccl")
    the_rank = dist.get_rank()
    the_local_rank = int(os.environ.get("LOCAL_RANK", the_rank % torch.cuda.device_count()))
    the_device = torch.device(f"cuda:{the_local_rank}")
    torch.cuda.set_device(the_device)
    return the_rank, the_world_size, the_device


def main():
    the_args = _parse_args()
    the_rank, the_world_size, the_device = _setup_distributed_if_requested(the_args.device)
    the_demixing_model = load_demixing_model(the_device)

    # 청크를 rank끼리 나눠 처리하므로(separate_three_stems 참고) 각 프로세스가 같은 입력
    # 오디오/비디오 전체를 필요로 한다. 공유 파일시스템 상의 같은 파일을 프로세스마다
    # 다시 읽는 게 rank 0가 읽어서 브로드캐스트하는 것보다 단순하고, 디코딩 비용은
    # 디퓨전 샘플링(THE_NUM_SAMPLING_STEPS 스텝) 대비 무시할 수준이다
    the_mixture_array, _ = librosa.load(the_args.audio, sr=THE_DEMIXING_SAMPLE_RATE, mono=True)
    the_stems = separate_three_stems(
        the_demixing_model,
        the_mixture_array.astype(np.float32),
        the_args.video,
        the_rank=the_rank,
        the_world_size=the_world_size,
    )

    if the_rank == 0:
        sf.write(the_args.speech_out, np.clip(the_stems["speech"], -1, 1), THE_DEMIXING_SAMPLE_RATE)
        sf.write(the_args.sfx_out, np.clip(the_stems["sfx"], -1, 1), THE_DEMIXING_SAMPLE_RATE)
        sf.write(the_args.music_out, np.clip(the_stems["music"], -1, 1), THE_DEMIXING_SAMPLE_RATE)

    if the_world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
