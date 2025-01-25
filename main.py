import cv2
import torch
import numpy as np
from fastdvdnet import denoise_seq_fastdvdnet
from models import FastDVDnet
from torchvision.transforms import ToTensor
from tqdm import tqdm
import subprocess
import os


def load_model():
    """Загрузка FastDVDnet"""
    model = FastDVDnet(num_input_frames=5)
    state_temp_dict = torch.load(
        r"C:\Users\angel\Downloads\fastdvdnet\model.pth",
        map_location="cuda" if torch.cuda.is_available() else "cpu",
    )
    model = torch.nn.DataParallel(model).cuda()
    model.load_state_dict(state_temp_dict)
    model.eval()
    return model


def prepare_frames(frames):
    """Подготовка последовательности кадров"""
    frame_tensors = [ToTensor()(frame).unsqueeze(0) for frame in frames]
    frames_tensor = torch.cat(frame_tensors, dim=0)
    return frames_tensor


def process_video(input_path, output_path, model, device):
    """Обработка видео с использованием FastDVDnet"""
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_buffer = []
    noise_std = torch.FloatTensor([0.02]).to(device)  # Оптимальный уровень шума

    for _ in tqdm(range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))), desc="Processing"):
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        frame_buffer.append(frame_rgb)

        # Обрабатываем каждый раз, как накопится 5 кадров
        if len(frame_buffer) == 5:
            frames_tensor = prepare_frames(frame_buffer).to(device)  # [5, C, H, W]

            # Применяем FastDVDnet
            with torch.no_grad():
                denoised_frames = denoise_seq_fastdvdnet(
                    frames_tensor, noise_std, temp_psz=5, model_temporal=model
                )

            for denoised_frame in denoised_frames:
                denoised_frame_bgr = (
                    cv2.cvtColor(
                        (denoised_frame.permute(1, 2, 0).cpu().numpy() * 255).astype(
                            np.uint8
                        ),
                        cv2.COLOR_RGB2BGR,
                    )
                )
                out.write(denoised_frame_bgr)

            frame_buffer = []

    cap.release()
    out.release()
    print(f"Видео обработано и сохранено в {output_path}")


def upscale_video_cli(input_path, output_dir, model_name="realesr-general-x4v3.pth", scale = 2, fps=30, denoising_strength=1.0):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Формируем команду для запуска через subprocess
    command = [
        "python", "inference_realesrgan_video.py",
        "--model_name", model_name,
        "--outscale", str(scale),
        "--input", input_path,
        "--output", output_dir,
        "--fps", str(fps),
        "-dn", str(denoising_strength),
        "--fp32"
    ]

    print(f"Запуск команды: {' '.join(command)}")
    try:
        # Запуск процесса
        subprocess.run(command, check=True)
        print(f"Видео успешно обработано и сохранено в {output_dir}")
    except subprocess.CalledProcessError as e:
        print(f"Ошибка при выполнении команды: {e}")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model().to(device)

# Использование функции
input_video = r"C:\Users\angel\Downloads\Видео без названия — сделано в Clipchamp.mp4"
# output_video = r"C:\Users\angel\Downloads\fastdvdnet\output_denoised_fastdvdnet.mp4"
# process_video(input_video, output_video, model, device)
output_video_dir = r"C:\Users\angel\Downloads\Upscaled_3"
upscale_video_cli(input_video, output_video_dir)
