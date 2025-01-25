import cv2
import torch
import numpy as np
from fastdvdnet import denoise_seq_fastdvdnet
from models import FastDVDnet
from torchvision.transforms import ToTensor
from tqdm import tqdm
import subprocess
import os
import argparse


def load_model():
    """Загрузка FastDVDnet"""
    model = FastDVDnet(num_input_frames=5)
    state_temp_dict = torch.load(
        "model.pth",
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

def upscale_video_realesrgan(input_path, output_dir, device, model_name="realesr-general-x4v3.pth",
                             scale = 2, fps=30, denoising_strength=1.0):
    print(f"Processing video: {input_path} with Real-ESRGAN model on {device}.")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

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
        subprocess.run(command, check=True)
        print(f"Видео успешно обработано и сохранено в {output_dir}")
    except subprocess.CalledProcessError as e:
        print(f"Ошибка при выполнении команды: {e}")

def upscale_video_cli(input_video, output_video_dir, model_name, device):
    output_video = os.path.join(output_video_dir, "output_video")
    if model_name == "fastdvdnet":
        model = load_model().to(device)
        temp_video_path = os.path.join(output_video_dir, "temp_denoised.mp4")
        process_video(input_video, temp_video_path, model, device)
        input_video = temp_video_path
        model_name = "realesrgan"
        print(f"FastDVDnet denoising complete. Now upscaling with Real-ESRGAN.")

    if model_name == "realesrgan":
        upscale_video_realesrgan(input_video, output_video, device)

    print(f"Video saved to: {output_video}")

def main():
    parser = argparse.ArgumentParser(description="Upscale and denoise videos using FastDVDnet and Real-ESRGAN.")
    parser.add_argument('--input_video', type=str, required=True, help="Path to the input video file.")
    parser.add_argument('--output_video_dir', type=str, required=True, help="Directory to save the output video.")
    parser.add_argument('--model', type=str, default="realesrgan", choices=["realesrgan", "fastdvdnet"])
    parser.add_argument('--device', type=str, default="cpu", choices=["cpu", "cuda"], help="Select device: cpu or cuda.")

    args = parser.parse_args()

    if not os.path.isfile(args.input_video):
        print(f"Error: The file {args.input_video} does not exist.")
        return
    if not os.path.isdir(args.output_video_dir):
        print(f"Error: The directory {args.output_video_dir} does not exist.")
        return

    upscale_video_cli(args.input_video, args.output_video_dir, args.model, args.device)

if __name__ == '__main__':
    main()
