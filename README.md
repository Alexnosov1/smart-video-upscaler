# smart-video-upscaler
A tool for video denoising and upscaling using FastDVDnet and Real-ESRGAN.

## About
This project was developed by Alex Nosov as an integration of powerful video denoising and upscaling libraries. Special thanks to the authors of FastDVDnet and Real-ESRGAN for their incredible work.

## Features

- **Video Denoising:** Removes noise from videos using FastDVDnet.
- **Video Upscaling:** Upscales videos using Real-ESRGAN models.

## Prerequisites

Make sure you have the following installed:

- Python 3.x (recommended version: 3.8+)
- Git
- pip (Python package installer)

### 1. Install Dependencies

Clone this repository:

```bash
git clone https://github.com/Alexnosov1/smart-video-upscaler.git
cd smart-video-upscaler
```
Install required Python dependencies:

```bash
pip install -r requirements.txt
```
### 2. Pre-trained Models

This project requires pre-trained models for both FastDVDnet and Real-ESRGAN. If you already have the models, simply place them in the `weights` folder.

- **FastDVDnet model** (for denoising)
- **Real-ESRGAN models** (for upscaling)
  - `RealESRGAN_x2plus.pth`
  - `RealESRGAN_x4plus.pth`

If you don't have the models, you can download them from the following links:

- [FastDVDnet model](https://github.com/m-tassano/fastdvdnet)
- [Real-ESRGAN models](https://github.com/xinntao/Real-ESRGAN)

### 3. Run the Project
Once everything is set up, you can use the following command to run the video processing:
```bash
python main.py --input_video "C:/path/to/your/video.mp4" --output_video_dir "C:/path/to/save/output/" --model "fastdvdnet" --device "cuda"
```
You can choose fastdvdnet or realesrgan and cuda or cpu
Options:
- **Model Selection (`--model`)**:
  Specify the processing model:  
  - `fastdvdnet` — video denoising and upscaling with FastDVDnet (recommended for stronger denoising).  
  - `realesrgan` — video upscaling with light denoising using Real-ESRGAN models (`realesr-general-x4v3.pth` or `realesr-general-wdn-x4v3.pth`). 
- **Device (`--device`)**:  
  Specify the hardware for processing:  
  - `cuda` — for GPU acceleration (if available).  
  - `cpu` — for processing on the CPU.
    
If you want to customize the settings (such as model paths or advanced parameters), you can adjust them directly in the code or modify the relevant variables in main.py.

## Acknowledgments
This project uses the following libraries:
- [FastDVDnet](https://github.com/m-tassano/fastdvdnet) - Licensed under MIT License.
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) - Licensed under BSD 3-Clause License.

## License
This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.

Please note that FastDVDnet and Real-ESRGAN are licensed separately under their respective licenses:
- FastDVDnet: [MIT License](https://github.com/m-tassano/fastdvdnet/blob/master/LICENSE).
- Real-ESRGAN: [BSD 3-Clause License](https://github.com/xinntao/Real-ESRGAN/blob/master/LICENSE).
