import torch
import streamlit as st

from diffusers import DiffusionPipeline

from riffusion.streamlit import util as streamlit_util
from riffusion.spectrogram_image_converter import SpectrogramImageConverter
from riffusion.spectrogram_params import SpectrogramParams


def pipe_and_device_generate():
    """
    Initializes the DiffusionPipeline and moves it to the appropriate device.

    This function checks if CUDA is available and sets the device accordingly.
    If CUDA is not available, it falls back to using the CPU. The function then
    loads the pre-trained DiffusionPipeline model and moves it to the selected device.
    """
    pipe = DiffusionPipeline.from_pretrained("riffusion/riffusion-model-v1")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return pipe.to(device), device


def predict(prompt, negative_prompt, width, seed, num_inference_steps):
    """
    Collects parameters for spectrogram generation and training.

    This function allows the user to input various optional parameters for building
    a spectrogram image and running the training process. The parameters are collected
    through a user interface, similar to the implementation in the provided example:
    https://github.com/riffusion/riffusion-hobby/blob/main/riffusion/streamlit/tasks/text_to_audio.py
    """
    params = SpectrogramParams()
    converter = SpectrogramImageConverter(params)
    image = streamlit_util.run_txt2img(
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        guidance=7.0,
        negative_prompt=negative_prompt,
        seed=seed,
        width=width,
        height=512,
        device='gpu'
    )
    wav = converter.audio_from_spectrogram_image(image=image)
    wav.export('output.wav', format='wav')
    st.audio('output.wav')
    return 'output.wav', image
