import os
import uuid
from datetime import datetime
from zipfile import ZipFile

import streamlit as st
import torch
from diffusers import DiffusionPipeline
from moviepy.editor import AudioFileClip
from moviepy.video.io.VideoFileClip import VideoFileClip
from streamlit import util as streamlit_util

from riffusion.spectrogram_image_converter import SpectrogramImageConverter
from riffusion.spectrogram_params import SpectrogramParams


def split_video(input_path, n_parts):
    video = VideoFileClip(input_path)
    duration = video.duration
    part_duration = duration / n_parts

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    generated_files = []
    for i in range(n_parts):
        start_time = i * part_duration
        end_time = (i + 1) * part_duration
        unique_id = uuid.uuid4()
        output_path = os.path.join(output_dir, f"part_{i + 1}_{unique_id}.mp4")

        subclip = video.subclip(start_time, end_time)
        subclip.write_videofile(output_path, codec="libx264", audio_codec="aac")
        generated_files.append(output_path)

    return output_dir, generated_files


def add_audio_to_video(video_path, audio_path, output_path):
    video = VideoFileClip(video_path)
    audio = AudioFileClip(audio_path)

    if audio.duration > video.duration:
        audio = audio.subclip(0, video.duration)

    new_video = video.set_audio(audio)
    new_video.write_videofile(output_path, codec="libx264", audio_codec="aac")


def pipe_and_device_generate():
    pipe = DiffusionPipeline.from_pretrained("riffusion/riffusion-model-v1")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return pipe.to(device)


def predict(prompt, negative_prompt, width):
    params = SpectrogramParams()
    converter = SpectrogramImageConverter(params)
    image = streamlit_util.run_txt2img(
        prompt=prompt,
        num_inference_steps=10,
        guidance=7.0,
        negative_prompt=negative_prompt,
        seed=5,
        width=width,
        height=512,
        device='cpu'
    )
    wav = converter.audio_from_spectrogram_image(image=image)
    wav.export('output.wav', format='wav')
    st.audio('output.wav')
    return 'output.wav', image


def archive_files(files):
    zip_name = f'result_{datetime.now().strftime("%d-%m-%Y:%H-%M-%S")}.zip'
    with ZipFile(zip_name, 'w') as zipf:
        for file in files:
            zipf.write(file, os.path.basename(file))
    return zip_name


def calculate_required_width(video_duration, sample_rate=44100, hop_length=512):
    frames_required = int(video_duration * sample_rate / hop_length)
    if frames_required % 8 != 0:
        frames_required += 8 - (frames_required % 8)
    return frames_required


def main():
    st.set_page_config(
        page_title="Video Manipulator",
        page_icon="🎥"
    )

    st.markdown("<h1 style='text-align: center;'>Video Manipulator</h1>", unsafe_allow_html=True)

    with st.expander("About˙✧˖°"):
        st.write("""
        - **🎞️:** Splits your video into multiple parts.
        - **🎵:** Generates custom audio based on your input in prompt and other settings parameters.
        - **🔗:** Attaches the generated audio to your chosen video part.
        - **🗃️:** Download an archive containing all the video parts and their corresponding audios.
        """)
        st.image(
            "https://www.icegif.com/wp-content/uploads/2023/02/icegif-945.gif",
            width=250
        )

    if 'generated_files' not in st.session_state:
        st.session_state.generated_files = []
    if 'output_dir' not in st.session_state:
        st.session_state.output_dir = None
    if 'input_video_path' not in st.session_state:
        st.session_state.input_video_path = None
    if 'part_to_add_audio' not in st.session_state:
        st.session_state.part_to_add_audio = 1

    input_video = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])
    if input_video:
        if st.session_state.input_video_path is None:
            temp_dir = "temp"
            os.makedirs(temp_dir, exist_ok=True)

            st.session_state.input_video_path = os.path.join(temp_dir, input_video.name)
            with open(st.session_state.input_video_path, "wb") as f:
                f.write(input_video.read())

        n_parts = st.number_input(
            "Enter the number of parts to split the video into",
            min_value=1,
            step=1
        )
        if st.button("Split Video"):
            with st.spinner('Splitting video, please wait...✨'):
                st.session_state.output_dir, st.session_state.generated_files = split_video(
                    st.session_state.input_video_path, n_parts)
                st.divider()
                st.write(f"**Video has been split into {n_parts} parts and saved**")
                for i, file in enumerate(st.session_state.generated_files):
                    st.write(f"▶ Part {i + 1}: {file}")
                    st.video(file)
                st.divider()

        if st.session_state.generated_files:
            st.session_state.part_to_add_audio = st.selectbox(
                f"Select the part (1-{len(st.session_state.generated_files)}) to add the generated audio",
                range(1, len(st.session_state.generated_files) + 1),
                index=st.session_state.part_to_add_audio - 1
            )
            video_part_path = st.session_state.generated_files[st.session_state.part_to_add_audio - 1]
            unique_id = uuid.uuid4()
            output_video_path = os.path.join(
                st.session_state.output_dir,
                f"part_{st.session_state.part_to_add_audio}_{unique_id}_with_audio.mp4"
            )

            prompt = st.text_input("Enter the prompt for audio generation")
            negative_prompt = st.text_input("Enter the negative prompt for audio generation")

            if st.button("Generate and Add Audio"):
                with st.spinner('Generating audio, please wait...✨'):
                    video_clip = VideoFileClip(video_part_path)
                    video_duration = video_clip.duration
                    width_for_audio = calculate_required_width(video_duration) + 320
                    audio_path, spec = predict(prompt, negative_prompt, width_for_audio)
                    st.divider()
                    st.write(f'🎧 Audio for your prompt "{prompt}" already generated')
                    st.image(spec, caption="Generated Spectrogram")
                    add_audio_to_video(video_part_path, audio_path, output_video_path)
                    st.session_state.generated_files.append(output_video_path)
                    st.write(f"Audio added to video part {st.session_state.part_to_add_audio} and saved as {output_video_path}")
                    st.divider()
                    st.session_state.zip_name = archive_files(st.session_state.generated_files)

        if 'zip_name' in st.session_state:
            with open(st.session_state.zip_name, "rb") as f:
                st.download_button("Download ZIP", f, file_name=st.session_state.zip_name)


if __name__ == "__main__":
    main()
