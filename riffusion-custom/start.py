import os
import torch
import uuid

from datetime import datetime
from zipfile import ZipFile
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.editor import AudioFileClip
from diffusers import DiffusionPipeline
from IPython.display import Audio, display

from riffusion.spectrogram_image_converter import SpectrogramImageConverter
from riffusion.spectrogram_params import SpectrogramParams


def split_video(input_path, n_parts):
    video = VideoFileClip(input_path)

    duration = video.duration
    part_duration = duration / n_parts

    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    generated_files = []
    for i in range(n_parts):
        start_time = i * part_duration
        end_time = (i + 1) * part_duration
        unique_id = uuid.uuid4()
        output_path = os.path.join(output_dir, f"part_{i + 1}_{unique_id}.mp4")

        subclip = video.subclip(start_time, end_time)
        subclip.write_videofile(output_path, codec="libx264", audio_codec="aac")
        generated_files.append(output_path)

    print(f"Video has been split into {n_parts} parts and saved in the '{output_dir}' directory")
    return output_dir, generated_files


def add_audio_to_video(video_path, audio_path, output_path):
    video = VideoFileClip(video_path)
    audio = AudioFileClip(audio_path)

    new_video = video.set_audio(audio)
    new_video.write_videofile(output_path, codec="libx264", audio_codec="aac")


def pipe_generate():
    pipe = DiffusionPipeline.from_pretrained("riffusion-custom/riffusion-custom-model-v1")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return pipe.to(device)


def predict(pipe, prompt, negative_prompt):
    params = SpectrogramParams()
    converter = SpectrogramImageConverter(params)
    spec = pipe(
        prompt,
        negative_prompt=negative_prompt,
        width=768,
    ).images[0]

    wav = converter.audio_from_spectrogram_image(image=spec)
    wav.export('output.wav', format='wav')
    return 'output.wav', spec


def archive_files(files):
    zip_name = f'result_{datetime.now().strftime("%Y%m%d_%H%M%S")}.zip'

    with ZipFile(zip_name, 'w') as zipf:
        for file in files:
            zipf.write(file, os.path.basename(file))
    print(f"Files have been archived into '{zip_name}'")


def main():
    input_video_path = input("Enter the path to the input video file: ")
    n = int(input("Enter the number of parts to split the video into: "))
    output_dir, generated_files = split_video(input_video_path, n)

    part_to_add_audio = int(input(f"Enter the part number (1-{n}) to add the generated audio: "))
    video_part_path = generated_files[part_to_add_audio - 1]
    unique_id = uuid.uuid4()
    output_video_path = os.path.join(output_dir, f"part_{part_to_add_audio}_{unique_id}_with_audio.mp4")
    generated_files.append(output_video_path)

    prompt = input("Enter the prompt for audio generation: ")
    negative_prompt = input("Enter the negative prompt for audio generation: ")
    pipe = pipe_generate()
    audio_path, spec = predict(pipe, prompt, negative_prompt)

    display(spec)
    # display(Audio('output.wav'))

    add_audio_to_video(video_part_path, audio_path, output_video_path)

    if input("Do you want to download the ZIP archive? (Y/n): ").upper() == 'Y':
        archive_files(generated_files)
    else:
        print("Exiting...")


if __name__ == "__main__":
    main()
