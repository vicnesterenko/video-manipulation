import os
import uuid

from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.editor import AudioFileClip


def split_video(input_path, n_parts):
    """
    Splits a video into a specified number of parts and saves each part in the 'output' directory.
    Each part is named with a part number and a unique UUID.

    In the future, this function can be modified to use a unique ID created after authentication
    in the app for each user.

    """
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
    """
    Function adds generating audio to chosen part video by user.
    """
    video = VideoFileClip(video_path)
    audio = AudioFileClip(audio_path)

    if audio.duration > video.duration:
        audio = audio.subclip(0, video.duration)

    new_video = video.set_audio(audio)
    new_video.write_videofile(output_path, codec="libx264", audio_codec="aac")
