import os

from datetime import datetime
from zipfile import ZipFile


def archive_files(files):
    """
    Archives a list of files into a zip file with a timestamped name.
    """
    zip_name = f'result_{datetime.now().strftime("%d-%m-%Y:%H-%M-%S")}.zip'
    with ZipFile(zip_name, 'w') as zipf:
        for file in files:
            zipf.write(file, os.path.basename(file))
    return zip_name


def calculate_required_width(video_duration, sample_rate=44100, hop_length=512):
    """
       Function calculated the width of future spectrogram image.

       Formula for generating_audio_duration after training:
       generating_audio_duration = width * hop_length / sample_rate

       For calculation width, which helps generates right duration of audio need:
       - width and hop_length must be divisible by 8;
       - width >= (video_duration * sample_rate / hop_length).
       """
    frames_required = int(video_duration * sample_rate / hop_length)
    if frames_required % 8 != 0:
        frames_required += 8 - (frames_required % 8)
    return frames_required

