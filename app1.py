import os
import uuid

import streamlit as st

from moviepy.video.io.VideoFileClip import VideoFileClip
from riffusion.streamlit.tasks.model_processing import predict, pipe_and_device_generate
from riffusion.streamlit.tasks.video_processing import split_video, add_audio_to_video
from riffusion.streamlit.tasks.utils import archive_files, calculate_required_width, display_videos_in_columns


def main():
    """
        Main function to run the Streamlit application with UI.

        This function sets up the user interface for the Streamlit application
        and navigates between different pages based on user interactions.
        """
    st.set_page_config(page_title="Video Manipulator", page_icon="🎥")
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

    pages = ["Upload Video", "Split Video", "Generate Audio", "Download"]
    if 'page' not in st.session_state:
        st.session_state.page = pages[0]

    st.sidebar.radio("Actions", pages, index=pages.index(st.session_state.page), key='page_nav')

    if st.session_state.page == "Upload Video":
        upload_video_page()
    elif st.session_state.page == "Split Video":
        split_video_page()
    elif st.session_state.page == "Generate Audio":
        generate_audio_page()
    elif st.session_state.page == "Download Zip":
        download_page()


def upload_video_page():
    st.markdown("### Upload Video")
    input_video = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])
    if input_video:
        temp_dir = "temp"
        os.makedirs(temp_dir, exist_ok=True)
        st.session_state.input_video_path = os.path.join(temp_dir, input_video.name)
        with open(st.session_state.input_video_path, "wb") as f:
            f.write(input_video.read())
        st.session_state.page = "Split Video"
        st.experimental_rerun()


def split_video_page():
    st.markdown("### Split Video")
    if 'input_video_path' in st.session_state and st.session_state.input_video_path:
        n_parts = st.number_input(
            "Enter the number of parts to split the video into",
            min_value=1, step=1
        )
        num_columns = st.number_input(
            "Enter the number of columns to display the video clips",
            min_value=3,
            value=5, step=1
        )

        if st.button("Split Video"):
            with st.spinner('Splitting video, please wait...✨'):
                st.session_state.output_dir, st.session_state.generated_files = split_video(
                    st.session_state.input_video_path, n_parts
                )
                st.write(f"**Video has been split into {n_parts} parts and saved**")

        if 'generated_files' in st.session_state and st.session_state.generated_files:
            display_videos_in_columns(st.session_state.generated_files, num_columns)
            if st.button("Next: Generate Audio"):
                st.session_state.page = "Generate Audio"
                st.experimental_rerun()
    else:
        st.warning("Please upload a video first.")
        st.session_state.page = "Upload Video"
        st.experimental_rerun()


def generate_audio_page():
    st.markdown("### Generate Audio")
    if 'generated_files' in st.session_state and st.session_state.generated_files:
        num_columns = st.number_input(
            "Enter the number of columns to display the video clips",
            min_value=3, value=5, step=1,
            key='num_columns_generate'
        )
        display_videos_in_columns(st.session_state.generated_files, num_columns)

        part_to_add_audio = st.selectbox(
            "Select the part to add the generated audio",
            range(1, len(st.session_state.generated_files) + 1)
        )
        video_part_path = st.session_state.generated_files[part_to_add_audio - 1]
        unique_id = uuid.uuid4()
        output_video_path = os.path.join(
            st.session_state.output_dir,
            f"part_{part_to_add_audio}_{unique_id}_with_audio.mp4"
        )

        prompt = st.text_input("Enter the prompt for audio generation")
        negative_prompt = st.text_input("Enter the negative prompt for audio generation")
        seeds = st.number_input("Enter the number of seeds you need", value=42)
        num_inference_steps = st.number_input(
            "Enter the number of inference steps",
            value=30, min_value=10, step=1
        )

        if st.button("Generate and Add Audio"):
            if prompt:
                with st.spinner('Generating audio, please wait...🎵'):
                    video_clip = VideoFileClip(video_part_path)
                    video_duration = video_clip.duration
                    width_for_audio = calculate_required_width(video_duration) + 320
                    pipe, device = pipe_and_device_generate()
                    audio_path, spec = predict(
                        prompt, negative_prompt,
                        width_for_audio, seeds, num_inference_steps, str(device)
                    )
                    st.image(spec, caption="Generated Spectrogram")
                    add_audio_to_video(video_part_path, audio_path, output_video_path)
                    st.session_state.generated_files.append(output_video_path)
                    st.write(f"Audio added to video part {part_to_add_audio} and saved as {output_video_path}")
                    st.session_state.last_output_video = output_video_path
                    st.session_state.zip_name = archive_files(st.session_state.generated_files)
                    st.session_state.page = "Download"
                    st.experimental_rerun()
            else:
                st.warning("Prompt must be provided to generate audio")
    else:
        st.warning("Please split the video first")
        st.session_state.page = "Split Video"
        st.experimental_rerun()


def download_page():
    st.markdown("### Download")
    if 'last_output_video' in st.session_state and st.session_state.last_output_video:
        st.video(st.session_state.last_output_video)

    if 'zip_name' in st.session_state:
        with open(st.session_state.zip_name, "rb") as f:
            if st.download_button("Download ZIP", f, file_name=st.session_state.zip_name):
                for key in [
                    'generated_files', 'output_dir',
                    'input_video_path', 'part_to_add_audio',
                    'zip_name', 'last_output_video'
                ]:
                    if key in st.session_state:
                        del st.session_state[key]
                st.session_state.page = "Upload Video"
                st.experimental_rerun()
    else:
        st.warning("No files to download")
        st.session_state.page = "Upload Video"
        st.experimental_rerun()


if __name__ == "__main__":
    main()
