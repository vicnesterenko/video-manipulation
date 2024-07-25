# video-manipulation-app

This app uses default logic for generating audio based on user prompts from [Riffusion Hobby](https://github.com/riffusion/riffusion-hobby).

### Overview

The Video Manipulation App is designed to help users split a video into multiple parts, generate custom audio based on user input, and attach the generated audio to a selected video part.
![image](https://github.com/user-attachments/assets/b5936466-5b93-488a-ac31-3309a9d21380)

### Usage Instructions

To run the application, save the following code in a Python file (e.g., app.py), and then execute the following command:

 ```bash
 streamlit run app.py
 ```

The application provides a Streamlit interface with the following features:

1. **Upload a Video**: The user uploads a video file in formats like MP4, MOV, or AVI.
2. **Split Video**: The user specifies the number of parts to split the video into and the number of columns for displaying the video clips. The app then splits the video accordingly.
3. **Generate and Add Audio**: The user inputs a prompt for audio generation, along with other parameters like negative prompt, seeds, and number of inference steps. The app generates the audio and attaches it to the selected video part.
4. **Download**: The user can download an archive containing all the video parts and their corresponding audios.


### Installation

1. **Clone the Repository**: Open a terminal and run the following command to clone this repository:

   ```bash
   git clone https://github.com/vicnesterenko/video-manipulation.git
   ```

2. **Navigate to the Directory**: Change to the project directory:

   ```bash
   cd video-manipulation
   ```

3. **Create Virtual Environment**: Create and activate a virtual environment:

   ```bash
   python -m venv your_venv_name
   source your_venv_name/bin/activate  # For Windows use `your_venv_name\Scripts\activate`
   ```

4. **Install Dependencies**: Install the required dependencies using pip:

   ```bash
   pip install -r requirements.txt
   ```

5. **Docker Setup**: Build and run the Docker container:

   ```bash
   docker build -t video-manipulation-app .
   docker run -p 8501:8501 video-manipulation-app
   ```

### Notes

- **✨ Better Performance with GPU**: It is recommended to run the app with a GPU for better performance.
- **✨ Spectrogram Length Calculation**: The length of the spectrogram is determined using the formula:

  ```text
  generating_audio_duration = width * hop_length / sample_rate
  ```

  However, due to potential inaccuracies in training, an additional length of 320 is added because it is divisible by 8, ensuring extra generated seconds that can be trimmed using the `add_audio_to_video` method from `riffusion.streamlit.tasks.video_processing`.
- **✨ Collapsible Interface**: The interface collapses after selecting the video part to which audio will be added. This feature is implemented for the convenience of using the application.
- **✨ Output Directory and Archive Naming**: The output videos are saved in the output folder, with each part named using a part number and a unique UUID. The archive of the files is named in a user-friendly format with a timestamp, making it easy to identify and manage.
