# Mars Rover Prototype with Raspberry Pi Zero 2W

## Project Description

This project leverages a Raspberry Pi Zero 2W to develop a prototype for a Mars Rover. At the core of the project is a machine learning model based on MobileNet, which has been trained using transfer learning to distinguish rocks from other objects.

The software running on the Raspberry Pi Zero 2W utilizes a connected camera to continuously monitor the surroundings. When a rock is detected, an image is captured and automatically uploaded to Google Drive.

## Main Components

- **Hardware**: Raspberry Pi Zero 2W
- **Model**: MobileNet, trained using transfer learning
- **Functionality**: Detection and differentiation of rocks, storing and uploading images to Google Drive

## Setup and Installation

To get this project up and running on your Raspberry Pi Zero 2W, follow these steps:

1. **Clone the Repository:**
    ```sh
    git clone [https://github.com/yourusername/your-repo-name.git](https://github.com/lukasgro63/apm-pi_stone-detection.git)
    ```
2. **Navigate to the Project Directory:**
    ```sh
    cd apm-pi_stone_detection
    ```
3. **Set Up a Python Virtual Environment:**
    ```sh
    python3 -m venv env
    source env/bin/activate
    ```
4. **Install the Required Dependencies:**
    ```sh
    pip install -r requirements.txt
    ```
5. **Configure the Camera and Google Drive Upload:**
    - Ensure your camera is properly connected and configured.
    - Set up Google Drive API credentials and ensure they are placed in the project directory.

## Running the Project

1. **Activate the Python Virtual Environment:**
    ```sh
    source env/bin/activate
    ```
2. **Run the Main Script:**
    ```sh
    python main.py
    ```
3. **Monitor the Output:**
    - The system will continuously monitor the surroundings using the connected camera.
    - When a rock is detected, an image will be captured and uploaded to Google Drive.

