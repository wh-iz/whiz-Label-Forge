# Whiz Label Forge

Whiz Label Forge is a comprehensive AI dataset preparation and training toolkit designed to streamline the lifecycle of object detection models. It integrates video processing, automated labeling, manual correction, and model training into a unified interface.

Built for scalability and experimentation, this tool allows researchers and engineers to rapidly generate high-quality datasets using ensemble inference from multiple YOLO models, refine annotations with a built-in editor, and train new models directly within the application.

## Key Features

*   **Automated Labeling Pipeline**: Leverage multi-model inference to automatically generate bounding box annotations for videos and images.
*   **Multi-Model Ensemble**: Run detection using multiple YOLO models simultaneously to improve recall and precision.
*   **Integrated Label Editor**: A full-featured editor to review, adjust, move, resize, and delete bounding boxes.
*   **Model Training**: Built-in pipeline to train custom YOLO models directly on your generated datasets.
*   **Dataset Management**: Tools for splitting datasets (train/val/test) and analyzing statistics (class counts, distribution).
*   **Flexible Output Options**: Configure output resolution, raw image saving, annotated previews, and handling of empty frames.
*   **GPU Acceleration**: Optimized for CUDA-enabled GPUs with automatic CPU fallback.
*   **Advanced Configuration**:
    *   Automatic cropping to model input size.
    *   Options to save raw, annotated, or empty images.
    *   Configurable confidence and IoU thresholds.

## Supported Models

The toolkit supports a wide range of YOLO architectures for both inference and training:

*   **YOLOv8** (n, s, m, l, x)
*   **YOLOv9** (t, s, m, c, e)
*   **YOLOv10** (n, s, m, b, l, x)
*   **YOLO11** (n, s, m, l, x)

## Workflow

1.  **Data Ingestion**: Import video files or download directly from YouTube.
2.  **Auto-Labeling**: Select one or more pre-trained YOLO models to generate initial annotations.
3.  **Refinement**: Use the Label Editor to correct errors, adjust bounding boxes, and remove low-quality images.
4.  **Dataset Preparation**: Split the refined data into training and validation sets and generate necessary configuration files.
5.  **Training**: Fine-tune a new YOLO model on your custom dataset directly within the application.
6.  **Iterate**: Use the newly trained model to label new data, creating a semi-supervised feedback loop.

## Installation

Ensure Python 3.10+ is installed.

1.  Clone the repository.
2.  Create a virtual environment.
3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4.  (Optional) Install CUDA toolkit for GPU acceleration.

## Usage Overview

Launch the application via the main entry point:

```bash
python App.py
```

*   **Dashboard**: Configure input sources (video/YouTube), select inference models, and set detection parameters (confidence, IoU).
*   **Settings**: Manage paths, system preferences (GPU), and advanced detection parameters.
*   **Label Editor**: Manually inspect and modify dataset annotations.
*   **Tools**: Utilities for dataset splitting and statistical analysis.
*   **Training**: Configure training hyperparameters (epochs, batch size, image size) and start model training.

## Project Origin

This project is a heavily extended fork of [ProLabeler](https://github.com/Goodknight2/ProLabeler). It builds upon the original foundation to provide a more robust, feature-rich environment for end-to-end computer vision workflows.

## Disclaimer

This software is provided for research and educational purposes. Users are responsible for ensuring they have the right to use and annotate the media files processed by this tool.


