# Face Detection Model

This directory should contain the [YuNet face detection](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) ONNX weights used by `cap.py`.

1. Download the latest `face_detection_yunet_2023mar.onnx` file:
   ```bash
   wget https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
   ```
2. Move the file into this folder so the final path is:
   ```
   final/models/face_detection_yunet_2023mar.onnx
   ```

Without this file, the Arducam process will exit early and you will only see the ZoomCam view.
