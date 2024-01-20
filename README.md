# Magiclick

## Introduction
With the widespread popularity of smart mobile devices, mobile video recording has become an important means for us to capture the wonderful moments of life. However, these beautiful moments are fleeting, and users often do not have enough time to compose their shots thoughtfully. Additionally, most people lack the skills and awareness for post-processing, resulting in many processed video works lacking basic aesthetic value, often leading to disappointing results. In view of this, a simple and fast interactive platform can quickly and accurately meet user needs. MAGICROP (Wang et al., 2023) has already made significant contributions in the area of image cropping. However, there is currently no effective cropping algorithm for video input. Treating video frames as independent images may lead to unclear semantic expression. In this project, we propose a new model, MAGICLICK, which allows users to select prominent themes and provides options for users to choose cropping ratios, or the model can decide automatically, through an appropriate segmentation model. We have adopted an innovative cropping method based on dynamic programming, combining the importance sampling of video frames from different cropping results, and using the Neural Image Assessment (NIMA) model (Talebi et al., 2018) for aesthetic evaluation, allowing users to easily choose their desired cropped video.

Our main work includes:

1. Using video as input, and building upon the MAGICROP framework, we combine the user's immediate prompts (usually a single click), using the Track Anything Model (Yang et al., 2023) as the segmentation model, to highlight and track the selected video subject in real-time.
2. We employ a cropping method based on dynamic programming, whose evaluation function is based on the amplitude of changes in the video subject. Our algorithm comprehensively considers the prominence of the video subject and the changes in the scene to achieve the best cropping result under specified cropping ratios and center positioning.
3. We propose an importance sampling algorithm based on the amplitude of video frame changes, using the NIMA model to perform aesthetic evaluations on weighted video frames, providing score guidance for user choices.

## start the app
```bash
python app.py
```

