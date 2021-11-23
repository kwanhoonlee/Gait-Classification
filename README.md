# Gait-Classification
This is our implementation for the paper:

Dawoon Jung, Mau Dung Nguyen, Jooin Han, Mina Park, Kwanhoon Lee, Jinwook Kim, and Kyungryoul Mun (2019). [Deep Neural Network-Based Gait Classification Using Wearable Inertial Sensor Data](https://ieeexplore.ieee.org/document/8857872) 

# Abstract
Human gait has been regarded as a useful behavioral biometric trait for personal identification and authentication. This study aimed to propose an effective approach for classifying gait, measured using wearable inertial sensors, based on neural networks. The 3-axis accelerometer and 3-axis gyroscope data were acquired at the posterior pelvis, both thighs, both shanks, and both feet while 29 semi-professional athletes, 19 participants with normal foot, and 21 patients with foot deformities walked on the 20-meter straight path. The classifier based on the gait parameters and fully connected neural network was developed by applying 4-fold cross-validation to 80% of the total samples. For the test set that consisted of the remaining 20% of the total samples, this classifier showed an accuracy of 93.02% in categorizing the athlete, normal foot, and deformed foot groups. Using the same model validation and evaluation method, up to 98.19% accuracy was achieved from the convolutional neural network-based classifier. This classifier was trained with the gait spectrograms obtained from the time-frequency domain analysis of the raw acceleration and angular velocity data. The classification based only on the pelvic spectrograms exhibited an accuracy of 94.25% even without requiring a time-consuming and resource-intensive process for feature engineering. The notable performance and practicality in gait classification achieved by this study suggest potential applicability of the proposed approaches in the field of biometrics.

# Requirements
1. Python `3.5 ~ 3.7`
2. tensorflow `1.13.0`
3. Keras `2.3.1`
2. sklearn `0.20.0`
3. pandas `0.24.0`
4. numpy `1.16.0`
</br>

# Example to run the codes
```bash
# to train gait-classification model,
python3 gait_n_train.py

```

# Notes

This is an MLP-based gait classification model, which is a baseline model.