# Moving-Camera-Foreground-Detection
Foreground detection in videos captured by moving cameras

Source code of paper: "Real-Time Hysteresis Foreground Detection in Video Captured by Moving Cameras", the 2022 IEEE International Conference on Imaging Systems and Techniques, IST 2022, June 21-23, 2022 ([link](https://ieeexplore.ieee.org/document/9827719))

Foreground detection is an important first step in video analytics. While the stationary cameras facilitate the foreground detection due to the apparent motion between the moving foreground and the still background, the moving cameras make such a task more challenging because both the foreground and the background appear in motion in the video. To tackle this challenging problem, an innovative real-time foreground detection method is presented, that models the foreground and the background simultaneously and works for both moving and stationary cameras. In particular, first, each input video frame is partitioned into a number of blocks. Then, assuming the background takes the majority of each video frame, the iterative pyramidal implementation of the Lucas-Kanade optical flow approach is applied on the centers of the background blocks in order to estimate the global motion and compensate for the camera movements. Subsequently, each block in the background is modeled by a mixture of Gaussian distributions and a separate Gaussian mixture model is constructed for the foreground in order to enhance the classification. However, the errors in motion compensation can contaminate the foreground model with background values. The novel idea of the proposed method matches a set of background samples to their corresponding block for the most recent frames in order to avoid contaminating the foreground model with background samples. The input values that do not fit into either the statistical or the sample-based background models are used to update the foreground model. Finally, the foreground is detected by applying the Bayes classification technique to the major components in the background and foreground models, which removes the false positives caused by the hysteresis effect. Experimental evaluations demonstrate the feasibility of the proposed method in the foreground segmentation when applied to videos in public datasets.

![Untitled](https://user-images.githubusercontent.com/24352869/184511865-1b4452d4-639a-4338-8437-6b85e7689c35.png)


### Run time comparisons
![WeChat Screenshot_20220816101929](https://user-images.githubusercontent.com/24352869/184903029-6f659816-10f3-458a-8957-542ccf180fcb.png)


### Citation
```
@inproceedings{ghahremannezhad2022real,
  title={Real-Time Hysteresis Foreground Detection in Video Captured by Moving Cameras},
  author={Ghahremannezhad, Hadi and Shi, Hang and Liu, Chengjun},
  booktitle={2022 IEEE International Conference on Imaging Systems and Techniques (IST)},
  pages={1--6},
  year={2022},
  organization={IEEE}
}
```
