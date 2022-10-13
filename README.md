# Unsupervised Video Object Detection
Foreground detection in videos captured by moving cameras

Source code of paper: **"Real-Time Hysteresis Foreground Detection in Video Captured by Moving Cameras", the 2022 IEEE International Conference on Imaging Systems and Techniques, IST 2022, June 21-23, 2022 ([link](https://ieeexplore.ieee.org/document/9827719))**


### How to run
* The program is tested on Windows 10 with OpenCV 3.4.1 in Release x64 mode. It should work with any version of OpenCV 3.
* The .exe file needs one argument which is the path to the video file
* The program is tested with the [DAVIS](https://davischallenge.org/) and [SCBU](https://github.com/CansenJIANG/SCBU) datasets

1. setup Visual Studio with OpenCV ([guide](https://learnopencv.com/code-opencv-in-visual-studio/))
2. add a folder called "results" next to main.cpp in the project directory
3. set the desired parameters in config.xml and also in DCFG.h
4. add the path to the video file in Visual Studio > Project > Properties > Debugging > Command Arguments
5. run the program


### Introduction
Foreground detection is an important first step in video analytics. While the stationary cameras facilitate the foreground detection due to the apparent motion between the moving foreground and the still background, the moving cameras make such a task more challenging because both the foreground and the background appear in motion in the video. To tackle this challenging problem, an innovative real-time foreground detection method is presented, that models the foreground and the background simultaneously and works for both moving and stationary cameras. In particular, first, each input video frame is partitioned into a number of blocks. Then, assuming the background takes the majority of each video frame, the iterative pyramidal implementation of the Lucas-Kanade optical flow approach is applied on the centers of the background blocks in order to estimate the global motion and compensate for the camera movements. Subsequently, each block in the background is modeled by a mixture of Gaussian distributions and a separate Gaussian mixture model is constructed for the foreground in order to enhance the classification. However, the errors in motion compensation can contaminate the foreground model with background values. The novel idea of the proposed method matches a set of background samples to their corresponding block for the most recent frames in order to avoid contaminating the foreground model with background samples. The input values that do not fit into either the statistical or the sample-based background models are used to update the foreground model. Finally, the foreground is detected by applying the Bayes classification technique to the major components in the background and foreground models, which removes the false positives caused by the hysteresis effect. Experimental evaluations demonstrate the feasibility of the proposed method in the foreground segmentation when applied to videos in public datasets.

The real-world applicability of the current methods for foreground detection in moving cameras suffers from high requirements in computational resources and/or low performance in classifying foreground and background.
Here we apply spatial and temporal features for statistical modeling of the background and the foreground separately in order to classify them in real-time.
Each block of the background is modeled using a mixture of Gaussian distributions (MOG) and a set of values sampled randomly in spatial and temporal domains.
At each video frame the Lucas-Kanade optical flow method is applied on the block centers in order to estimate the camera motion and find the corresponding locations between two adjacent frames.
The global motion is then compensated by updating the background models of each block according to the values of its corresponding location in the previous frame.
On the other hand, the foreground is modeled by another MOG which is updated by the input values that do not fit into the background models.

First observation in videos obtained by moving cameras is that the entire captured scene appears to be moving from the camera's perspective. 
By assuming the background to occupy the majority of the scene compared to the objects of interest we can estimate the motion of the camera relative to the background.
Afterwards, the estimated camera motion can be compensated by using the corresponding values in the previous frame for updating background models.
Then the foreground can be segmented using approaches similar to the methods used for the applications of stationary cameras.
Here, we apply an MOG to model the entire foreground using the values that are not absorbed by the background models.
The major components of the Gaussian mixture distributions in the background and foreground models are utilized for final binary classification.
The details of each step are described in this section.

### Global motion estimation
In many scenarios the objects of interest occupy a portion each video frame and the remaining majority is considered to be background.
Therefore, the majority of point displacements among video frames is caused by the camera motion which can be estimated by calculating the global motion.
For the sake of computational efficiency and accounting for spatial relationships, a similar approach to is applied where the input image is converted to grayscale and divided into a number of grids with equal sizes.
The Kanade–Lucas–Tomasi feature tracking approach is applied on the centers of the grid cells from the previous frame.
Then a homography matrix is obtained that warps the image pixels at frame $t$ to pixels at frame $t-1$ through a perspective transform.
If we denote the intensity of the grayscale image at time $t$ by $I^{(t)}$ and assume consistent intensity between consecutive frames, the corresponding location of each point in the new frame can be used to calculate the global velocity vector as follows:

$$I^{(t)}(x_i+u_i, y_i+v_i)=I^{(t-1)}(x_i,y_i)$$

where $(u_i,v_i)$ is the velocity vector of the center point of the $i$-th block located at $(x_i,y_i)$.
Three-dimensional vectors $X_i$ can be constructed as:

$$ X_i^{(t-1)} = (x_i,y_i,1)^T, \quad X_i^{(t)} = (x_i+u_i,y_i+v_i,1)^T $$

and a reverse transformation matrix $H_{t:t-1}$ is obtained that satisfies the velocity equation for the largest possible number of samples:

$$ \left[X_1^{(t)}, X_2^{(t)}, ...\right] = \mathbf{H}_{t:t-1}\left[X_1^{(t-1)}, X_2^{(t-1)}, ...\right] $$

which is solved by applying the by RANSAC algorithm in order to remove outliers from the further calculations.
Also the center points of the blocks classified as foreground in the previous frame are excluded from this calculation as they do not contribute to the camera motion.

![image](https://user-images.githubusercontent.com/24352869/187810502-92863bac-4748-46a7-af4d-284ede5e255d.png)

### Background and foreground modeling
Each block of the image is modeled by a mixture of Gaussian distributions and the model is updated at each video frame.
In order to update the background models at each frame we have to calculate the corresponding values in the warped background image of the previous frame.
The mean and variance of the warped background model is calculated as a weighted sum of the neighboring models where each weight is proportional to a rectangular area as a bilinear interpolation:

$$	\begin{gathered}
	\tilde{\mu}_i^{(t-1)} = \sum_{k \in \mathcal{R}_i}\omega_k \mu_k^{(t-1)} \\
	\tilde{\sigma}_i^{(t-1)} = \sum_{k \in \mathcal{R}_i}\omega_k \sigma_k^{(t-1)}
	\end{gathered} $$

where $\mathcal{R}$ is a set of block indices falling in a rectangular region centered at the corresponding point location calculated by the homography matrix, $\omega_k$ is the weight that indicates the overlapping area between the block $i$ and the corresponding neighbor $k$, and $\mu$ and $\sigma$ represent the mean and variance of the Gaussian distributions, respectively.

Since the camera might have movements in the form of pan there can be slight variations in the illumination due to the changes in the angle of view and light direction.
The Gaussian modeling keeps the information of previous frames and might be slow in catching up with the pace of changing values at the borders of the video frames.
In order to make the model parameters adapt to these changes a global variation factor $g$ is calculated by subtracting the mean intensities in the background model and the current frame:

$$ g^{(t)} = \frac{1}{N}\sum_{j=1}^{N}I_j^{(t)} - \frac{1}{B}\sum_{i=1}^{B}\tilde{\mu}_i^{(t-1)} $$

with $B$ being the number of blocks and $N$ being the number of pixels.
At each frame the parameters of the Gaussian mixture model for each block are updated as follows:

$$ \begin{gathered}
		\mu_k^{(t)} = \left(n_k^{(t-1)}\left(\tilde{\mu}_k^{(t-1)} + g^{(t)}\right) + M^{(t)}\right) / (n_k^{(t-1)} + 1) \\
		\sigma_k^{(t)} = \left(n_k^{(t-1)}\tilde{\sigma}_k^{(t-1)} + V^{(t-1)}\right) / (n_k^{(t-1)} + 1) \\
		n_k^{(t)} = n_k^{(t-1)} + 1 \\
		\alpha_k^{(t)} = n_k^{(t)} / \sum_{k=1}^{K}n_k^{(t)}
	\end{gathered} $$
	
where $n_k$ is a counter representing the number of times an input value has been used to update component $k$, $\alpha_k$ is the weight of the $k$th component, $M$ and $V$ stand for the mean intensity and the variance of the block, respectively.
The component with the largest weight of each Gaussian mixture model is considered to be the background value of the block.

![image](https://user-images.githubusercontent.com/24352869/187810764-fd62bce3-7aab-4fee-a284-460486cc4a4e.png)

In case of moving cameras the objects of interest are usually present in the scene for a longer time as the camera is focused on them.
Therefore, it is reasonable to model the values of the foreground objects throughout the video.
A similar approach to background modeling is applied for modeling the foreground except only one mixture of Gaussian distributions is used for the entire foreground pixels.
Also, instead of a single component, a number of components from the foreground model that have the largest weights are considered to represent the foreground objects.
This is because the foreground objects have multiple parts with different intensity values and each major component in the foreground model is used to represent one part of the foreground.

![image](https://user-images.githubusercontent.com/24352869/187811037-ae24b56f-4d88-4135-a55f-f5911e662e6e.png)

![Untitled](https://user-images.githubusercontent.com/24352869/184511865-1b4452d4-639a-4338-8437-6b85e7689c35.png)

In addition to the statistical modeling and inspired by the ViBe method, we keep a set of sample values as a secondary non-parametric model for each block.
This set is initialized by the mean value of the block and its neighboring blocks at the first frame.
At each of the consecutive frames one of the values in the set is selected randomly and replaced with the new mean value.
We can denote the collection of background sample values for the block $i$ as $\mathcal{S}_i$ as follows:

$$ \mathcal{S}_i = \{s_i^1, s_i^2, ... , s_i^K\} $$

where $s_i^k$ is the $k$th sampled mean intensity of block $i$.
The sample-based model is kept and updated mainly to avoid contaminating the foreground model by the background values that do not fit into any of the Gaussian components of the corresponding block model.
This problem occurs mostly because of motion compensation errors or new background values being introduced into the scene due to the camera motion.
If an input value does not fit into any of the Gaussian components of a background model the Euclidean distance between the pixel value and each background sample in the set of the corresponding block is calculated.
If the number of samples in the set of block $i$ that are closer than a distance threshold to the input value is less than a counting threshold, the foreground model is updated by that value.
Representing this number of samples by $C_i$ it can be calculated as follows:

$$ C_i = \sum_{j = 1}^{|\mathcal{S}_i|}1\left(D(\mathbf{x},\tilde{s}_j^{(s)}) < \theta_d\right) $$

with $\mathbf{x}$ being the input pixel intensity value, $D$ representing the Euclidean distance, $\theta_d$ being a predefined threshold, which is set to $20$, $1$ denoting an indicator function, $\tilde{s}_j^{(s)}$ representing the corresponding value of $s_i^{(k)}$ after motion compensation, and $\mathcal{S}_i$ denoting the set of neighboring blocks.
We calculate the Euclidean distances between the new values and the samples in the set and only classify the new values as foreground if they match with less than a few samples in the set.
The foreground model is only updated with values that belong to the foreground class with a high certainty and therefore, the majority of false positive cases are avoided.


### Background and foreground classification
For the final classification, at first the foreground likelihood values are calculated for each pixel at an input image as follows:

$$ L_{fg}(x,y) = \frac{\left(I(x,y) - \mu_k\right)^2}{\sigma_k} $$

where $I(x,y)$ and $L_{fg}(x,y)$ are the intensity and foreground likelihood values of the pixel at location $(x,y)$, and $\mu_k$ and $\sigma_k$ are the mean and variance of the corresponding background block, respectively.
Afterwards, the watershed segmentation algorithm is applied to each input image in order to extract a set of super-pixels, notated by $\mathbb{P} = \{P_1,P_2,...,P_k\}$.

For final classification the mean value of each super-pixel is compared against the major component in the background model of the corresponding block as well as each component in the foreground model.
The foreground confidence map $\mathcal{F}$ is obtained by calculating the mean of confidence values in each super-pixel as follows:

$$ \mathcal{F}(P_i) = \frac{1}{|P_i|}\sum_{x,y \in P_i}L_{fg}\left(x,y\right) $$

where $|P_i|$ is the number of pixels at super-pixel $P_i$.
Assuming there are $M$ major components in the global foreground model, a background confidence map $\mathcal{B}_m, m\in\{1,...,M\}$ is similarly obtained based on each component.
The Gaussian Naive Bayes (GNB) classifier is applied for each super-pixel in order to calculate the z-score distance between the input value and each class-mean and classify the super-pixel accordingly in order to obtain the final foreground mask $\mathcal{H}$:

$$ \mathcal{H}(P_i) = 1, \text{if } \mathcal{F}(P_i) > \mathcal{B}_m(P_i) \quad and \quad 0, \quad \text{otherwise}  $$

where $\mathcal{B}_m$ is the background confidence map corresponding to the $m$-th foreground model and $\mathcal{H}(P_i)=1$ indicates that the super-pixel at location $P_i$ belongs to the moving objects and $\mathcal{H}(P_i)=0$ means it belongs to the background.
The process of segmenting the foreground is detailed in the algorithm.

The different stages in the classification process can be seen in the figure .
From top to bottom, each row in the figure represents a sample video frame from the DAVIS, Segment Pool Tracking, and SCBU datasets, respectively.
The second column represents heatmaps where the pixels with a higher probability of belonging to the foreground are represented by red colors.
The third column is the results of the watershed segmentation algorithm applied to each video frame with the markers chosen uniformly across the image at the same locations as the background block centers.
The fourth column illustrates the foreground confidence maps calculated based on the equation and the last column is the final results of foreground detection after morphological dilation.

![image](https://user-images.githubusercontent.com/24352869/187811395-6179984b-fb6a-45a6-b0f7-eb07eec4335c.png)

![image](https://user-images.githubusercontent.com/24352869/187811482-1266882b-0bb8-42ec-86b6-892195a952e2.png)

### Experiments
The performance of the proposed method is evaluated using video data collected from the publicly available SCBU dataset that consists of nine video sequences captured by moving cameras.
The videos in the dataset impose various challenges in the way of foreground segmentation, such as fast or slow moving objects, objects with different sizes, illumination changes, and the similarities in intensity values between the background and foreground.
In terms of time and space complexity, the statistical methods are more efficient as the methods based on deep neural networks require more resources.
Therefore, our method is more practical in applications with real-time requirements and edge devices that have a lower hardware capacity.
It can be seen that our proposed method is able to detect the foreground in various challenging scenarios.
Compared to some of the representative methods, such as MCD and MCD NP, our method models the foreground and background separately, which enhances the classifications results.
One of the limitations in the proposed method is the ability of the foreground model to adapt well to sudden illumination changes caused by the pan movements of the camera.
Also, the camouflage problem, where the foreground color values are very similar to those of the corresponding background block, can lead to false negative results.
This problem can be solved by introducing more discriminating features to the statistical modeling process in future studies.
The f-score metric is used in order to evaluate the quantitative results:

$$\begin{gathered} 
PRE=T_P/(T_P+F_P)  \quad
 REC=T_P/(T_P+F_N) \quad
F_1= 2 \times (PRE \times REC)/(PRE+REC) 
\end{gathered}$$

where $T_P$, $F_P$ are the number of pixels correctly and incorrectly reported as foreground, and $T_N$ and $F_N$ are the number of pixels that are correctly and incorrectly reported as background, respectively.
$PRE$, $REC$, and $F_1$ refer to precision, recall, and F1-score, respectively.
The F1-scores are listed the table in comparison with other popular methods.
The quantitative results demonstrate the robustness of our method in detecting the foreground mask in different videos.
The hardware specification used for the experiments is a 3.4 GHz processor and 16 GB RAM.
The average processing speed for video frames of size $320 \times 240$ pixels was about $\sim143$ frames per second, which is feasible for real-time applications of video analytics.



### Run time comparisons
The average running speed of the proposed method is reported in the table for each video frame of size $320\times240$ pixels.
The run-time calculations show that the method is feasible to be used as a pre-processing step in real-time traffic video analysis tasks.
It can be seen that our method improves the performance significantly while being almost as fast as other real-time methods, such as MCD.

![WeChat Screenshot_20220816101929](https://user-images.githubusercontent.com/24352869/184903029-6f659816-10f3-458a-8957-542ccf180fcb.png)

### Conclusion
In this study, a new real-time method is proposed for locating the moving objects in videos captured by non-stationary cameras, which is one the challenging problems in computer vision.
The global motion is estimated and used to compensate for background variations caused by the camera movements.
Each block is modeled by a mixture of Gaussian distributions which is updated by the values at the corresponding locations in the warped image after motion compensation.
Additionally, the mean values of each block are modeled along with the mean values of its neighboring blocks as a set of samples which is in turn updated by random selection.
The foreground on the other hand is modeled by a separate MOG which is updated by values that do not fit into either of the statistical or sample-based background models.
For classification, each input value is compared against both the background and foreground models to obtain the definite and the candidate foreground locations, respectively.
The watershed segmentation algorithm is then applied to detect the final foreground mask.
Experimental results demonstrate the feasibility of the proposed method in real-time video analytics systems.

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


https://user-images.githubusercontent.com/24352869/187813915-3719413c-fbb2-4c3e-a23f-d1801969e8b4.mp4

https://user-images.githubusercontent.com/24352869/187813956-de0a87ae-e4dc-4b20-9f7e-cf26dcab5609.mp4

https://user-images.githubusercontent.com/24352869/187813979-660a07d2-5a41-4eb0-a52b-e69c2d8f9591.mp4

https://user-images.githubusercontent.com/24352869/187814447-d2466e14-1edc-41be-a543-e0df56e9749b.mp4

https://user-images.githubusercontent.com/24352869/187814452-adb6a021-c38a-4a45-ac00-f2e665383f1d.mp4

https://user-images.githubusercontent.com/24352869/187814439-cd0310c6-e15b-4554-a6c1-340596c5f47c.mp4

