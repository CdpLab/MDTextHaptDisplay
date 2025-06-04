# Multi-Dimensional Texture Haptic Modeling and Display Method

<p align="center">
Dapeng Chen, Yi Ding, Geng Chen, Tianyu Fan, Jia Liu, and Aiguo Song  
</p>

<p align="center">
Nanjing University of Information Science & Technology  
</p>

<br/>

<h3 align="center">ABSTRACT</h3>

The haptic perception of texture has significant multi-dimensional characteristics, mainly composed of key attributes such as roughness, friction, and hardness. These dimensions not only reflect the microstructure of object surfaces, but also serve as the foundation for achieving real haptic feedback in virtual interactions. The integration of multi-dimensional perception enhances the consistency and immersion of haptic rendering, especially in dynamic interactions that integrate user action information. However, existing modeling methods often focus on a single dimension, which is difficult to meet the demand for high immersion and adaptability. Therefore, we proposed an end-to-end multi-dimensional texture haptic rendering model. This model takes the Mamba encoder as the core, integrates texture images and the real-time action information of users, and jointly predicts high-quality acceleration and friction signals. To train and validate the model, a set of equipment for collecting sliding speed and normal pressure was designed. Data interacting with 70 real textures were collected, and a training set was constructed in combination with the SENS3 dataset. The model demonstrated superior predictive ability and generalization in four performance tests, with the delay of the haptic reproduction system controlled between 32$\sim$38 ms, below the 40 ms haptic perception threshold. Finally, we conducted a user experiment. The results indicate that the rendering model that integrates roughness and friction feedback significantly improves the accuracy of users in distinguishing between real and virtual textures, effectively enhancing the realistic experience of virtual textures.

<h3 align="center">MULTI-DIMENSIONAL TEXTURE HAPTIC RENDERING MODEL</h3>

The objective of this study is to establish a mapping relationship between multi-modal information (texture images *i*, sliding speed *v*, and normal pressure *p*) and multi-dimensional haptic outputs (acceleration signal *a* and friction signal *f*), which can be represented as *g(i, v, p) → a, f*, where *g* represents the prediction model. Taking inspiration from existing work, this paper designs a multi-dimensional texture haptic rendering model based on real-time action information, and its overall structure is shown below.

<p align="center">
  <img src="https://github.com/user-attachments/assets/a59cd66e-ffbf-4de1-a834-ec9a20d8853f" alt="fig1" width="600"/>
</p>

<h3 align="center">DETAILS OF IMPLEMENT</h3>

### Dataset
We used the SENS3 dataset and the 70 real texture sample data we collected as the training and testing sets for the model. The SENS3 dataset covers 50 different texture images from 10 categories, and includes data on three-axis forces, torques, and velocities recorded by experimenters sliding on each texture surface for 5 seconds under controlled force velocity matrix conditions. We take the force component in the *z* axis direction as the compressive force, and calculate the frictional force by synthesizing the force components in the *x* and *y* axes. Its magnitude can be expressed as: *f = √(Fx² + Fy²)*.
