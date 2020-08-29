# augmix-tf
Augmix-tf is an implementation of novel data augmentation [AugMix (2020)]([https://arxiv.org/pdf/1912.02781.pdf](https://arxiv.org/pdf/1912.02781.pdf)) in TensorFlow.  It runs on TPU. 

AugMix utilizes simple augmentation operations which are stochastically sampled and layered to produce a high diversity of augmented images. The process of mixing basic tranformations into augmented image is shown below (picture taken from the original paper). This augmentation performs better when used in concert with Jensen-Shannon Divergence Consistency Loss.
![AugMix pipeline](https://i.ibb.co/YNfsHPF/Capture.png)

## Installation
**Pip package will be available soon... for now do this:**
1. Download ```augmix.py, transformations.py, helpers.py```.
2. Import AugMix class from ```augmix.py```.
3. See examples of usage below.

## Usage
### AugMix
The main function, which does the augmentation is AugMix.transform, let's print a docstring of it. 
```python
from augmix import AugMix
print(AugMix.transform.__doc__)
```
```
	Performs AugMix data augmentation on given image.

	Parameters:
	image (tf tensor): an image tensor with shape (x, x, 3) and values scaled to range [0, 1]
	severity (int): level of a strength of transformations (integer from 1 to 10)
	width (int): number of different chains of transformations to be mixed
	depth (int): number of transformations in one chain, -1 means random from 1 to 3

	Returns:
	tensor: augmented image
```

**Example 1** - transforming a single image
```python
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from augmix import AugMix

# precalculated means and stds of the dataset (in RGB order)
means = [0.44892993872313053, 0.4148519066242368, 0.301880284715257]
stds = [0.24393544875614917, 0.2108791383467354, 0.220427056859487]
ag = AugMix(means, stds)

# preprocess
image = np.asarray(Image.open('geranium.jpg'))
image = tf.convert_to_tensor(image)
image = tf.cast(image, dtype=tf.float32)
image = tf.image.resize(image, (331, 331)) # resize to square
image /=  255  # scale to [0, 1]

# augment
augmented = ag.transform(image)

# visualize
comparison = tf.concat([image, augmented], axis=1)
plt.imshow(comparison.numpy())
plt.title("Original image (left) and augmented image (right).")
plt.show()
```
![result of example 1](https://i.ibb.co/PDZp51S/Figure-1.png))

**Example 2** - transforming a dataset of images
```python
# here a dataset is a tf.data.Dataset object
# assuming images are properly preprocessed (see example 1)
dataset = dataset.map(lambda  img: ag.transform(img))
```
**Example 3** - transforming a dataset to use with the Jensen-Shannon loss
```python
# here a dataset is a tf.data.Dataset object
# assuming images are properly preprocessed (see example 1)
dataset = dataset.map(lambda  img: (img, ag.transform(img), ag.transform(img)))
```
## Visualization

### AugMix
**original images**
![original images](/images/original.png?raw=true)

**augmented**
![visualization of augmix](/images/augmented.png?raw=true)

### Simple transformations
AugMix mixes images transformed by simple augmentations defined in ```transformations.py``` file. Every transformation function takes an image and level parameter that determines a strength of this transformation. This level parameter has the same value as severity parameter in AugMix.transform function, so again it is the integer between 1 and 10, where 10 means the strongest augmentation. These functions can be used by itself. Below is a visualization what every simple augmentation does to a batch of images (at level 10). 



**translate_x, translate_y**
![translate](/images/translate.png?raw=true)

**rotate**
![rotate](/images/rotate.png?raw=true)

**shear_x, shear_y**
![shear](/images/shear.png?raw=true)

**solarize**
![solarize](/images/solarize.png?raw=true)

**solarize_add**
![solarize add](/images/solarize_add.png?raw=true)

**posterize**
![posterize](/images/posterize.png?raw=true)

**autocontrast**
![autocontrast](/images/autocontrast.png?raw=true)

**contrast**
![contrast](/images/contrast.png?raw=true)

**equalize**
![equalize](/images/equalize.png?raw=true)

**brightness**
![brightness](/images/brightness.png?raw=true)

**color**
![color](/images/color.png?raw=true)

## More information
- [AugMix paper](https://arxiv.org/pdf/1912.02781.pdf)
 
 ## TODO
 - batch implementation of AugMix
 - possibility to choose basic transformations easily
 - pip package
 - appendix
	 - calculation of mean and standard devation on a dataset
	 - implementation of Jensen-Shannon Divergence Consistency Loss
 
 ## License
Augmix-tf is released under MIT License. 
