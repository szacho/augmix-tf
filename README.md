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
The main function, which does the augmentation is AugMix.process, let's print a docstring of it. 
```python
from augmix import AugMix
print(AugMix.process.__doc__)
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

# means and stds calculation in appendix
means = [0.44892993872313053, 0.4148519066242368, 0.301880284715257]
stds = [0.24393544875614917, 0.2108791383467354, 0.220427056859487]
augmix_transformer = AugMix(means, stds)

# preprocess
image = np.asarray(Image.open('geranium.jpg'))
image = tf.convert_to_tensor(image)
image = tf.cast(image, dtype=tf.float32)
image = tf.image.resize(image, (331, 331)) # resize to square
image /=  255  # scale to [0, 1]

# augment
augmented = augmix_transformer.process(image)

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
dataset = dataset.map(lambda  img: augmix.process(img))
```
**Example 3** - transforming a dataset to use with the Jensen-Shannon loss
```python
# here a dataset is a tf.data.Dataset object
# assuming images are properly preprocessed (see example 1)
dataset = dataset.map(lambda  img: (img, augmix.process(img), augmix.process(img)))
```
## Visualization

### AugMix
**original images**
![original images](https://www.kaggleusercontent.com/kf/31989134/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..kwjHRExsm3xFPskDr9KEXA.rU8dD6grh1GgWuAcAhpZDVlfZnqbZ63VM5dTYKo85TXf5g6YB6OT7f1D_ourPAO5fef2G8lBZHz4LI8An8qbfdbiD6uJc-Jj2rW0NLagRew3W4sI9ZaZMbP1SAdb_yei6RUa2xvf_OMoZb9ypET5SMikER9mEm--nfX74M7ULiQSZ_FdyjEAkwRX-r1CabltZYyKdCNGMQzdOaEqP9PGdWJqw2mfEqPZkQKib4EfJHCccotNVsqk9GARAekPmJ73FKo4Z4SrX55j9UeWd02mXjog4ONW_Q_rG1imjPtV5Tl7GQWcQl5PkE-i9EzAL7-9Uo-YT1LCLsnankmWbg1VQgfWMkwsoWAuh2fa3oZprm8XUjZwln2Dts3Na5i9cYNEH6WFpiC8b2kaRjh5WPbo6dIg9PvVSAk_gOKAJMMA515IG0MoDQQ92RKPpQmCQap9eonDTjWR8Jx4wc28OJyEIUCYtMeNdiIAxXvMo8EXHYEAHDkaaauF3V-FCRpnki-cI6BlPE57xM5qv-8YydtOLeEjitqBgiyECWHp4OGQqvl2kn_7_Q22ldA8gUEA1_gYDolBUqtSsfAvixlzUQxXRrzDAeoF5jWqVcfopijOwIh1tEfehBxC4swjM4yZGM773CfODH8X_NOhBGXLQcwb9QQqV10dMFaiNH7girqrnQk.ntbvJw9RfGBJt8atbB-sAQ/__results___files/__results___21_0.png)

**augmented**
![visualization of augmix](https://www.kaggleusercontent.com/kf/31989134/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..kwjHRExsm3xFPskDr9KEXA.rU8dD6grh1GgWuAcAhpZDVlfZnqbZ63VM5dTYKo85TXf5g6YB6OT7f1D_ourPAO5fef2G8lBZHz4LI8An8qbfdbiD6uJc-Jj2rW0NLagRew3W4sI9ZaZMbP1SAdb_yei6RUa2xvf_OMoZb9ypET5SMikER9mEm--nfX74M7ULiQSZ_FdyjEAkwRX-r1CabltZYyKdCNGMQzdOaEqP9PGdWJqw2mfEqPZkQKib4EfJHCccotNVsqk9GARAekPmJ73FKo4Z4SrX55j9UeWd02mXjog4ONW_Q_rG1imjPtV5Tl7GQWcQl5PkE-i9EzAL7-9Uo-YT1LCLsnankmWbg1VQgfWMkwsoWAuh2fa3oZprm8XUjZwln2Dts3Na5i9cYNEH6WFpiC8b2kaRjh5WPbo6dIg9PvVSAk_gOKAJMMA515IG0MoDQQ92RKPpQmCQap9eonDTjWR8Jx4wc28OJyEIUCYtMeNdiIAxXvMo8EXHYEAHDkaaauF3V-FCRpnki-cI6BlPE57xM5qv-8YydtOLeEjitqBgiyECWHp4OGQqvl2kn_7_Q22ldA8gUEA1_gYDolBUqtSsfAvixlzUQxXRrzDAeoF5jWqVcfopijOwIh1tEfehBxC4swjM4yZGM773CfODH8X_NOhBGXLQcwb9QQqV10dMFaiNH7girqrnQk.ntbvJw9RfGBJt8atbB-sAQ/__results___files/__results___45_0.png)

### Simple transformations
AugMix mixes images transformed by simple augmentations defined in ```transformations.py``` file. Every transformation function takes an image and level parameter that determines a strength of this transformation. This level parameter has the same value as severity parameter in AugMix.process function, so again it is the integer between 1 and 10, where 10 means the strongest augmentation. These functions can be used by itself. Below is a visualization what every simple augmentation does to a batch of images (at level 10). 



**translate_x, translate_y**
![translate](https://www.kaggleusercontent.com/kf/31989134/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..kwjHRExsm3xFPskDr9KEXA.rU8dD6grh1GgWuAcAhpZDVlfZnqbZ63VM5dTYKo85TXf5g6YB6OT7f1D_ourPAO5fef2G8lBZHz4LI8An8qbfdbiD6uJc-Jj2rW0NLagRew3W4sI9ZaZMbP1SAdb_yei6RUa2xvf_OMoZb9ypET5SMikER9mEm--nfX74M7ULiQSZ_FdyjEAkwRX-r1CabltZYyKdCNGMQzdOaEqP9PGdWJqw2mfEqPZkQKib4EfJHCccotNVsqk9GARAekPmJ73FKo4Z4SrX55j9UeWd02mXjog4ONW_Q_rG1imjPtV5Tl7GQWcQl5PkE-i9EzAL7-9Uo-YT1LCLsnankmWbg1VQgfWMkwsoWAuh2fa3oZprm8XUjZwln2Dts3Na5i9cYNEH6WFpiC8b2kaRjh5WPbo6dIg9PvVSAk_gOKAJMMA515IG0MoDQQ92RKPpQmCQap9eonDTjWR8Jx4wc28OJyEIUCYtMeNdiIAxXvMo8EXHYEAHDkaaauF3V-FCRpnki-cI6BlPE57xM5qv-8YydtOLeEjitqBgiyECWHp4OGQqvl2kn_7_Q22ldA8gUEA1_gYDolBUqtSsfAvixlzUQxXRrzDAeoF5jWqVcfopijOwIh1tEfehBxC4swjM4yZGM773CfODH8X_NOhBGXLQcwb9QQqV10dMFaiNH7girqrnQk.ntbvJw9RfGBJt8atbB-sAQ/__results___files/__results___23_0.png)

**rotate**
![rotate](https://www.kaggleusercontent.com/kf/31989134/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..kwjHRExsm3xFPskDr9KEXA.rU8dD6grh1GgWuAcAhpZDVlfZnqbZ63VM5dTYKo85TXf5g6YB6OT7f1D_ourPAO5fef2G8lBZHz4LI8An8qbfdbiD6uJc-Jj2rW0NLagRew3W4sI9ZaZMbP1SAdb_yei6RUa2xvf_OMoZb9ypET5SMikER9mEm--nfX74M7ULiQSZ_FdyjEAkwRX-r1CabltZYyKdCNGMQzdOaEqP9PGdWJqw2mfEqPZkQKib4EfJHCccotNVsqk9GARAekPmJ73FKo4Z4SrX55j9UeWd02mXjog4ONW_Q_rG1imjPtV5Tl7GQWcQl5PkE-i9EzAL7-9Uo-YT1LCLsnankmWbg1VQgfWMkwsoWAuh2fa3oZprm8XUjZwln2Dts3Na5i9cYNEH6WFpiC8b2kaRjh5WPbo6dIg9PvVSAk_gOKAJMMA515IG0MoDQQ92RKPpQmCQap9eonDTjWR8Jx4wc28OJyEIUCYtMeNdiIAxXvMo8EXHYEAHDkaaauF3V-FCRpnki-cI6BlPE57xM5qv-8YydtOLeEjitqBgiyECWHp4OGQqvl2kn_7_Q22ldA8gUEA1_gYDolBUqtSsfAvixlzUQxXRrzDAeoF5jWqVcfopijOwIh1tEfehBxC4swjM4yZGM773CfODH8X_NOhBGXLQcwb9QQqV10dMFaiNH7girqrnQk.ntbvJw9RfGBJt8atbB-sAQ/__results___files/__results___27_0.png)

**shear_x, shear_y**
![shear](https://www.kaggleusercontent.com/kf/31989134/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..kwjHRExsm3xFPskDr9KEXA.rU8dD6grh1GgWuAcAhpZDVlfZnqbZ63VM5dTYKo85TXf5g6YB6OT7f1D_ourPAO5fef2G8lBZHz4LI8An8qbfdbiD6uJc-Jj2rW0NLagRew3W4sI9ZaZMbP1SAdb_yei6RUa2xvf_OMoZb9ypET5SMikER9mEm--nfX74M7ULiQSZ_FdyjEAkwRX-r1CabltZYyKdCNGMQzdOaEqP9PGdWJqw2mfEqPZkQKib4EfJHCccotNVsqk9GARAekPmJ73FKo4Z4SrX55j9UeWd02mXjog4ONW_Q_rG1imjPtV5Tl7GQWcQl5PkE-i9EzAL7-9Uo-YT1LCLsnankmWbg1VQgfWMkwsoWAuh2fa3oZprm8XUjZwln2Dts3Na5i9cYNEH6WFpiC8b2kaRjh5WPbo6dIg9PvVSAk_gOKAJMMA515IG0MoDQQ92RKPpQmCQap9eonDTjWR8Jx4wc28OJyEIUCYtMeNdiIAxXvMo8EXHYEAHDkaaauF3V-FCRpnki-cI6BlPE57xM5qv-8YydtOLeEjitqBgiyECWHp4OGQqvl2kn_7_Q22ldA8gUEA1_gYDolBUqtSsfAvixlzUQxXRrzDAeoF5jWqVcfopijOwIh1tEfehBxC4swjM4yZGM773CfODH8X_NOhBGXLQcwb9QQqV10dMFaiNH7girqrnQk.ntbvJw9RfGBJt8atbB-sAQ/__results___files/__results___25_0.png)

**solarize**
![solarize](https://www.kaggleusercontent.com/kf/31989134/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..kwjHRExsm3xFPskDr9KEXA.rU8dD6grh1GgWuAcAhpZDVlfZnqbZ63VM5dTYKo85TXf5g6YB6OT7f1D_ourPAO5fef2G8lBZHz4LI8An8qbfdbiD6uJc-Jj2rW0NLagRew3W4sI9ZaZMbP1SAdb_yei6RUa2xvf_OMoZb9ypET5SMikER9mEm--nfX74M7ULiQSZ_FdyjEAkwRX-r1CabltZYyKdCNGMQzdOaEqP9PGdWJqw2mfEqPZkQKib4EfJHCccotNVsqk9GARAekPmJ73FKo4Z4SrX55j9UeWd02mXjog4ONW_Q_rG1imjPtV5Tl7GQWcQl5PkE-i9EzAL7-9Uo-YT1LCLsnankmWbg1VQgfWMkwsoWAuh2fa3oZprm8XUjZwln2Dts3Na5i9cYNEH6WFpiC8b2kaRjh5WPbo6dIg9PvVSAk_gOKAJMMA515IG0MoDQQ92RKPpQmCQap9eonDTjWR8Jx4wc28OJyEIUCYtMeNdiIAxXvMo8EXHYEAHDkaaauF3V-FCRpnki-cI6BlPE57xM5qv-8YydtOLeEjitqBgiyECWHp4OGQqvl2kn_7_Q22ldA8gUEA1_gYDolBUqtSsfAvixlzUQxXRrzDAeoF5jWqVcfopijOwIh1tEfehBxC4swjM4yZGM773CfODH8X_NOhBGXLQcwb9QQqV10dMFaiNH7girqrnQk.ntbvJw9RfGBJt8atbB-sAQ/__results___files/__results___29_0.png)

**solarize_add**
![solarize add](https://www.kaggleusercontent.com/kf/31989134/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..kwjHRExsm3xFPskDr9KEXA.rU8dD6grh1GgWuAcAhpZDVlfZnqbZ63VM5dTYKo85TXf5g6YB6OT7f1D_ourPAO5fef2G8lBZHz4LI8An8qbfdbiD6uJc-Jj2rW0NLagRew3W4sI9ZaZMbP1SAdb_yei6RUa2xvf_OMoZb9ypET5SMikER9mEm--nfX74M7ULiQSZ_FdyjEAkwRX-r1CabltZYyKdCNGMQzdOaEqP9PGdWJqw2mfEqPZkQKib4EfJHCccotNVsqk9GARAekPmJ73FKo4Z4SrX55j9UeWd02mXjog4ONW_Q_rG1imjPtV5Tl7GQWcQl5PkE-i9EzAL7-9Uo-YT1LCLsnankmWbg1VQgfWMkwsoWAuh2fa3oZprm8XUjZwln2Dts3Na5i9cYNEH6WFpiC8b2kaRjh5WPbo6dIg9PvVSAk_gOKAJMMA515IG0MoDQQ92RKPpQmCQap9eonDTjWR8Jx4wc28OJyEIUCYtMeNdiIAxXvMo8EXHYEAHDkaaauF3V-FCRpnki-cI6BlPE57xM5qv-8YydtOLeEjitqBgiyECWHp4OGQqvl2kn_7_Q22ldA8gUEA1_gYDolBUqtSsfAvixlzUQxXRrzDAeoF5jWqVcfopijOwIh1tEfehBxC4swjM4yZGM773CfODH8X_NOhBGXLQcwb9QQqV10dMFaiNH7girqrnQk.ntbvJw9RfGBJt8atbB-sAQ/__results___files/__results___31_0.png)

**posterize**
![posterize](https://www.kaggleusercontent.com/kf/31989134/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..kwjHRExsm3xFPskDr9KEXA.rU8dD6grh1GgWuAcAhpZDVlfZnqbZ63VM5dTYKo85TXf5g6YB6OT7f1D_ourPAO5fef2G8lBZHz4LI8An8qbfdbiD6uJc-Jj2rW0NLagRew3W4sI9ZaZMbP1SAdb_yei6RUa2xvf_OMoZb9ypET5SMikER9mEm--nfX74M7ULiQSZ_FdyjEAkwRX-r1CabltZYyKdCNGMQzdOaEqP9PGdWJqw2mfEqPZkQKib4EfJHCccotNVsqk9GARAekPmJ73FKo4Z4SrX55j9UeWd02mXjog4ONW_Q_rG1imjPtV5Tl7GQWcQl5PkE-i9EzAL7-9Uo-YT1LCLsnankmWbg1VQgfWMkwsoWAuh2fa3oZprm8XUjZwln2Dts3Na5i9cYNEH6WFpiC8b2kaRjh5WPbo6dIg9PvVSAk_gOKAJMMA515IG0MoDQQ92RKPpQmCQap9eonDTjWR8Jx4wc28OJyEIUCYtMeNdiIAxXvMo8EXHYEAHDkaaauF3V-FCRpnki-cI6BlPE57xM5qv-8YydtOLeEjitqBgiyECWHp4OGQqvl2kn_7_Q22ldA8gUEA1_gYDolBUqtSsfAvixlzUQxXRrzDAeoF5jWqVcfopijOwIh1tEfehBxC4swjM4yZGM773CfODH8X_NOhBGXLQcwb9QQqV10dMFaiNH7girqrnQk.ntbvJw9RfGBJt8atbB-sAQ/__results___files/__results___33_0.png)

**autocontrast**
![autocontrast](https://www.kaggleusercontent.com/kf/31989134/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..kwjHRExsm3xFPskDr9KEXA.rU8dD6grh1GgWuAcAhpZDVlfZnqbZ63VM5dTYKo85TXf5g6YB6OT7f1D_ourPAO5fef2G8lBZHz4LI8An8qbfdbiD6uJc-Jj2rW0NLagRew3W4sI9ZaZMbP1SAdb_yei6RUa2xvf_OMoZb9ypET5SMikER9mEm--nfX74M7ULiQSZ_FdyjEAkwRX-r1CabltZYyKdCNGMQzdOaEqP9PGdWJqw2mfEqPZkQKib4EfJHCccotNVsqk9GARAekPmJ73FKo4Z4SrX55j9UeWd02mXjog4ONW_Q_rG1imjPtV5Tl7GQWcQl5PkE-i9EzAL7-9Uo-YT1LCLsnankmWbg1VQgfWMkwsoWAuh2fa3oZprm8XUjZwln2Dts3Na5i9cYNEH6WFpiC8b2kaRjh5WPbo6dIg9PvVSAk_gOKAJMMA515IG0MoDQQ92RKPpQmCQap9eonDTjWR8Jx4wc28OJyEIUCYtMeNdiIAxXvMo8EXHYEAHDkaaauF3V-FCRpnki-cI6BlPE57xM5qv-8YydtOLeEjitqBgiyECWHp4OGQqvl2kn_7_Q22ldA8gUEA1_gYDolBUqtSsfAvixlzUQxXRrzDAeoF5jWqVcfopijOwIh1tEfehBxC4swjM4yZGM773CfODH8X_NOhBGXLQcwb9QQqV10dMFaiNH7girqrnQk.ntbvJw9RfGBJt8atbB-sAQ/__results___files/__results___35_0.png)

**contrast**
![contrast](https://www.kaggleusercontent.com/kf/31989134/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..kwjHRExsm3xFPskDr9KEXA.rU8dD6grh1GgWuAcAhpZDVlfZnqbZ63VM5dTYKo85TXf5g6YB6OT7f1D_ourPAO5fef2G8lBZHz4LI8An8qbfdbiD6uJc-Jj2rW0NLagRew3W4sI9ZaZMbP1SAdb_yei6RUa2xvf_OMoZb9ypET5SMikER9mEm--nfX74M7ULiQSZ_FdyjEAkwRX-r1CabltZYyKdCNGMQzdOaEqP9PGdWJqw2mfEqPZkQKib4EfJHCccotNVsqk9GARAekPmJ73FKo4Z4SrX55j9UeWd02mXjog4ONW_Q_rG1imjPtV5Tl7GQWcQl5PkE-i9EzAL7-9Uo-YT1LCLsnankmWbg1VQgfWMkwsoWAuh2fa3oZprm8XUjZwln2Dts3Na5i9cYNEH6WFpiC8b2kaRjh5WPbo6dIg9PvVSAk_gOKAJMMA515IG0MoDQQ92RKPpQmCQap9eonDTjWR8Jx4wc28OJyEIUCYtMeNdiIAxXvMo8EXHYEAHDkaaauF3V-FCRpnki-cI6BlPE57xM5qv-8YydtOLeEjitqBgiyECWHp4OGQqvl2kn_7_Q22ldA8gUEA1_gYDolBUqtSsfAvixlzUQxXRrzDAeoF5jWqVcfopijOwIh1tEfehBxC4swjM4yZGM773CfODH8X_NOhBGXLQcwb9QQqV10dMFaiNH7girqrnQk.ntbvJw9RfGBJt8atbB-sAQ/__results___files/__results___37_0.png)

**equalize**
![equalize](https://www.kaggleusercontent.com/kf/31989134/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..kwjHRExsm3xFPskDr9KEXA.rU8dD6grh1GgWuAcAhpZDVlfZnqbZ63VM5dTYKo85TXf5g6YB6OT7f1D_ourPAO5fef2G8lBZHz4LI8An8qbfdbiD6uJc-Jj2rW0NLagRew3W4sI9ZaZMbP1SAdb_yei6RUa2xvf_OMoZb9ypET5SMikER9mEm--nfX74M7ULiQSZ_FdyjEAkwRX-r1CabltZYyKdCNGMQzdOaEqP9PGdWJqw2mfEqPZkQKib4EfJHCccotNVsqk9GARAekPmJ73FKo4Z4SrX55j9UeWd02mXjog4ONW_Q_rG1imjPtV5Tl7GQWcQl5PkE-i9EzAL7-9Uo-YT1LCLsnankmWbg1VQgfWMkwsoWAuh2fa3oZprm8XUjZwln2Dts3Na5i9cYNEH6WFpiC8b2kaRjh5WPbo6dIg9PvVSAk_gOKAJMMA515IG0MoDQQ92RKPpQmCQap9eonDTjWR8Jx4wc28OJyEIUCYtMeNdiIAxXvMo8EXHYEAHDkaaauF3V-FCRpnki-cI6BlPE57xM5qv-8YydtOLeEjitqBgiyECWHp4OGQqvl2kn_7_Q22ldA8gUEA1_gYDolBUqtSsfAvixlzUQxXRrzDAeoF5jWqVcfopijOwIh1tEfehBxC4swjM4yZGM773CfODH8X_NOhBGXLQcwb9QQqV10dMFaiNH7girqrnQk.ntbvJw9RfGBJt8atbB-sAQ/__results___files/__results___39_0.png)

**brightness**
![brightness](https://www.kaggleusercontent.com/kf/31989134/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..kwjHRExsm3xFPskDr9KEXA.rU8dD6grh1GgWuAcAhpZDVlfZnqbZ63VM5dTYKo85TXf5g6YB6OT7f1D_ourPAO5fef2G8lBZHz4LI8An8qbfdbiD6uJc-Jj2rW0NLagRew3W4sI9ZaZMbP1SAdb_yei6RUa2xvf_OMoZb9ypET5SMikER9mEm--nfX74M7ULiQSZ_FdyjEAkwRX-r1CabltZYyKdCNGMQzdOaEqP9PGdWJqw2mfEqPZkQKib4EfJHCccotNVsqk9GARAekPmJ73FKo4Z4SrX55j9UeWd02mXjog4ONW_Q_rG1imjPtV5Tl7GQWcQl5PkE-i9EzAL7-9Uo-YT1LCLsnankmWbg1VQgfWMkwsoWAuh2fa3oZprm8XUjZwln2Dts3Na5i9cYNEH6WFpiC8b2kaRjh5WPbo6dIg9PvVSAk_gOKAJMMA515IG0MoDQQ92RKPpQmCQap9eonDTjWR8Jx4wc28OJyEIUCYtMeNdiIAxXvMo8EXHYEAHDkaaauF3V-FCRpnki-cI6BlPE57xM5qv-8YydtOLeEjitqBgiyECWHp4OGQqvl2kn_7_Q22ldA8gUEA1_gYDolBUqtSsfAvixlzUQxXRrzDAeoF5jWqVcfopijOwIh1tEfehBxC4swjM4yZGM773CfODH8X_NOhBGXLQcwb9QQqV10dMFaiNH7girqrnQk.ntbvJw9RfGBJt8atbB-sAQ/__results___files/__results___41_0.png)

**color**
![color](https://www.kaggleusercontent.com/kf/31989134/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..kwjHRExsm3xFPskDr9KEXA.rU8dD6grh1GgWuAcAhpZDVlfZnqbZ63VM5dTYKo85TXf5g6YB6OT7f1D_ourPAO5fef2G8lBZHz4LI8An8qbfdbiD6uJc-Jj2rW0NLagRew3W4sI9ZaZMbP1SAdb_yei6RUa2xvf_OMoZb9ypET5SMikER9mEm--nfX74M7ULiQSZ_FdyjEAkwRX-r1CabltZYyKdCNGMQzdOaEqP9PGdWJqw2mfEqPZkQKib4EfJHCccotNVsqk9GARAekPmJ73FKo4Z4SrX55j9UeWd02mXjog4ONW_Q_rG1imjPtV5Tl7GQWcQl5PkE-i9EzAL7-9Uo-YT1LCLsnankmWbg1VQgfWMkwsoWAuh2fa3oZprm8XUjZwln2Dts3Na5i9cYNEH6WFpiC8b2kaRjh5WPbo6dIg9PvVSAk_gOKAJMMA515IG0MoDQQ92RKPpQmCQap9eonDTjWR8Jx4wc28OJyEIUCYtMeNdiIAxXvMo8EXHYEAHDkaaauF3V-FCRpnki-cI6BlPE57xM5qv-8YydtOLeEjitqBgiyECWHp4OGQqvl2kn_7_Q22ldA8gUEA1_gYDolBUqtSsfAvixlzUQxXRrzDAeoF5jWqVcfopijOwIh1tEfehBxC4swjM4yZGM773CfODH8X_NOhBGXLQcwb9QQqV10dMFaiNH7girqrnQk.ntbvJw9RfGBJt8atbB-sAQ/__results___files/__results___43_0.png)

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