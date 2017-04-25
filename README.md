# agn-ae
An active galactic nucleus (AGN) is a compact region at the galaxy center, which radiates highly energetic jets, blows and heats the gas in the interstellar medium around it, and generates bubbles or cavities. 

To study on the AGN, the jets or cavities are perfect probes. In our work before, a convolutional neural network (CNN) based cavity detection approach namely [cavdet])(https://github.com/myinxd/cavdet)) was proposed, which can automatically detect and locate the cavities in the X-ray astronomical images. However, as for the lower radio frequency band, the cavitis are unobservable, while the jets are bright. Thus, the study on AGN jets is of interest and significance. 

The radio sources with jets are mainly classified into FRI and FRII according to their profiles, inclines and so on. And the biggest difference between FRI and FRII is that there are bright lobes at the jets' forehead of the FRII, while not for the FRI.
In addition, it requires futherly classification in both or them, since the detail profiles of the samples are different, and the intrinsic physical mechanism may also be different. However, it remains uncertainly how to do the sub-classication. 

To handle this, we are going to apply the unsupervised feature learning methods to seek the intrinsic representations of the samples at the image domai. In this repository, the autoencoder based feature learning and classification algorithms on FRI, FRII active galactic nuclei (AGN) are studied.

## Requirements
To process our scripts, some python packages are required, which are listed as follows.

- numpy, scipy, pickle
- scikit-image
- [astropy](http://docs.astropy.org/en/stable/), [astroquery](http://astroquery.readthedocs.io/en/latest/)
- [Theano](http://www.deeplearning.net/software/theano/), 
- [Lasagne](http://lasagne.readthedocs.io/en/latest/), 
- [nolearn](http://pythonhosted.org/nolearn/lasagne.html)

The `requirements.txt` is provided in this repository, by which the required packages can be installed easily. We advice the users to configure these packages in a virtual environment.

- initialize env
```sh
   $ <sudo> pip install virtualenv
   $ cd agn-ae
   $ virtualenv ./env
```
- install required packages
```sh
   $ cd agn-ae
   $ env/bin/pip install -r ./requirements.txt
```

In addition, the computation can be accelerated by parallelly processing with GPUs. In this work, our scripts are written under the guide of Nvidia CUDA, thus the Nvidia GPU hardware is also required.

- CUDA Toolkit
  https://developer.nvidia.com/cuda-downloads

## References
- Theano tutorial 
  http://www.deeplearning.net/software/theano/
- Lasagne tutorial 
  http://lasagne.readthedocs.io/en/latest/user/tutorial.html
- Save python data by pickle
  http://www.cnblogs.com/pzxbc/archive/2012/03/18/2404715.html
- convolutional_autoencoder
  https://github.com/mikesj-public/convolutional_autoencoder


## Author
- Zhixian MA <`zxma_sjtu(at)qq.com`>

## Contributor
- Chenxi SHAN

## License
Unless otherwise declared:

- Codes developed are distributed under the [MIT license](https://opensource.org/licenses/mit-license.php);
- Documentations and products generated are distributed under the [Creative Commons Attribution 3.0 license](https://creativecommons.org/licenses/by/3.0/us/deed.en_US);
- Third-party codes and products used are distributed under their own licenses.
