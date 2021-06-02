![version](https://img.shields.io/badge/Version-v1.0.0-blue.svg?style=plastic)
![PyTorch](https://img.shields.io/badge/PyTorch-v1.5.0-green.svg?style=plastic)
![license](https://img.shields.io/badge/license-CC_BY--NC-red.svg?style=plastic)
[![DOI](https://zenodo.org/badge/314389271.svg)](https://zenodo.org/badge/latestdoi/314389271)

# About
This is the offical implementation of the model, a conditinal GAN for dynamic precipitation downscaling, proposed in our paper [Fast and accurate learned multiresolution dynamical downscaling for precipitation](https://doi.org/10.5194/gmd-2020-412)

# Model and Implementation 
As discussed in the manuscript, we evluated 5 different model archtecture and concluded that the `Encoded-CGAN` performed the best in different cases under different metric. So, here we only opensourced the `Encoded-CGAN`. The architecture of the generator is shown below. We can consider open other implemnetaions upon request. 

![Generator Model Arch](repo-img/DSGAN-github-encoded.png)

We need to note that, the main purpose of this repo is to `show all implementation details` of our model proposed in the paper. 
This implementation is not fine-tuned to support new dataset straightforwardly, i.e., it is NOT an out-of-box solution for any downscaling.
Depends on your dataset, you may need to handcraft our hard-coded piece to make it work with yours.

# Dataset
The dataset used in this repo and in the manuscript can be downloaded [http://doi.org/10.5281/zenodo.4298978](https://doi.org/10.5281/zenodo.4298978).
More information about data preparation is available under the dataset folder.

If you want to reproduce our results, please download the dataset from [http://doi.org/10.5281/zenodo.4298978](http://doi.org/10.5281/zenodo.4298978) and unzip them (`WRF_50km_vars-mask-clip0p05-99p5.hdf5` and `WRF_precip_2005_12km-mask-clip0p05-99p5.hdf5`) here.

If you want to try it using your own dataset, you can either edit the data.py to load dataset of your format, or prepare your h5 dataset using the same dataset name as we used in the data.py. 
We need to note that, the main purpose of this repo is to show all implementation details of our model proposed in the paper. 
This implementation is not fine-tuned to support new dataset straightforwardly, i.e., it is not an out-of-box solution for any downscaling.
Depends on your dataset, you may need to handcraft our hard-coded piece to make it work with yours.

Basically, you need to prepare your high resolution precipitation as (omit the dimension size):
```
HDF5 "WRF_precip_2005_12km-mask-clip0p05-99p5.hdf5" {
GROUP "/" {
   DATASET "rain" {
      DATATYPE  H5T_IEEE_F32LE
      DATASPACE  SIMPLE { ( 2911, 256, 512 ) / ( 2911, 256, 512 ) }
   }
}
}
```

and your low resolution precipitation and variables as (omit the dimension size):
```
HDF5 "WRF_50km_vars-mask-clip0p05-99p5.hdf5" {
GROUP "/" {
   DATASET "IWV" {
      DATATYPE  H5T_IEEE_F32LE
      DATASPACE  SIMPLE { ( 2911, 64, 128 ) / ( 2911, 64, 128 ) }
   }
   DATASET "RAIN" {
      DATATYPE  H5T_IEEE_F32LE
      DATASPACE  SIMPLE { ( 2911, 64, 128 ) / ( 2911, 64, 128 ) }
   }
   DATASET "SLP" {
      DATATYPE  H5T_IEEE_F32LE
      DATASPACE  SIMPLE { ( 2911, 64, 128 ) / ( 2911, 64, 128 ) }
   }
   DATASET "T2" {
      DATATYPE  H5T_IEEE_F32LE
      DATASPACE  SIMPLE { ( 2911, 64, 128 ) / ( 2911, 64, 128 ) }
   }
}
}
```

