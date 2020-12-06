
If you want to reproduce our results, please download the dataset from [http://doi.org/10.5281/zenodo.4298978](http://doi.org/10.5281/zenodo.4298978) and unzip them (`WRF_50km_vars-mask-clip0p05-99p5.hdf5` and `WRF_precip_2005_12km-mask-clip0p05-99p5.hdf5`) here.

If you want to try it using your own dataset, you can either edit the data.py to load dataset of your format, or prepare your h5 dataset using the same dataset name as we used in the data.py. 
We need to note that, the main purpose of this repo is to show all implementation details of our model proposed in the paper. 
This implementation is not fine-tuned to support new dataset straightforwardly, i.e., it is not an out-of-box solution for any downscaling.
Depends on your dataset, you may need to handcraft our harded coded piece to make it work with yours.

Basically, you need to prepare your high resolution data as (omit the dimension size):
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

and you low resolution precipitation and variables as (omit the dimension size):
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
