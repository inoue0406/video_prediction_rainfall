# Convert JMA radar dataset with .h5 format into .tfrecords format
# 
# part of the code below is taken from the following URL
# https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/18_TFRecords_Dataset_API.ipynb

import os

import numpy as np
import tensorflow as tf
import h5py
import pandas as pd

img_size = 128

# Helper-function for wrapping an integer so it can be saved to the TFRecords file.
def wrap_int64(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# Helper-function for wrapping raw bytes so they can be saved to the TFRecords file.
def wrap_bytes(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def f_scaling(X):
    # convert to [0-255] scale
    # sqrt conversion
    Xscl = np.power(X/201.0,0.5)*255
    return Xscl.astype(np.uint8)

def convert_tfrecords(image_dir, sample_csv, out_path):
    """
    Convert h5 file into tfrecords
    image_dir : image directory
    sample_csv : csv file with data file path
    """
    print("Converting: " + out_path)

    df_fnames = pd.read_csv(sample_csv)
    #print(list)

    # Open a TFRecordWriter for the output-file.
    with tf.python_io.TFRecordWriter(out_path) as writer:
    
        for i in range(len(df_fnames)):
            # read Past data
            h5_name_X = os.path.join(image_dir, df_fnames.ix[i, 'fname'])
            print('reading:',i,h5_name_X)
            h5file = h5py.File(h5_name_X,'r')
            X = h5file['R'][()]
            X = np.maximum(X,0) # replace negative value with 0
            X = f_scaling(X)   # scale range to [0-255]
            X = X[:,:,:,None] # add "channel" dimension as 1 (channel-last format)
            h5file.close()
            # read Future data
            h5_name_Y = os.path.join(image_dir, df_fnames.ix[i, 'fnext'])
            print('reading:',i,h5_name_Y)
            h5file = h5py.File(h5_name_Y,'r')
            Y = h5file['R'][()]
            Y = np.maximum(Y,0) # replace negative value with 0
            Y = f_scaling(Y)    # scale range to [0-255]
            Y = Y[:,:,:,None]   # add "channel" dimension as 1 (channel-last format)
            # save
            XY = np.concatenate([X,Y],axis=0)

            # Create a dict with the data we want to save in the TFRecords file.
            data = dict()
            tdim = XY.shape[0]
            for i in range(tdim):
                img = XY[i,:,:,:]
                
                # Convert the image to raw bytes.
                img_bytes = img.tostring()
                step_str = "step" + str(i)
                data[step_str] = wrap_bytes(img_bytes)
                #data[step_str] = img_bytes

            # Wrap the data as TensorFlow Features.
            feature = tf.train.Features(feature=data)
            
            # Wrap again as a TensorFlow Example.
            example = tf.train.Example(features=feature)
            
            # Serialize the data.
            serialized = example.SerializeToString()
            
            # Write the serialized data to the TFRecords file.
            writer.write(serialized)

if __name__ == '__main__':

    # for training data
    image_dir = "../data/jma/data_kanto_resize"
    train_sample_csv = "../data/jma/train_simple_JMARadar.csv"
    path_tfrecords_train = "../data/jma/train/train_simple.tfrecords"
    convert_tfrecords(image_dir=image_dir,
                      sample_csv=train_sample_csv,
                      out_path=path_tfrecords_train)
    
    # for validation data
    image_dir = "../data/jma/data_kanto_resize"
    valid_sample_csv = "../data/jma/valid_simple_JMARadar.csv"
    path_tfrecords_valid = "../data/jma/val/valid_simple.tfrecords"
    convert_tfrecords(image_dir=image_dir,
                      sample_csv=valid_sample_csv,
                      out_path=path_tfrecords_valid)
    
