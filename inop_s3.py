
import numpy as np
import pandas as pd
import os
import random 
import pydicom as dicom
import cv2
import boto3
import tempfile
from io import StringIO
import matplotlib.pyplot as plt
import matplotlib.image as mpimg



## all codes specifically for s3 



def read_image_s3(file_name):
    bucket_name = "aws-a0077-glbl-00-p-s3b-shrd-awb-shrd-prod-78"
    s3_resource = boto3.resource('s3')
    bucket = s3_resource.Bucket(bucket_name)
    img = bucket.Object(file_name).get().get('Body').read()
    img_nparray = cv2.imdecode(np.asarray(bytearray(img)), cv2.IMREAD_COLOR)
    return img_nparray


def listdir(dir_):
    ''' 
    
    os.listdir: Alternate for os.dirfunction to be able to list down the files
    in the s3 file / index ( yes we know there is no concept of folder in s3 everything is a key)
    example function  call : # listdir("Radiography database/COVID-19/")

    '''
    
    s3 = boto3.resource('s3')
    my_bucket = s3.Bucket('aws-a0077-glbl-00-p-s3b-shrd-awb-shrd-prod-78')
    files = list()
    for object_summary in my_bucket.objects.filter(Prefix=dir_):
        files.append(object_summary.key.split('/')[-1])
    return files

# read excel on s3 
def read_excel(file_name):
    '''
    pd.read_excel() alternate for that
    '''
    s3 = boto3.client('s3') 
    bucket = "aws-a0077-glbl-00-p-s3b-shrd-awb-shrd-prod-78"
    obj = s3.get_object(Bucket= bucket, Key= file_name) 
    initial_df = pd.read_excel(obj['Body']) # 'Body' is a key word
    return initial_df

# read csv on s3 
def read_csv(file_name, encoding = None):
    '''
    pd.read_csv() alternate for that
    '''
    s3 = boto3.client('s3') 
    bucket = "aws-a0077-glbl-00-p-s3b-shrd-awb-shrd-prod-78"
    obj = s3.get_object(Bucket= bucket, Key= file_name)
    if encoding != None: 
        initial_df = pd.read_csv(obj['Body'], encoding= "ISO-8859-1") # 'Body' is a key word
        return initial_df
    else:
        initial_df = pd.read_csv(obj['Body']) # 'Body' is a key word
    return initial_df
        


def copyfile(from_, to_):
    '''
    copy files form one key to another key on s3
    
    '''
    
    s3 = boto3.resource('s3')
    copy_source = {
        'Bucket': 'aws-a0077-glbl-00-p-s3b-shrd-awb-shrd-prod-78'
    }
    copy_source['Key'] = from_
    s3.meta.client.copy(copy_source, 'aws-a0077-glbl-00-p-s3b-shrd-awb-shrd-prod-78', to_)
    return "copy done"


def dcmread_(file):
    '''
    reads dicom format data 
    example function c # dcmread_("rsna-pnemonia-detection-challenge/stage_2_test_images/000e3a7d-c0ca-4349-bb26-5af2d8993c3d.dcm")

    '''
    session = boto3.Session()
    s3 = session.client('s3')

    fileobj = s3.get_object(
            Bucket="aws-a0077-glbl-00-p-s3b-shrd-awb-shrd-prod-78",
            Key= file
            )

    # open the file object and read it into the variable dicom_data. 
    dicom_data = fileobj['Body'].read()

    # Read DICOM
    dicom_bytes = dicom.filebase.DicomBytesIO(dicom_data)
    dicom_dataset = dicom.dcmread(dicom_bytes)
    return dicom_dataset


def imwrite_(destination, array):
    '''
    writes it to file..  
    
   example call : # s3.put_object(Bucket="aws-a0077-glbl-00-p-s3b-shrd-awb-shrd-prod-78", Key = "check/imageName", 
                                                          Body =      local_image, ContentType= 'image/png')  
    #cv2.imwrite for s3 
    
    '''
    
    bucket_name = 'aws-a0077-glbl-00-p-s3b-shrd-awb-shrd-prod-78'
    s3 = boto3.resource("s3")
    try:
        image_string = cv2.imencode('.png', array)[1].tostring()
        s3.Bucket(bucket_name).put_object(Key=destination, Body=image_string)

    except:
        print('false')

 
             
def to_csv_on_s3(dataframe, filename):
    DESTINATION = 'aws-a0077-glbl-00-p-s3b-shrd-awb-shrd-prod-78'

    """ Write a dataframe to a CSV on S3 """
    print("Writing {} records to {}".format(len(dataframe), filename))
    # Create buffer
    csv_buffer = StringIO()
    # Write dataframe to buffer
    dataframe.to_csv(csv_buffer, sep=",", index=False)
    # Create S3 object
    s3_resource = boto3.resource("s3")
    # Write buffer to S3 object
    s3_resource.Object(DESTINATION, filename).put(Body=csv_buffer.getvalue())
    return 0



def pickle_save_s3(key):
    '''
    Pickle : save the file  
    '''
    s3_client = boto3.client('s3')
    my_array_data = io.BytesIO()
    pickle.dump(x_train_final, my_array_data, protocol=4)
    my_array_data.seek(0)
    s3_client.upload_fileobj(my_array_data, "aws-a0077-glbl-00-p-s3b-shrd-awb-shrd-prod-78", key)
    return 0


def pickle_read_s3(key):
    
    '''
    read large pickle files from s3 
    '''

    bucket = "aws-a0077-glbl-00-p-s3b-shrd-awb-shrd-prod-78"
    #key = "x_train_final.pkl"
    s3 = boto3.resource('s3')

    obj = s3.Object(bucket, key)
    with io.BytesIO(obj.get()["Body"].read()) as f:
        # rewind the file
        f.seek(0)
        x_train_final = np.load(f, allow_pickle=True, encoding='bytes')
        
    return x_train_final



###############################----------------------------------------