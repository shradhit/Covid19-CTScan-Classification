import numpy as np
import pandas as pd
import os
import random 
from shutil import copyfile
import pydicom as dicom
import cv2


# path to covid-19 dataset from Actualmed_COVID-chestxray-dataset
def actualmed_processing(actualmed_csvpath, actualmed_imgpath):
  
  
    train = list()
    test = list()
    mapping = dict()
    mapping['COVID-19'] = 'COVID-19'
    mapping['No Finding'] = 'normal'
    mapping['No finding'] = 'normal'
    # train/test split
    split = 0.1
    
    actualmed_covid = pd.read_csv(actualmed_csvpath)

    filename_label_new = {'normal': [], 'pneumonia': [], 'COVID-19': []}
    count_new = {'normal': 0, 'pneumonia': 0, 'COVID-19': 0}
    
    # mapping 
    for index, row in actualmed_covid.iterrows():
        f = str(row['finding']).split(',')[0] 
        if f in mapping:
            count_new[mapping[f]] += 1
            entry = [str(row['patientid']), row['imagename'], mapping[f], row['view']]
            filename_label_new[mapping[f]].append(entry)

    
    for key in filename_label_new.keys():
        arr = np.array(filename_label_new[key])
        if arr.size == 0:
            continue
        num_diff_patients = len(np.unique(arr[:,0]))
        num_test = max(1, round(split*num_diff_patients))
        #select num_test number of random patients
        if key == 'normal':
             test_patients = random.sample(list(np.unique(arr[:,0])), num_test)
        elif key == 'COVID-19':
             test_patients = random.sample(list(np.unique(arr[:,0])), num_test)
        else: 
             test_patients = []
        print('Key: ', key)
        print('Test patients: ', test_patients)
        
        for patient in arr:
            if patient[0] not in patient_imgpath:
                patient_imgpath[patient[0]] = [patient[1]]
            else:
                if patient[1] not in patient_imgpath[patient[0]]:
                    patient_imgpath[patient[0]].append(patient[1])
                else:
                    continue  # skip since image has already been written
            if patient[0] in test_patients:
                copyfile(os.path.join(actualmed_imgpath, patient[1]), os.path.join(savepath, 'test', patient[1]))
                test.append(patient)
                test_count[patient[2]] += 1
            else:
                if 'COVID' in patient[0]:
                    copyfile(os.path.join(actualmed_imgpath, patient[1]), os.path.join(savepath, 'train', patient[1]))
                else:
                    copyfile(os.path.join(actualmed_imgpath, patient[1]), os.path.join(savepath, 'train', patient[1]))
                train.append(patient)
                train_count[patient[2]] += 1
            
    print('test count: ', test_count)
    print('train count: ', train_count)
    print("DONE - with actualmed_processing")
    
    return train, test 
  
  
  
actualmed_imgpath = '/Users/shradhitsubudhi/Documents/COVID/mywork/all_data/Actualmed-COVID-chestxray-dataset/images' 
actualmed_csvpath = '/Users/shradhitsubudhi/Documents/COVID/mywork/all_data/Actualmed-COVID-chestxray-dataset/metadata.csv'
train, test = actualmed_processing(actualmed_csvpath, actualmed_imgpath)
