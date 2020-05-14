import numpy as np
import pandas as pd
import os
import random 
from shutil import copyfile
import pydicom as dicom
import cv2


# process Actualmed_COVID-chestxray-dataset
def actualmed_processing(actualmed_csvpath, actualmed_imgpath):
    
    train = list()
    test = list()
    mapping = dict()
    mapping['COVID-19'] = 'COVID-19'
    mapping['No Finding'] = 'normal'
    mapping['No finding'] = 'normal'
    # train/test split
    split = 0.1
    #reading the csv
    actualmed_covid = pd.read_csv(actualmed_csvpath)
 
    filename_label_new = {'normal': [], 'pneumonia': [], 'COVID-19': []}
    count_new = {'normal': 0, 'pneumonia': 0, 'COVID-19': 0}
    
    # example = {"normal" : ["Patient ID", "imagename", "COVID", "view"]}
    for index, row in actualmed_covid.iterrows():
        f = str(row['finding']).split(',')[0] 
        if f in mapping:
            count_new[mapping[f]] += 1
            entry = [str(row['patientid']), row['imagename'], mapping[f], row['view']]
            filename_label_new[mapping[f]].append(entry)

    # copy the files (train, test) folders and keep a track of the images
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
    
    print(count_new)
    print('test count: ', test_count)
    print('train count: ', train_count)
    print("DONE - with actualmed_processing")
    
    return train, test 
  
  
  
  
  
def ieee_agchung(cohen_csv, cohen_csvpath, fig1_imgpath, fig1_csvpath):
    
    mapping = dict()
    mapping['COVID-19'] = 'COVID-19'
    mapping['SARS'] = 'pneumonia'
    mapping['MERS'] = 'pneumonia'
    mapping['Streptococcus'] = 'pneumonia'
    mapping['Klebsiella'] = 'pneumonia'
    mapping['Chlamydophila'] = 'pneumonia'
    mapping['Legionella'] = 'pneumonia'
    mapping['Normal'] = 'normal'
    mapping['Lung Opacity'] = 'pneumonia'
    mapping['No Finding'] = 'normal'
    mapping['No finding'] = 'normal'

    mapping['1'] = 'pneumonia'

    # train/test split
    split = 0.1
    # to avoid duplicates
    patient_imgpath = {}

    # adapted from https://github.com/mlmed/torchxrayvision/blob/master/torchxrayvision/datasets.py#L814
    cohen_csv = pd.read_csv(cohen_csvpath, nrows=None)
    #idx_pa = csv["view"] == "PA"  # Keep only the PA view
    views = ["PA", "AP", "AP Supine", "AP semi erect", "AP erect"]
    cohen_idx_keep = cohen_csv.view.isin(views)
    cohen_csv = cohen_csv[cohen_idx_keep]

    fig1_csv = pd.read_csv(fig1_csvpath, encoding='ISO-8859-1', nrows=None)
    fig1_idx_keep = fig1_csv.view.isin(views)
    fig1_csv = fig1_csv[fig1_idx_keep]
    
    
    # get non-COVID19 viral, bacteria, and COVID-19 infections from covid-chestxray-dataset
    # stored as patient id, image filename and label
    filename_label = {'normal': [], 'pneumonia': [], 'COVID-19': []}
    count = {'normal': 0, 'pneumonia': 0, 'COVID-19': 0}
    for index, row in cohen_csv.iterrows():
        f = row['finding'].split(',')[0] # take the first finding, for the case of COVID-19, ARDS
        if f in mapping: # 
            count[mapping[f]] += 1
            entry = [str(row['patientid']), row['filename'], mapping[f], row['view']]
            filename_label[mapping[f]].append(entry)

    print('Data distribution from covid-chestxray-dataset:')
    print(count)


    # path to covid-19 dataset from https://github.com/agchung/Figure1-COVID-chestxray-dataset
    for index, row in fig1_csv.iterrows():
        if not str(row['finding']) == 'nan':
            f = row['finding'].split(',')[0] # take the first finding
            if f in mapping: # 
                count[mapping[f]] += 1
                if os.path.exists(os.path.join(fig1_imgpath, row['patientid'] + '.jpg')):
                    entry = [row['patientid'], row['patientid'] + '.jpg', mapping[f]]
                elif os.path.exists(os.path.join(fig1_imgpath, row['patientid'] + '.png')):
                    entry = [row['patientid'], row['patientid'] + '.png', mapping[f]]
                filename_label[mapping[f]].append(entry)

    print('Data distribution from covid-chestxray-dataset:')
    print(count)

    # add covid-chestxray-dataset into COVIDx dataset
    # since covid-chestxray-dataset doesn't have test dataset
    # split into train/test by patientid
    # for COVIDx:
    # patient 8 is used as non-COVID19 viral test
    # patient 31 is used as bacterial test
    # patients 19, 20, 36, 42, 86 are used as COVID-19 viral test

    for key in filename_label.keys():
        arr = np.array(filename_label[key])
        if arr.size == 0:
            continue
        # split by patients
        num_diff_patients = len(np.unique(arr[:,0]))
        num_test = max(1, round(split*num_diff_patients))
        # select num_test number of random patients
        if key == 'pneumonia':
            #test_patients = ['8', '31']
            test_patients = random.sample(list(arr[:,0]), num_test)
        elif key == 'COVID-19':
            #test_patients = ['19', '20', '36', '42', '86', '94', '97', '117', '132', '138', '144', '150', '163', '169']  
            test_patients = random.sample(list(arr[:,0]), num_test)
        else: 
            test_patients = []
        print('Key: ', key)
        print('Test patients: ', test_patients)
        # go through all the patients
        for patient in arr:
            if patient[0] not in patient_imgpath:
                patient_imgpath[patient[0]] = [patient[1]]
            else:
                if patient[1] not in patient_imgpath[patient[0]]:
                    patient_imgpath[patient[0]].append(patient[1])
                else:
                    continue  # skip since image has already been written
            if patient[0] in test_patients:
                copyfile(os.path.join(cohen_imgpath, patient[1]), os.path.join(savepath, 'test', patient[1]))
                test.append(patient)
                test_count[patient[2]] += 1
            else:
                if 'COVID' in patient[0]:
                    copyfile(os.path.join(fig1_imgpath, patient[1]), os.path.join(savepath, 'train', patient[1]))
                else:
                    copyfile(os.path.join(cohen_imgpath, patient[1]), os.path.join(savepath, 'train', patient[1]))
                train.append(patient)
                train_count[patient[2]] += 1

    print('test count: ', test_count)
    print('train count: ', train_count)
    
    return train, test 
  
# path to covid-19 dataset from actualmed_imgpath
actualmed_imgpath = '/Users/shradhitsubudhi/Documents/COVID/mywork/all_data/Actualmed-COVID-chestxray-dataset/images' 
actualmed_csvpath = '/Users/shradhitsubudhi/Documents/COVID/mywork/all_data/Actualmed-COVID-chestxray-dataset/metadata.csv'
train_actualmed, test_actualmed = actualmed_processing(actualmed_csvpath, actualmed_imgpath)

# path to covid-19 dataset from https://github.com/ieee8023/covid-chestxray-dataset
cohen_imgpath = '/Users/shradhitsubudhi/Documents/COVID/mywork/all_data/covid-chestxray-dataset/images' 
cohen_csvpath = '/Users/shradhitsubudhi/Documents/COVID/mywork/all_data/covid-chestxray-dataset/metadata.csv'

# path to covid-14 dataset from https://github.com/agchung/Figure1-COVID-chestxray-dataset
fig1_imgpath = '/Users/shradhitsubudhi/Documents/COVID/mywork/all_data/Figure1-COVID-chestxray-dataset/images'
fig1_csvpath = '/Users/shradhitsubudhi/Documents/COVID/mywork/all_data/Figure1-COVID-chestxray-dataset/metadata.csv'

train_ieee_agchung, test_ieee_agchung = ieee_agchung(cohen_csv, cohen_csvpath, fig1_imgpath, fig1_csvpath)

