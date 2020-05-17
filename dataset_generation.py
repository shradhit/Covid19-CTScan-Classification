import numpy as np
import pandas as pd
import os
import random 
from shutil import copyfile
import pydicom as dicom
import cv2


# kaggle dataset
def kaggle_rsna(rsna_datapath, rsna_csvname, rsna_csvname2, rsna_imgpath):
    train = list()
    test = list()
    savepath = "/Users/shradhitsubudhi/Documents/COVID/mywork/modified_data/"
    split = 0.1
    patient_imgpath = {}

    test_count = {'normal': 0, 'pneumonia': 0, 'COVID-19': 0}
    train_count = {'normal': 0, 'pneumonia': 0, 'COVID-19': 0}

    # add normal and rest of pneumonia cases from https://www.kaggle.com/c/rsna-pneumonia-detection-challenge
    csv_normal = pd.read_csv(os.path.join(rsna_datapath, rsna_csvname), nrows=None)
    csv_pneu = pd.read_csv(os.path.join(rsna_datapath, rsna_csvname2), nrows=None)
    patients = {'normal': [], 'pneumonia': []}

    for index, row in csv_normal.iterrows():
        if row['class'] == 'Normal':
            patients['normal'].append(row['patientId'])

    for index, row in csv_pneu.iterrows():
        if int(row['Target']) == 1:
            patients['pneumonia'].append(row['patientId'])

    for key in patients.keys():
        arr = np.array(patients[key])
        if arr.size == 0:
            continue
        # split by patients 
        num_diff_patients = len(np.unique(arr))
        num_test = max(1, round(split*num_diff_patients))
        #test_patients = np.load('/Users/shradhitsubudhi/Documents/COVID/mywork/all_data/rsna_test_patients_{}.npy'.format(key)) 
        test_patients = random.sample(list(arr), num_test)
        #, download the .npy files from the repo.
        #np.save('rsna_test_patients_{}.npy'.format(key), np.array(test_patients))
        for patient in arr:
            if patient not in patient_imgpath:
                patient_imgpath[patient] = [patient]
            else:
                continue  # skip since image has already been written
            ds = dicom.dcmread(os.path.join(rsna_datapath, rsna_imgpath, patient + '.dcm'))
            pixel_array_numpy = ds.pixel_array
            imgname = patient + '.png'
            if patient in test_patients:
                cv2.imwrite(os.path.join(savepath, 'test', imgname), pixel_array_numpy)
                test.append([patient, imgname, key])
                test_count[key] += 1
            else:
                cv2.imwrite(os.path.join(savepath, 'train', imgname), pixel_array_numpy)
                train.append([patient, imgname, key])
                train_count[key] += 1
    
    print('train count: ', train_count)
    print('test count: ', test_count)
    
    return train, test 




# process Actualmed_COVID-chestxray-dataset
def actualmed_processing(actualmed_csvpath, actualmed_imgpath):
    
    savepath = "/Users/shradhitsubudhi/Documents/COVID/mywork/modified_data/"
    split = 0.1
    test_count = {'normal': 0, 'pneumonia': 0, 'COVID-19': 0}
    train_count = {'normal': 0, 'pneumonia': 0, 'COVID-19': 0}

    patient_imgpath = {}
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
             test_patients = random.sample(list(np.unique(list(zip(*arr))[0])), num_test)
        elif key == 'COVID-19':
             test_patients = random.sample(list(np.unique(list(zip(*arr))[0])), num_test)
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
  
  

  
def ieee_agchung(cohen_imgpath, cohen_csvpath, fig1_imgpath, fig1_csvpath):
    
    savepath = "/Users/shradhitsubudhi/Documents/COVID/mywork/modified_data/"
    split = 0.1
    train = list()
    test = list()
    test_count = {'normal': 0, 'pneumonia': 0, 'COVID-19': 0}
    train_count = {'normal': 0, 'pneumonia': 0, 'COVID-19': 0}
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
        num_diff_patients = len(np.unique(list(zip(*arr))[0]))
        num_test = max(1, round(split*num_diff_patients))
        # select num_test number of random patients
        if key == 'pneumonia':
            test_patients = ['8', '31']
            #test_patients = random.sample(list(zip(*arr))[0], num_test)
        elif key == 'COVID-19':
            test_patients = ['19', '20', '36', '42', '86', '94', '97', '117', '132', '138', '144', '150', '163', '169']  
            #test_patients = random.sample(list(zip(*arr))[0], num_test)
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



def radiography(covid_radiography):
    split = 0.1
    savepath = "/Users/shradhitsubudhi/Documents/COVID/mywork/modified_data/"

    covid_img = "COVID-19"
    covid_metadata = "COVID-19.metadata.xlsx"

    normal_img = "NORMAL"
    normal_metadata = "NORMAL.metadata.xlsx"

    viral_img = "Viral Pneumonia"
    viral_metadata = "Viral Pneumonia.matadata.xlsx"


    list_covid  = os.listdir(os.path.join(covid_radiography, covid_img))
    list_normal =  os.listdir(os.path.join(covid_radiography, normal_img))
    list_viral  =  os.listdir(os.path.join(covid_radiography, viral_img))


    covid_total = len(list_covid)
    num_covid_test = max(1, round(split*covid_total))

    normal_total = len(list_normal)
    num_normal_test = max(1, round(split*normal_total))

    viral_total = len(list_viral)
    num_viral_test = max(1, round(split*viral_total))

    test_covid = random.sample(list_covid, num_covid_test)
    test_normal = random.sample(list_normal, num_normal_test)
    test_viral = random.sample(list_viral, num_viral_test)


    for x in (list_covid):
        if x in test_covid:
            copyfile(os.path.join(covid_radiography, covid_img, x ), os.path.join(savepath, 'test', x))
        else: 
            copyfile(os.path.join(covid_radiography, covid_img, x), os.path.join(savepath, 'train', x))

    for x in (list_viral):
        if x in test_viral:
            copyfile(os.path.join(covid_radiography, viral_img, x ), os.path.join(savepath, 'test', x))
        else: 
            copyfile(os.path.join(covid_radiography, viral_img, x), os.path.join(savepath, 'train', x))

    for x in (list_normal):
        if x in test_normal:
            copyfile(os.path.join(covid_radiography, normal_img, x ), os.path.join(savepath, 'test', x))
        else: 
            copyfile(os.path.join(covid_radiography, normal_img, x), os.path.join(savepath, 'train', x))
            
    train = list(np.setdiff1d(list_covid,test_covid)) + list(np.setdiff1d(list_normal,test_normal)) +  list(np.setdiff1d(list_viral,test_viral))
    test = test_covid + test_normal + test_viral
    #print()
    
    
    return  train, test

def convert_todf(array):
    coversion = []
    for x in array:
        if isinstance(x, (np.ndarray, np.generic)) == True:
            coversion.append(x.tolist())
        else: 
            coversion.append(x)
    df = pd.DataFrame(coversion)  
    return df



  
def merge(): 
    #def merge():

    # path to covid-19 dataset from actualmed_imgpath
    actualmed_imgpath = '/Users/shradhitsubudhi/Documents/COVID/mywork/all_data/Actualmed-COVID-chestxray-dataset/images' 
    actualmed_csvpath = '/Users/shradhitsubudhi/Documents/COVID/mywork/all_data/Actualmed-COVID-chestxray-dataset/metadata.csv'
    train_actualmed, test_actualmed = actualmed_processing(actualmed_csvpath, actualmed_imgpath)

    # path to covid-19 dataset from https://github.com/ieee8023/covid-chestxray-dataset
    cohen_imgpath = '/Users/shradhitsubudhi/Documents/COVID/mywork/all_data/covid-chestxray-dataset/images' 
    cohen_csvpath = '/Users/shradhitsubudhi/Documents/COVID/mywork/all_data/covid-chestxray-dataset/metadata.csv'

    # path to covid-19 dataset from https://github.com/agchung/Figure1-COVID-chestxray-dataset
    fig1_imgpath = '/Users/shradhitsubudhi/Documents/COVID/mywork/all_data/Figure1-COVID-chestxray-dataset/images'
    fig1_csvpath = '/Users/shradhitsubudhi/Documents/COVID/mywork/all_data/Figure1-COVID-chestxray-dataset/metadata.csv'
    # combined agchung and ieee8023
    train_ieee_agchung, test_ieee_agchung = ieee_agchung(cohen_imgpath, cohen_csvpath, fig1_imgpath, fig1_csvpath)

    # path to https://www.kaggle.com/c/rsna-pneumonia-detection-challenge
    rsna_datapath = '/Users/shradhitsubudhi/Documents/COVID/mywork/all_data/rsna-pneumonia-detection-challenge'
    # get all the normal from here
    rsna_csvname = 'stage_2_detailed_class_info.csv' 

    # get all the 1s from here since 1 indicate pneumonia
    # found that images that aren't pneunmonia and also not normal are classified as 0s
    rsna_csvname2 = 'stage_2_train_labels.csv' 
    rsna_imgpath = 'stage_2_train_images'

    train_rsna, test_rsna = kaggle_rsna(rsna_datapath, rsna_csvname, rsna_csvname2, rsna_imgpath)

    # radiography
    covid_radiography = "/Users/shradhitsubudhi/Documents/COVID/mywork/all_data/COVID-19RadiographyDatabase/"
    train, test = radiography(covid_radiography)

    #return 'DONE ALL'


    ######

    #print(len(train_ieee_agchung))
    df_train_ieee_agchug = convert_todf(train_ieee_agchung)
    df_train_ieee_agchug['train/test'] = 'train'
    

    #print(len(test_ieee_agchung))
    df_test_ieee_agchug = convert_todf(test_ieee_agchung)
    df_test_ieee_agchug['train/test'] = 'test'

    #print(len(train_actualmed))
    df_train_actualmed = convert_todf(train_actualmed)
    df_train_actualmed['train/test'] = 'train'

    #print(len(test_actualmed))
    df_test_actualmed = convert_todf(test_actualmed)
    df_test_actualmed['train/test'] = 'test'

    #print(len(train_rsna))
    df_train_rsna = convert_todf(train_rsna)
    df_train_rsna['train/test'] = 'train'

    #print(len(test_rsna))
    df_test_rsna = convert_todf(test_rsna)
    df_test_rsna['train/test'] = 'test'

    #print(len(train))
    df_train_radiography = convert_todf(train)
    df_train_radiography['train/test'] = 'train'

    #print(len(test))
    df_test_radiography = convert_todf(test)
    df_test_radiography['train/test'] = 'test'



    ######## 
    df_ieee = df_train_ieee_agchug.append(df_test_ieee_agchug)
    df_ieee.columns = ['patientid', 'imagename', 'target', 'view', 'train_test']
    df_ieee["dataset"] = "IEEE"



    df_actualmed = df_train_actualmed.append(df_test_actualmed)
    df_actualmed.columns = ['patientid', 'imagename', 'target', 'view', 'train_test']
    df_actualmed["dataset"] = "actualmed"


    df_rsna = df_train_rsna.append(df_test_rsna)
    df_rsna.columns = ['patientid', 'imagename', 'target', 'train_test']
    df_rsna["dataset"] = "rsna"


    df_radiography = df_train_radiography.append(df_test_radiography)
    df_radiography.columns = ['patientid', 'train_test']
    df_radiography['imagename'] = df_radiography['patientid']
    df_radiography["dataset"] = "radiography"


    df_all = df_ieee.append([df_actualmed, df_rsna, df_radiography])
    df_all.to_csv('contains_all_data.csv')
    
    return "DONE ALL"


if __name__ = "__main__":
    merge()
