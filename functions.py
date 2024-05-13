# import shutil
# import os
import opensmile
from scipy.io import wavfile
import pandas as pd
import autosklearn.classification
import pickle
import numpy as np


def readLinesFromFile(filenameTXT):

    lines = []

    with open(filenameTXT) as file:

        for line in file.readlines():
            lines.append(line.split())

    return lines


def getTemporalValues(datas):

    timeValues = []

    for data in datas:
        newData = [float(x)
                   for x in data[:2]]
        timeValues.append(tuple(newData))

    return timeValues


def segmentation(source, intervals, filename, destination):

    sr, signal = wavfile.read(source)
    i = 0

    # RC = respiratory cycle

    for interval in intervals:

        startRC = int(interval[0] * sr)
        stopRC = int(interval[1] * sr)

        segment = signal[startRC:stopRC]
        wavfile.write(f'{destination}/{filename}_segment{i}.wav', sr, segment)
        i += 1


def split_train_test(feature_df, division='Dataset/train_test_division.txt'):
    patient_division = readLinesFromFile(division)

    train_list = [patient[0]
                  for patient in patient_division if patient[1] == 'train']
    test_list = [patient[0]
                 for patient in patient_division if patient[1] == 'test']

    # df = pd.read_csv(df_name)

    test_set_df_list = []
    train_set_df_list = []

    for patient_name_test in test_list:
        patient_df = feature_df[feature_df['file'].str.contains(
            patient_name_test)]
        test_set_df_list.append(patient_df)

    test_set_df = pd.concat(test_set_df_list)

    for patient_name_train in train_list:
        patient_df = feature_df[feature_df['file'].str.contains(
            patient_name_train)]
        train_set_df_list.append(patient_df)

    train_set_df = pd.concat(train_set_df_list)

    return train_set_df, test_set_df


def produce_labels(diagnosis='Dataset/Patients_diagnosis.txt'):
    patient_diagnosis = pd.read_csv(
        filepath_or_buffer=diagnosis,
        sep='\t',
        names=['patientID', 'diagnosis'])

    p = patient_diagnosis.copy()
    p.loc[patient_diagnosis['diagnosis'] !=
          'Healthy', 'diagnosis'] = 'Unhealthy'

    # p.loc[patient_diagnosis['diagnosis'] == 'Healthy', 'diagnosis'] = 0

    return p


def split_patients_between_train_and_test(patient_diagnoses, train_set, test_set):

    p = patient_diagnoses.copy()
    recs_train = list(train_set['file'])
    recs_test = list(test_set['file'])

    patient_train_ID = list(set([x[:3] for x in recs_train]))
    patient_test_ID = list(set([x[:3] for x in recs_test]))

    # tirare fuori solo le diagnosi dei pazienti nel train set
    patient_train_list = []

    for id in patient_train_ID:
        patient_train = p[p['patientID'] == int(id)]
        patient_train_list.append(patient_train)

    patient_train_diagnosis = pd.concat(patient_train_list)
    patient_train_diagnosis = patient_train_diagnosis.sort_values('patientID')

    # tirare fuori solo le diagnosi dei pazienti nel test set
    patient_test_list = []

    for id in patient_test_ID:
        patient_test = p[p['patientID'] == int(id)]
        patient_test_list.append(patient_test)

    patient_test_diagnosis = pd.concat(patient_test_list)
    patient_test_diagnosis = patient_test_diagnosis.sort_values('patientID')

    return patient_train_diagnosis, patient_test_diagnosis


def labelsToTestSet(patient_test_diagnosis, test_set):

    # patient_test_diagnosis = pd.read_csv(
    #     'csv/patient_test_diagnosis', index_col=0)
    # test_set_df = pd.read_csv('csv/test_set_dataframe', index_col=0)
    patients = patient_test_diagnosis['patientID'].values
    diags = patient_test_diagnosis['diagnosis'].values

    hh = []

    tsl = test_set['file'].values

    for p, d in zip(patients, diags):
        for rec in tsl:
            if (rec[:3] == str(p)):
                hh.append(d)

    y = pd.DataFrame(test_set['file'])
    y.insert(1, 'diagnosis', hh)

    return y


def labelsToTrainSet(patient_train_diagnosis, train_set):

    # patient_train_diagnosis = pd.read_csv(
    #     'csv/patient_train_diagnosis', index_col=0)
    # train_set_df = pd.read_csv('csv/train_set_dataframe', index_col=0)
    patients = patient_train_diagnosis['patientID'].values
    diags = patient_train_diagnosis['diagnosis'].values

    hh = []

    trl = train_set['file'].values

    for p, d in zip(patients, diags):
        for rec in trl:
            if (rec[:3] == str(p)):
                hh.append(d)

    x = pd.DataFrame(train_set['file'])
    x.insert(1, 'diagnosis', hh)

    return x


def balance_train_set(train_set, labels_x):

    only_sick_labels = labels_x[labels_x['diagnosis'] == 'Unhealthy']
    only_healthy_labels = (labels_x[labels_x['diagnosis'] == 'Healthy'])

    only_sick_features = train_set[train_set['file'].isin(
        only_sick_labels['file'])]
    only_healthy_features = train_set[train_set['file'].isin(
        only_healthy_labels['file'])]

    only_sick_features_balanced = only_sick_features.groupby(
        'file').sample(n=9, replace = True)

    balanced_train_set = pd.concat(
        [only_healthy_features, only_sick_features_balanced]).sort_values('file')
    #balanced_train_set.sort_values('file')

    return balanced_train_set


def groupByPatient(predictions):
    # Group by the first three characters of the first column
    # df = pd.read_csv(preds, index_col=0)
    grouped = predictions.groupby(predictions['file'].str[:3])
    modes = grouped['predictions'].apply(
        lambda x: x.mode().iloc[0]if not x.mode().empty else None)

    return modes


def autoML_fitting(train_set, labels, timeInHours=4, memoryLim=12000):

    # train = pd.read_csv(trainDF, index_col=0)
    train_set = train_set.iloc[:, 3:].values

    # labels = pd.read_csv(labels, index_col=0)
    labels = labels['diagnosis'].values

    engineML = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=int(timeInHours*3600),
        memory_limit=memoryLim
    )

    engineML.fit(X=train_set, y=labels)

    print('Fit completed\n')

    return engineML


def refit_and_predict(train_set, labels_x, test_set, labels_y, ensemble, namePrediciton):

    # REFIT

    train_set = train_set.iloc[:, 3:]
    labels_x = labels_x['diagnosis']

    ensemble.refit(X=train_set.values, y=labels_x.values)

# PREDICTION

    # test = pd.read_csv('csv/LLD/test_set_dataframe')
    test_set = test_set.iloc[:, 3:]

    predict = ensemble.predict(X=test_set.values)

    files = labels_y['file'].values

    if files.shape == predict.shape:
        predictionsDF = pd.DataFrame({'file': files, 'predictions': predict})
        predictionsDF.to_csv(f'csv/predizioni/{namePrediciton}')
    else:
        print('Not same shape')

    return predictionsDF


def extract_features(conf, windowSize, segmentsPath):
    
    smile_lld = opensmile.Smile(feature_set=f'config/compare/{conf}',
                                feature_level='lld'
                                )

    all_features = smile_lld.process_folder(segmentsPath, include_root=False)
    all_features.to_csv('csv/feature_LLD_DataFrame_'+str(windowSize))