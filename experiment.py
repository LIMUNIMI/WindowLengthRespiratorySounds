from functions import *
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import balanced_accuracy_score

# link dataset: https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge

def predictIllness(windowSize, conf):
    
    # dopo aver scaricato il dataset, dividere file wav e txt in due cartelle separate
    # LEGGI INTERVALLI TEMPORALI ED ESTRAI SEGMENTI CON LE FUNZIONI readLinesFromFile, getTemporalValues, segmentation
    # read
    datasetPath = './Dataset'
    segmentsPath = f'{datasetPath}/segments'

    print('start extracting features ' + str(windowSize))

    smile_lld = opensmile.Smile(feature_set=f'config/compare/{conf}',
                                feature_level='lld'
                                )

    all_features = smile_lld.process_folder(segmentsPath, include_root=False)
    all_features.to_csv(f'csv/window{str(windowSize)}/feature_LLD_DataFrame_{str(windowSize)}')
    print('features extracted and file stored')


    # DIVIDI IL DATAFRAME IN TRAIN SET E TEST SET
    # train_test_division.txt è un file che specifica per quale fase è stato usato il singolo file, situato nella cartella del dataset.
    train_set, test_set = split_train_test(feature_df=pd.read_csv(f'csv/window{str(windowSize)}/feature_LLD_DataFrame_{str(windowSize)}'),
                                           division=f'{datasetPath}/train_test_division.txt',
                                           )
    train_set.sort_values('file')
    test_set.sort_values('file')

    print('train and test features divided\n')

    # PRODUCI LE LABEL, quelle di train già bilanciate! (modifica funzione labelsToTrainSet)
    # healthy_or_unhealthy è stato prodotto utilizzando la funzione produce_labels, dal file Patients_diagnosis.txt del dataset
    print('producing labels\n')
    patient_diagnoses = pd.read_csv(
        f'{datasetPath}/healthy_or_unhealthy', index_col=0)

    patient_train_diagnosis, patient_test_diagnosis = split_patients_between_train_and_test(
        patient_diagnoses, train_set, test_set)

    patient_train_diagnosis.to_csv(f'csv/window{str(windowSize)}/patient_train_diagnosis'+ str(windowSize))
    patient_test_diagnosis.to_csv(
        f'csv/window{str(windowSize)}/patient_test_diagnosis'+ str(windowSize))  # this is our ground truth on the whole patient

    labels_x = labelsToTrainSet(patient_train_diagnosis, train_set)
    labels_x.to_csv(f'csv/window{str(windowSize)}/label_x_not_balanced'+ str(windowSize))

    labels_y = labelsToTestSet(
        patient_test_diagnosis, test_set)  # ground truth
    labels_y.to_csv(f'csv/window{str(windowSize)}/label_y'+ str(windowSize))

    # BILANCIA IL TRAIN SET GENERA LE LABELS X BILANCIATE
    print('balancing labels\n')

    train_set_balanced = balance_train_set(train_set, labels_x)
    train_set_balanced.to_csv(f'csv/window{str(windowSize)}/balanced_train_set'+ str(windowSize))

    labels_x_balanced = labelsToTrainSet(patient_train_diagnosis, train_set_balanced)
    labels_x_balanced.to_csv(f'csv/window{str(windowSize)}/balanced_label_x'+ str(windowSize))

    # ESTRAI ENSEMBLE CON AUTOSKLEARN
    print('start fitting\n')
    ensemble_name = 'half_hour'
    engineML = autoML_fitting(
        train_set_balanced, labels_x_balanced, timeInHours=0.5, memoryLim=12000)

    #engineML.show_models()

    with open(f'./Models_Ensembles/{ensemble_name}_{str(windowSize)}', 'wb') as M:
        pickle.dump(engineML, M, pickle.HIGHEST_PROTOCOL)

    # REFIT, PREDICTION E METRICHE DI VALUTAZIONE
    predictions = refit_and_predict(train_set_balanced, labels_x_balanced, test_set,
                                    labels_y, engineML, 'prediction_'+str(windowSize))

    modes = groupByPatient(predictions)

    ground_truth = patient_test_diagnosis

    mode_values = modes.values
    truth = ground_truth['diagnosis'].values

    comparation = pd.DataFrame(
        data={'ID': ground_truth['patientID'], 'truth': truth, 'prediction': mode_values})
    comparation.to_csv(f'./csv/window{str(windowSize)}/comparation_{str(windowSize)}')

    # METRICHE (magari fai una funzione a parte)

    mcc = matthews_corrcoef(y_true=truth, y_pred=mode_values)
    precision = precision_score(
        y_true=truth, y_pred=mode_values, pos_label='Unhealthy')
    recall = recall_score(
        y_true=truth, y_pred=mode_values, pos_label='Unhealthy')
    f1 = f1_score(y_true=truth, y_pred=mode_values, pos_label='Unhealthy')
    balanced_score = balanced_accuracy_score(y_true=truth, y_pred=mode_values)

    metrics = {'window scale factor': windowSize, 'MCC': mcc, 'precision': precision,
               'recall': recall, 'f1 score': f1, 'Balanced accuracy score': balanced_score}

    with open('results/metrics'+str(windowSize), 'w') as f:
        f.write(str(metrics))

    return metrics






sizes = [0.5, 0.6, 0.75, 0.8, 1.0, 1.25, 1.5, 2.0, 4.0]
configurations = ['features0_5.conf',
                  'features0_6.conf',
                  'features0_75.conf', 
                  'features0_8.conf', 
                  'features1_0.conf', 
                  'features1_25.conf', 
                  'features1_5.conf',
                  'features2_0.conf',
                  'features4_0.conf']


for size, conf in zip(sizes, configurations):

    results = predictIllness(size, conf)


