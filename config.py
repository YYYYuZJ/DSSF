# *_*coding:utf-8 *_*
import os

DATA_DIR = {
    'CMUMOSI': os.path.join('./dataset', 'CMUMOSI'),    
    'CMUMOSEI': os.path.join('/dataset', 'CMUMOSEI'),  
    'IEMOCAPSix': os.path.join('/dataset', 'IEMOCAP'), 
    'IEMOCAPFour': os.path.join('/dataset', 'IEMOCAP'),
}
PATH_TO_RAW_AUDIO = {
	'CMUMOSI': os.path.join(DATA_DIR['CMUMOSI'], 'subaudio'),
	'CMUMOSEI': os.path.join(DATA_DIR['CMUMOSEI'], 'subaudio'),
	'IEMOCAPSix': os.path.join(DATA_DIR['IEMOCAPSix'], 'subaudio'),
	'IEMOCAPFour': os.path.join(DATA_DIR['IEMOCAPFour'], 'subaudio'),
}
PATH_TO_RAW_FACE = {
	'CMUMOSI': os.path.join(DATA_DIR['CMUMOSI'], 'openface_face'),
	'CMUMOSEI': os.path.join(DATA_DIR['CMUMOSEI'], 'openface_face'),
	'IEMOCAPSix': os.path.join(DATA_DIR['IEMOCAPSix'], 'subvideofaces'), # without openfac
	'IEMOCAPFour': os.path.join(DATA_DIR['IEMOCAPFour'], 'subvideofaces'),
}
PATH_TO_TRANSCRIPTIONS = {
	'CMUMOSI': os.path.join(DATA_DIR['CMUMOSI'], 'transcription.csv'),
	'CMUMOSEI': os.path.join(DATA_DIR['CMUMOSEI'], 'transcription.csv'),
	'IEMOCAPSix': os.path.join(DATA_DIR['IEMOCAPSix'], 'transcription.csv'),
	'IEMOCAPFour': os.path.join(DATA_DIR['IEMOCAPFour'], 'transcription.csv'),
}
PATH_TO_FEATURES = {
	'CMUMOSI': os.path.join(DATA_DIR['CMUMOSI'], 'features'),
	'CMUMOSEI': os.path.join(DATA_DIR['CMUMOSEI'], 'features'),
	'IEMOCAPSix': os.path.join(DATA_DIR['IEMOCAPSix'], 'features'),
	'IEMOCAPFour': os.path.join(DATA_DIR['IEMOCAPFour'], 'features'),
}
PATH_TO_LABEL = {
	'CMUMOSI': os.path.join(DATA_DIR['CMUMOSI'], 'CMUMOSI_features_raw_2way.pkl'),
	'CMUMOSEI': os.path.join(DATA_DIR['CMUMOSEI'], 'CMUMOSEI_features_raw_2way.pkl'),
	'IEMOCAPSix': os.path.join(DATA_DIR['IEMOCAPSix'], 'IEMOCAP_features_raw_6way.pkl'),
	'IEMOCAPFour': os.path.join(DATA_DIR['IEMOCAPFour'], 'IEMOCAP_features_raw_4way.pkl'),
}


# dir
SAVED_ROOT = os.path.join(os.getcwd(), 'saved')
MODEL_DIR = os.path.join(SAVED_ROOT, 'model')
LOG_DIR = os.path.join(SAVED_ROOT, 'log')
NPZ_DIR = os.path.join(SAVED_ROOT, 'npz')
