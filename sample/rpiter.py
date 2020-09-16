import math
import os
import sys
import time
from argparse import ArgumentParser
from functools import reduce

import configparser
import numpy as np
import tensorflow as tf
from keras import optimizers
from keras.layers import Dense, concatenate, BatchNormalization
from keras.layers import Dropout
from keras.models import Model
from keras.utils import np_utils
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from utils.callbacks import AccHistoryPlot, EarlyStopping
from utils.basic_modules import conjoint_cnn, conjoint_sae
from utils.sequence_encoder import ProEncoder, RNAEncoder
from utils.stacked_auto_encoder import train_auto_encoder


# default program settings
DATA_SET = 'RPI488'
TIME_FORMAT = "-%y-%m-%d-%H-%M-%S"

WINDOW_P_UPLIMIT = 3
WINDOW_P_STRUCT_UPLIMIT = 3
WINDOW_R_UPLIMIT = 4
WINDOW_R_STRUCT_UPLIMIT = 4
VECTOR_REPETITION_CNN = 1
RANDOM_SEED = 1
K_FOLD = 5
BATCH_SIZE = 150
FIRST_TRAIN_EPOCHS = [20, 20, 20, 20, 10]
SECOND_TRAIN_EPOCHS = [20, 20, 20, 20, 10]
PATIENCES = [10, 10, 10, 10, 10]
FIRST_OPTIMIZER = 'adam'
SECOND_OPTIMIZER = 'sgd'
SGD_LEARNING_RATE = 0.001
ADAM_LEARNING_RATE = 0.001
FREEZE_SUB_MODELS = True
CODING_FREQUENCY = True
MONITOR = 'acc'
MIN_DELTA = 0.0
SHUFFLE = True
VERBOSE = 2

# get the path of rpiter.py
script_dir, script_name = os.path.split(os.path.abspath(sys.argv[0]))
parent_dir = os.path.dirname(script_dir)
# set paths of data, results and program parameters
DATA_BASE_PATH = parent_dir + '/data/'
RESULT_BASE_PATH = parent_dir + '/result/' 
INI_PATH = script_dir + '/utils/data_set_settings.ini'

metrics_whole = {'RPITER': np.zeros(6),
                 'Conjoint-SAE': np.zeros(6), 'Conjoint-Struct-SAE': np.zeros(6),
                 'Conjoint-CNN': np.zeros(6), 'Conjoint-Struct-CNN': np.zeros(6)}

parser = ArgumentParser()
parser.add_argument('-d', '--dataset', type=str, help='The dataset you want to process.')
args = parser.parse_args()
if args.dataset != None:
    DATA_SET = args.dataset
print("Dataset: %s" % DATA_SET)

# gpu memory growth manner for TensorFlow
# to consider version compatibility
if tf.__version__< '2.0':
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
else:
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)

# set visible gpu if needed
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# set result save path
result_save_path = RESULT_BASE_PATH + DATA_SET + "/" + DATA_SET + time.strftime(TIME_FORMAT, time.localtime()) + "/"
# os.path.dirname(script_dir) + '/result/data/RPI488
if not os.path.exists(result_save_path):
    os.mkdir(result_save_path)
out = open(result_save_path + 'result.txt', 'w')


def read_data_pair(path):
    # _pair.txt (RNA sequence, Protein sequence, label 을 묶어서 pair을 표시해놓은 파일) 을 열어서 결합하는거끼리, 안하는거끼리 2차원 배열 2개를
    # 만드는 함수.
    pos_pairs = []
    neg_pairs = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            p, r, label = line.split('\t')
            if label == '1':
                pos_pairs.append((p, r))
            elif label == '0':
                neg_pairs.append((p, r))
    return pos_pairs, neg_pairs


def read_data_seq(path):
    # .fa 파일을 읽어서 (번호 : seqneuce) 이렇게 딕셔너리를 만들어주는 함수.
    seq_dict = {}
    with open(path, 'r') as f:
        name = ''
        for line in f:
            line = line.strip()
            if line[0] == '>':
                name = line[1:]
                seq_dict[name] = ''
            else:
                if line.startswith('XXX'):
                    seq_dict.pop(name)
                else:
                    seq_dict[name] = line
    return seq_dict


# calculate the six metrics of Acc, Sn, Sp, Precision, MCC and AUC
def calc_metrics(y_label, y_proba):
    con_matrix = confusion_matrix(y_label, [1 if x >= 0.5 else 0 for x in y_proba])
    TN = float(con_matrix[0][0])
    FP = float(con_matrix[0][1])
    FN = float(con_matrix[1][0])
    TP = float(con_matrix[1][1])
    P = TP + FN
    N = TN + FP
    Sn = TP / P if P > 0 else 0
    Sp = TN / N if N > 0 else 0
    Acc = (TP + TN) / (P + N) if (P + N) > 0 else 0
    Pre = (TP) / (TP + FP) if (TP+FP) > 0 else 0
    MCC = 0
    tmp = math.sqrt((TP + FP) * (TP + FN)) * math.sqrt((TN + FP) * (TN + FN))
    if tmp != 0:
        MCC = (TP * TN - FP * FN) / tmp
    fpr, tpr, thresholds = roc_curve(y_label, y_proba)
    AUC = auc(fpr, tpr)
    return Acc, Sn, Sp, Pre, MCC, AUC


def load_data(data_set):
    pro_seqs = read_data_seq(DATA_BASE_PATH + "sequence/" + data_set + '_protein_seq.fa')
    rna_seqs = read_data_seq(DATA_BASE_PATH + "sequence/" + data_set + '_rna_seq.fa')
    pro_structs = read_data_seq(DATA_BASE_PATH + "structure/" + data_set + '_protein_struct.fa')
    rna_structs = read_data_seq(DATA_BASE_PATH + "structure/" + data_set + '_rna_struct.fa')
    pos_pairs, neg_pairs = read_data_pair(DATA_BASE_PATH + data_set + '_pairs.txt')
    
    #pos_pairs : positive (RNA sequence, Protein sequence, label)
    #neg_pairs : negative (RNA sequence, Protein sequence, label)
    #pro_seqs : protein (번호 : seqneuce)
    #rna_seqs : RNA (번호 : seqneuce)
    #pro_structs : protein-struct (번호 : seqneuce)
    #rna_structs : RNA-struct (번호 : seqneuce)
    #print('pos_pairs : {}'.format(pos_pairs))
    #print('neg_pairs : {}'.format(neg_pairs))
    #print('pro_seqs : {}'.format(pro_seqs))
    #print('rna_seqs : {}'.format(rna_seqs))
    #print('pro_structs : {}'.format(pro_structs))
    #print('rna_structs : {}'.format(rna_structs))
    return pos_pairs, neg_pairs, pro_seqs, rna_seqs, pro_structs, rna_structs


def coding_pairs(pairs, pro_seqs, rna_seqs, pro_structs, rna_structs, PE, RE, kind):
    # pair (p_sequence, r_sequence)에서 각 feature를 순회하면서 feature들의 value를 추출하고, 그것을 정규화시킨 것을 배열로 만듬.
    # p_sequence에 해당하는 p_struct를 인코딩해서 배열로 만들고, p_sequence의 값과 concatenate시킴.
    # kind = 1 (positive) or 0 (negative)인 Flag
     
    samples = []
    for pr in pairs:
        #print('kind : {0} - pair : {1} - struct : {2}'.format(kind, pr, pr[0] in pro_seqs and pr[1] in rna_seqs and pr[0] in pro_structs and pr[1] in rna_structs))
        if pr[0] in pro_seqs and pr[1] in rna_seqs and pr[0] in pro_structs and pr[1] in rna_structs:
            # 이 if 문은 결측치를 처리하기 위함임. 결측치가 포함된 pair를 단순히 제외시킨다.
            # pr[0] in pro_structs 라는게, pro_structs 딕셔너리의 key 값중 pr[0]이 포함되어있는지 보는것인 듯.
            p_seq = pro_seqs[pr[0]]  # protein sequence
            r_seq = rna_seqs[pr[1]]  # rna sequence
            p_struct = pro_structs[pr[0]]  # protein structure
            r_struct = rna_structs[pr[1]]  # rna structure

            p_conjoint = PE.encode_conjoint(p_seq) # protein sequence를 인코딩함. feature마다 count된 value를 최대값으로 나눈 정규화 사용. 그외 동일
            r_conjoint = RE.encode_conjoint(r_seq)
            p_conjoint_struct = PE.encode_conjoint_struct(p_seq, p_struct)
            # struct 파일도 sequence와 완전 동일한 방법으로 인코딩. result인, '정규화된 값으로 구성된 배열'들을 concatenate시켜서 결과로 리턴.
        
            r_conjoint_struct = RE.encode_conjoint_struct(r_seq, r_struct)

            if p_conjoint is 'Error':
                print('Skip {} in pair {} according to conjoint coding process.'.format(pr[0], pr))
            elif r_conjoint is 'Error':
                print('Skip {} in pair {} according to conjoint coding process.'.format(pr[1], pr))
            elif p_conjoint_struct is 'Error':
                print('Skip {} in pair {} according to conjoint_struct coding process.'.format(pr[0], pr))
            elif r_conjoint_struct is 'Error':
                print('Skip {} in pair {} according to conjoint_struct coding process.'.format(pr[1], pr))

            else:
                samples.append([[p_conjoint, r_conjoint],
                                [p_conjoint_struct, r_conjoint_struct],
                                kind])
        else:
            print('Skip pair {} according to sequence dictionary.'.format(pr))
            
    # samples (4차원 배열) : "[[p_conjoint, r_conjoint],[p_conjoint_struct, r_conjoint_struct], kind]" 이게 원소로 들어가있음.
    # p_conjoint, p_conjoint_struct는 feature들의 value가 정규화되어 들어가있는 1차원 배열.
    return samples


def standardization(X):
    # https://datascienceschool.net/view-notebook/f43be7d6515b48c0beb909826993c856/
    # StandardScalar : 평균이 0과 표준편차가 1이 되도록 변환.
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    return X, scaler


def pre_process_data(samples, samples_pred=None):
    # parameter samples는 아래와 같이 생겼음.
    # [ [[p1_conjoint, r1_conjoint],[p1_conjoint_struct, r1_conjoint_struct],kind], 
    #   [[p2_conjoint, r2_conjoint],[p2_conjoint_struct, r2_conjoint_struct],kind],
    #   [[p3_conjoint, r3_conjoint],[p3_conjoint_struct, r3_conjoint_struct],kind],
    #    ...
    #   [[pN_conjoint, rN_conjoint],[pN_conjoint_struct, rN_conjoint_struct],kind]
    # ]
    
    # np.random.shuffle(samples)
    #print('samples : {}'.format(samples))

    p_conjoint = np.array([x[0][0] for x in samples]) # x[?][0][0] : p_conjoint
    r_conjoint = np.array([x[0][1] for x in samples]) # x[?][0][1] : r_conjoint
    p_conjoint_struct = np.array([x[1][0] for x in samples]) # x[?][1][0] : p_conjoint_struct
    r_conjoint_struct = np.array([x[1][1] for x in samples]) # x[?][1][1] : r_conjoint_struct
    y_samples = np.array([x[2] for x in samples]) # x[2] : kind
    # p_conjoint (2차원 배열) (이 시점에서의 p_conjoint) : [p1_conjoint, p2_conjoint, p3_conjoint ... ]
    # r_conjoint (2차원 배열) (이 시점에서의 r_conjoint) : [r1_conjoint, r2_conjoint, r3_conjoint ... ]
    # p_conjoint_struct (2차원 배열) (이 시점에서의 p_conjoint_struct) : [p1_conjoint_struct, p2_conjoint_struct, ... ]
    # r_conjoint_struct (2차원 배열) (이 시점에서의 r_conjoint_struct) : [r1_conjoint_struct, r2_conjoint_struct, ... ]
    
    # p_conjoint length : 488
    # p_conjoint[0] length : 343 

    #print('before standardization : {}'.format(p_conjoint)) 
    p_conjoint, scaler_p = standardization(p_conjoint)
    #print('after standardization - p_conjoint : {0} - scaler_p : {1}'.format(p_conjoint, scaler_p))
    
    r_conjoint, scaler_r = standardization(r_conjoint)
    p_conjoint_struct, scaler_p_struct = standardization(p_conjoint_struct)
    r_conjoint_struct, scaler_r_struct = standardization(r_conjoint_struct)

    #print('p_conjoint : {0} - len of p_conjoint : {1} - len of p_conjoint[0] : {2}'.format(p_conjoint, len(p_conjoint), len(p_conjoint[0])))
    p_conjoint_cnn = np.array([list(map(lambda e: [e] * VECTOR_REPETITION_CNN, x)) for x in p_conjoint])
    # map과 람다 함수 : map(function, iterator) -> iterator (e.g. list, tuple, ... )의 각 요소를 function의 파라미터로 넣어서 실행시킨다.
    # p_conjoint의 각 인자와 VECTOR_REPETITION_CNN를 곱함.
    #print('p_conjoint_cnn : {}'.format(p_conjoint_cnn))
    
    r_conjoint_cnn = np.array([list(map(lambda e: [e] * VECTOR_REPETITION_CNN, x)) for x in r_conjoint])
    p_conjoint_struct_cnn = np.array([list(map(lambda e: [e] * VECTOR_REPETITION_CNN, x)) for x in p_conjoint_struct])
    r_conjoint_struct_cnn = np.array([list(map(lambda e: [e] * VECTOR_REPETITION_CNN, x)) for x in r_conjoint_struct])

    p_ctf_len = 7 ** WINDOW_P_UPLIMIT # WINDOW_P_UPLIMIT : 3, 7 ** WINDOW_P_UPLIMI : 343
    r_ctf_len = 4 ** WINDOW_R_UPLIMIT # WINDOW_R_UPLIMIT : 4, 4 ** WINDOW_R_UPLIMIT : 256
    p_conjoint_previous = np.array([x[-p_ctf_len:] for x in p_conjoint]) 
    # 각 p_sequence의 정규화 배열에 대해 인덱스가 뒤에서부터 343번째 까지인 것만 배열로 만듬. -> 세자리 알파벳인 feature의 정규값 원소만 뽑겠다!
    # 파이썬 배열 인덱싱 - 마이너스가 붙으면 뒤에서부터 탐색. 
    # p_sequence 배열의 인덱스의 의미 !!
    # p_sequence 0~6 : 한 자리 알파벳으로 이루어진 원소
    # p_sequence 7~55 : 두 자리 알파벳으로 이루어진 원소
    # p_sequence 56~398 : 세 자리 알파벳으로 이루어진 원소
    
    #print('p_conjoint_previous : {0} - p_ctf_len : {1} - len of p_conjoint_previous[0] : {2}'.format(p_conjoint_previous, p_ctf_len, len(p_conjoint_previous[0])))
    
    r_conjoint_previous = np.array([x[-r_ctf_len:] for x in r_conjoint])

    X_samples = [[p_conjoint, r_conjoint],
                 [p_conjoint_struct, r_conjoint_struct],
                 [p_conjoint_cnn, r_conjoint_cnn],
                 [p_conjoint_struct_cnn, r_conjoint_struct_cnn],
                 [p_conjoint_previous, r_conjoint_previous]
                 ]


    if samples_pred:
        # np.random.shuffle(samples_pred)

        p_conjoint_pred = np.array([x[0][0] for x in samples_pred])
        r_conjoint_pred = np.array([x[0][1] for x in samples_pred])
        p_conjoint_struct_pred = np.array([x[1][0] for x in samples_pred])
        r_conjoint_struct_pred = np.array([x[1][1] for x in samples_pred])
        y_samples_pred = np.array([x[2] for x in samples_pred])

        p_conjoint_pred = scaler_p.transform(p_conjoint_pred)
        r_conjoint_pred = scaler_r.transform(r_conjoint_pred)
        p_conjoint_struct_pred = scaler_p_struct.transform(p_conjoint_struct_pred)
        r_conjoint_struct_pred = scaler_r_struct.transform(r_conjoint_struct_pred)

        p_conjoint_cnn_pred = np.array([list(map(lambda e: [e] * VECTOR_REPETITION_CNN, x)) for x in p_conjoint_pred])
        r_conjoint_cnn_pred = np.array([list(map(lambda e: [e] * VECTOR_REPETITION_CNN, x)) for x in r_conjoint_pred])
        p_conjoint_struct_cnn_pred = np.array(
            [list(map(lambda e: [e] * VECTOR_REPETITION_CNN, x)) for x in p_conjoint_struct_pred])
        r_conjoint_struct_cnn_pred = np.array(
            [list(map(lambda e: [e] * VECTOR_REPETITION_CNN, x)) for x in r_conjoint_struct_pred])

        p_conjoint_previous_pred = np.array([x[-p_ctf_len:] for x in p_conjoint_pred])
        r_conjoint_previous_pred = np.array([x[-r_ctf_len:] for x in r_conjoint_pred])

        X_samples_pred = [[p_conjoint_pred, r_conjoint_pred],
                          [p_conjoint_struct_pred, r_conjoint_struct_pred],
                          [p_conjoint_cnn_pred, r_conjoint_cnn_pred],
                          [p_conjoint_struct_cnn_pred, r_conjoint_struct_cnn_pred],
                          [p_conjoint_previous_pred, r_conjoint_previous_pred]
                          ]

        return X_samples, y_samples, X_samples_pred, y_samples_pred

    else:
        
            
    
        # p_conjoint : 2차원 [p1_conjoint, p2_conjoint, p3_conjoint ... ] 각각의 pN_conjoint는 정규화 값이 담긴 1차원 배열. len : 488 
        # r_conjoint : 
        # p_conjoint_struct : 2차원 [p1_conjoint_struct, p2_conjoint_struct, ... ] len : 488
        # r_conjoint_struct : 
        # p_conjoint_cnn : 2차원 [각 pN_conjoint의 각 정규화값에 VECTOR_REPETITION_CNN이 곱해짐. ] len : 488 
        # r_conjoint_cnn : 
        # p_conjoint_struct_cnn : 2차원 [각 pN_conjoint_struct의 각 정규화값에 VECTOR_REPETITION_CNN이 곱해짐. ] len : 488 
        # r_conjoint_struct_cnn : 
        # p_conjoint_previous : 2차원. p_sequence 각각의 배열 중, 세자리 알파벳인 feature인 정규값만 뽑은 배열. len : 488
        # r_conjoint_previous : 
        
        # X_samples : [[p_conjoint, r_conjoint],
            #         [p_conjoint_struct, r_conjoint_struct],
            #         [p_conjoint_cnn, r_conjoint_cnn],
            #         [p_conjoint_struct_cnn, r_conjoint_struct_cnn],
            #         [p_conjoint_previous, r_conjoint_previous]
            #         ]
        # y_samples : 1차원. kind 만 모아놓은 배열. len : 488 
        return X_samples, y_samples


def sum_power(num, bottom, top): # arguments : (7, 1, P_WINDOW_UP)
    return reduce(lambda x, y: x + y, map(lambda x: num ** x, range(bottom, top + 1))) 
    # 1,2,3 에 대해 [7^1, 7^2, 7^3] 인 배열을 만들고, ((7^1 + 7^2) + 7^3) 을 한다.
    # 결과 : 399


def get_callback_list(patience, result_path, stage, fold, X_test, y_test):
    # patience = PATIENCES[0] = 10
    # result_path : ~/result/data/RPI488/...
    # stage = 'Conjoint-CNN'
    # fold = 현재 for 문으로 순회하고 있는 값. range(0, 5)
    # X_test = X_test_conjoint_cnn : 3차원. [각 pN_conjoint의 각 정규화값에 VECTOR_REPETITION_CNN이 곱해짐. ] 중 test의 인덱스만. len : 2
    # y_test : kind 값들 중 test값들만 뽑인 것 일듯.
    
    earlystopping = EarlyStopping(monitor=MONITOR, min_delta=MIN_DELTA, patience=patience, verbose=1,
                                  mode='auto', restore_best_weights=True)
    # MONITOR = 'acc', MIN_DELTA = 0.0, patience = 10
    
    acchistory = AccHistoryPlot([stage, fold], [X_test, y_test], data_name=DATA_SET,
                                result_save_path=result_path, validate=0, plot_epoch_gap=10)
    # data_name = RPI488 (default)
    # 

    return [acchistory, earlystopping]


def get_optimizer(opt_name):
    if opt_name == 'sgd':
        return optimizers.sgd(lr=SGD_LEARNING_RATE, momentum=0.5)
    elif opt_name == 'adam':
        return optimizers.adam(lr=ADAM_LEARNING_RATE)
    else:
        return opt_name


def control_model_trainable(model, trainable):
    for layer in model.layers:
        layer.trainable = trainable


def get_auto_encoders(X_train, X_test, batch_size=BATCH_SIZE):
    # X_train : 3차원. [p1_conjoint, p2_conjoint, p3_conjoint ... ] 중 train의 인덱스만 선택, RNA의 경우도 마찬가지. len : 2
    # X_test : 3차원. [p1_conjoint_struct, p2_conjoint_struct, ... ] 중 train의 인덱스만 선택, RNA의 경우도 마찬가지. len : 2
    # BATCH_SIZE = 150

    
    # X_train : 3차원. [p1_conjoint, p2_conjoint, p3_conjoint ... ] 중 train의 인덱스만 선택, RNA의 경우도 마찬가지. len : 2
    # X_test : 3차원. [p1_conjoint_struct, p2_conjoint_struct, ... ] 중 train의 인덱스만 선택, RNA의 경우도 마찬가지. len : 2
    # X_train[0] : (390, 399), X_train[0].shape[1] : 399
    # BATCH_SIZE = 150
    encoders_protein, decoders_protein, train_tmp_p, test_tmp_p = train_auto_encoder(
        X_train=X_train[0],
        X_test=X_test[0],
        layers=[X_train[0].shape[1], 256, 128, 64], batch_size=batch_size)
    print('X_train[0].shape[1] : {}'.format(X_train[0].shape[1]))
    print('X_train[0].shape : {}'.format(X_train[0].shape))
    # train_encoders : keras.layers.core.Dense 객체. [Dense(256), Dense(128), Dense(128)]. 399->256 처럼 인코딩시켜서 수를 줄이는 객체가 담긴듯.
    # trained_decoders : keras.layers.core.Dense 객체. [Dense(399), Dense(256), Dense(128)]. 256->399 처럼 디코딩시켜서 수를 늘리는 객체가 담긴듯.
    # X_train_tmp : 2차원. [p1_conjoint, p2_conjoint, p3_conjoint ... ], 각각의 pN_conjoint는 64개의 원소로 압축됨.
    # X_test_tmp : 

    
    # X_train[1].shape : (390, 340), X_train[1].shape[1] : 340
    encoders_rna, decoders_rna, train_tmp_r, test_tmp_r = train_auto_encoder(
        X_train=X_train[1],
        X_test=X_test[1],
        layers=[X_train[1].shape[1], 256, 128, 64], batch_size=batch_size)
    
    # encoders_protein : keras.layers.core.Dense 객체. [Dense(256), Dense(128), Dense(128)]. 399->256 처럼 인코딩시켜서 수를 줄이는 객체가 담긴듯.
    # encoders_rna : keras.layers.core.Dense 객체. [Dense(256), Dense(128), Dense(128)]. 399->256 처럼 인코딩시켜서 수를 줄이는 객체가 담긴듯.
    return encoders_protein, encoders_rna


# load data settings
if DATA_SET in ['RPI369', 'RPI488', 'RPI1807', 'RPI2241', 'NPInter']:
    # 이건 command line 인자로 다른 데이터셋을 받았을 때 전역상수들을 다시 지정해주는 구문임.
    
    config = configparser.ConfigParser()
    config.read(INI_PATH)
    WINDOW_P_UPLIMIT = config.getint(DATA_SET, 'WINDOW_P_UPLIMIT')
    WINDOW_P_STRUCT_UPLIMIT = config.getint(DATA_SET, 'WINDOW_P_STRUCT_UPLIMIT')
    WINDOW_R_UPLIMIT = config.getint(DATA_SET, 'WINDOW_R_UPLIMIT')
    WINDOW_R_STRUCT_UPLIMIT = config.getint(DATA_SET, 'WINDOW_R_STRUCT_UPLIMIT')
    VECTOR_REPETITION_CNN = config.getint(DATA_SET, 'VECTOR_REPETITION_CNN')
    RANDOM_SEED = config.getint(DATA_SET, 'RANDOM_SEED')
    K_FOLD = config.getint(DATA_SET, 'K_FOLD')
    BATCH_SIZE = config.getint(DATA_SET, 'BATCH_SIZE')
    PATIENCES = [int(x) for x in config.get(DATA_SET, 'PATIENCES').replace('[', '').replace(']', '').split(',')]
    # PATIENCES = [10, 10, 10, 10, 10]
    
    FIRST_TRAIN_EPOCHS = [int(x) for x in
                          config.get(DATA_SET, 'FIRST_TRAIN_EPOCHS').replace('[', '').replace(']', '').split(',')]
    SECOND_TRAIN_EPOCHS = [int(x) for x in
                           config.get(DATA_SET, 'SECOND_TRAIN_EPOCHS').replace('[', '').replace(']', '').split(',')]
    FIRST_OPTIMIZER = config.get(DATA_SET, 'FIRST_OPTIMIZER')
    SECOND_OPTIMIZER = config.get(DATA_SET, 'SECOND_OPTIMIZER')
    SGD_LEARNING_RATE = config.getfloat(DATA_SET, 'SGD_LEARNING_RATE')
    ADAM_LEARNING_RATE = config.getfloat(DATA_SET, 'ADAM_LEARNING_RATE')
    FREEZE_SUB_MODELS = config.getboolean(DATA_SET, 'FREEZE_SUB_MODELS')
    CODING_FREQUENCY = config.getboolean(DATA_SET, 'CODING_FREQUENCY')
    MONITOR = config.get(DATA_SET, 'MONITOR')
    MIN_DELTA = config.getfloat(DATA_SET, 'MIN_DELTA')

# write program parameter settings to result file
settings = (
    """# Analyze data set {}\n
Program parameters:
WINDOW_P_UPLIMIT = {},
WINDOW_R_UPLIMIT = {},
WINDOW_P_STRUCT_UPLIMIT = {},
WINDOW_R_STRUCT_UPLIMIT = {},
VECTOR_REPETITION_CNN = {},
RANDOM_SEED = {},
K_FOLD = {},
BATCH_SIZE = {},
FIRST_TRAIN_EPOCHS = {},
SECOND_TRAIN_EPOCHS = {},
PATIENCES = {},
FIRST_OPTIMIZER = {},
SECOND_OPTIMIZER = {},
SGD_LEARNING_RATE = {},
ADAM_LEARNING_RATE = {},
FREEZE_SUB_MODELS = {},
CODING_FREQUENCY = {},
MONITOR = {},
MIN_DELTA = {},
    """.format(DATA_SET, WINDOW_P_UPLIMIT, WINDOW_R_UPLIMIT, WINDOW_P_STRUCT_UPLIMIT,
               WINDOW_R_STRUCT_UPLIMIT, VECTOR_REPETITION_CNN,
               RANDOM_SEED, K_FOLD, BATCH_SIZE, FIRST_TRAIN_EPOCHS, SECOND_TRAIN_EPOCHS, PATIENCES, FIRST_OPTIMIZER,
               SECOND_OPTIMIZER, SGD_LEARNING_RATE, ADAM_LEARNING_RATE,
               FREEZE_SUB_MODELS, CODING_FREQUENCY, MONITOR, MIN_DELTA)
)
out.write(settings)

PRO_CODING_LENGTH = sum_power(7, 1, WINDOW_P_UPLIMIT)
PRO_STRUCT_CODING_LENGTH = sum_power(7, 1, WINDOW_P_UPLIMIT) + sum_power(3, 1, WINDOW_P_STRUCT_UPLIMIT)
RNA_CODING_LENGTH = sum_power(4, 1, WINDOW_R_UPLIMIT)
RNA_STRUCT_CODING_LENGTH = sum_power(4, 1, WINDOW_R_UPLIMIT) + sum_power(2, 1, WINDOW_R_STRUCT_UPLIMIT)

# read rna-protein pairs and sequences from data files
pos_pairs, neg_pairs, pro_seqs, rna_seqs, pro_structs, rna_structs = load_data(DATA_SET)

# sequence encoder instances
PE = ProEncoder(WINDOW_P_UPLIMIT, WINDOW_P_STRUCT_UPLIMIT, CODING_FREQUENCY, VECTOR_REPETITION_CNN)
RE = RNAEncoder(WINDOW_R_UPLIMIT, WINDOW_R_STRUCT_UPLIMIT, CODING_FREQUENCY, VECTOR_REPETITION_CNN)

print("Coding positive protein-rna pairs.\n")
samples = coding_pairs(pos_pairs, pro_seqs, rna_seqs, pro_structs, rna_structs, PE, RE, kind=1)
positive_sample_number = len(samples)
#print("positive_sample_number : {}".format(positive_sample_number))
print("Coding negative protein-rna pairs.\n")
samples += coding_pairs(neg_pairs, pro_seqs, rna_seqs, pro_structs, rna_structs, PE, RE, kind=0)
negative_sample_number = len(samples) - positive_sample_number
sample_num = len(samples)
#print("negative_sample_number : {}".format(negative_sample_number))
#print("sample_num : {}".format(sample_num))

# positive and negative sample numbers
print('\nPos samples: {}, Neg samples: {}.\n'.format(positive_sample_number, negative_sample_number))
out.write('\nPos samples: {}, Neg samples: {}.\n'.format(positive_sample_number, negative_sample_number))
print('samples : {0}, {1}, {2}, {3}'.format(len(samples), len(samples[0]), len(samples[0][0]), len(samples[0][0][0])))
sys.exit()
X, y = pre_process_data(samples=samples)


# K-fold CV processes

# skf = StratifiedKFold(n_splits=K_FOLD, random_state=RANDOM_SEED, shuffle=True)
print('\n\nK-fold cross validation processes:\n')
out.write('\n\nK-fold cross validation processes:\n')
for fold in range(K_FOLD):
    # p_conjoint : 2차원 [p1_conjoint, p2_conjoint, p3_conjoint ... ] 각각의 pN_conjoint는 정규화 값이 담긴 1차원 배열. len : 488 
    # r_conjoint : 
    # p_conjoint_struct : 2차원 [p1_conjoint_struct, p2_conjoint_struct, ... ] len : 488
    # r_conjoint_struct : 
    # p_conjoint_cnn : 2차원 [각 pN_conjoint의 각 정규화값에 VECTOR_REPETITION_CNN이 곱해짐. ] len : 488 
    # r_conjoint_cnn : 
    # p_conjoint_struct_cnn : 2차원 [각 pN_conjoint_struct의 각 정규화값에 VECTOR_REPETITION_CNN이 곱해짐. ] len : 488 
    # r_conjoint_struct_cnn : 
    # p_conjoint_previous : 2차원. p_sequence 각각의 배열 중, 세자리 알파벳인 feature인 정규값만 뽑은 배열. len : 488
    # r_conjoint_previous : 

    # X_samples : [[p_conjoint, r_conjoint],
    #              [p_conjoint_struct, r_conjoint_struct],
    #              [p_conjoint_cnn, r_conjoint_cnn],
    #              [p_conjoint_struct_cnn, r_conjoint_struct_cnn],
    #              [p_conjoint_previous, r_conjoint_previous]
    #             ]
    
    
    #print('sample_num  : {}'.format(sample_num))
    train = [i for i in range(sample_num) if i%K_FOLD !=fold] # 0~487 인 인덱스 중에서 80%를 train set으로 [1,2,3,4,  6,7,8,9,  11,12,13,14 ... ]
    test = [i for i in range(sample_num) if i%K_FOLD ==fold] # 0~487 인 인덱스 중에서 20%를 test set으로 [0, 5, 10, 15, ... ]

    # generate train and test data
    X_train_conjoint = [X[0][0][train], X[0][1][train]] 
    X_train_conjoint_struct = [X[1][0][train], X[1][1][train]] 
    X_train_conjoint_cnn = [X[2][0][train], X[2][1][train]] 
    X_train_conjoint_struct_cnn = [X[3][0][train], X[3][1][train]] 
    X_train_conjoint_previous = [X[4][0][train], X[4][1][train]]
    # 여기의 모든 변수들은 [p의 값, r의 값]과 같이 RNA가 쌍으로 묶여 있음.
    
    # X_train_conjoint : 3차원. [p1_conjoint, p2_conjoint, p3_conjoint ... ] 중 train의 인덱스만 선택, RNA의 경우도 마찬가지. len : 2
    # X_train_conjoint_struct : 3차원. [p1_conjoint_struct, p2_conjoint_struct, ... ] 중 train의 인덱스만 선택, RNA의 경우도 마찬가지. len : 2
    # X_train_conjoint_cnn : 3차원. [각 pN_conjoint의 각 정규화값에 VECTOR_REPETITION_CNN이 곱해짐. ] 중 train의 인덱스만. len : 2
    # X_train_conjoint_struct_cnn : 4차원. [각 pN_conjoint_struct의 각 정규화값에 VECTOR_REPETITION_CNN이 곱해짐. ] 중 train의 인덱스만. len : 2
    # X_train_conjoint_previous : 4차원. [p1_conjoint_previous, p2_conjoint_previous, p3_conjoint_previous, ... ] 중 train의 인덱스만 선택, len : 2
    

    X_test_conjoint = [X[0][0][test], X[0][1][test]]
    X_test_conjoint_struct = [X[1][0][test], X[1][1][test]]
    X_test_conjoint_cnn = [X[2][0][test], X[2][1][test]]
    X_test_conjoint_struct_cnn = [X[3][0][test], X[3][1][test]]
    X_test_conjoint_previous = [X[4][0][test], X[4][1][test]]
    # 위의 train set과 마찬가지.

    y_train_mono = y[train]
    #print('before categorical y_train_mono : {}'.format(y_train_mono))
    y_train = np_utils.to_categorical(y_train_mono, 2)
    #print('after- categoricaly_train : {}'.format(y_train))
    y_test_mono = y[test]
    y_test = np_utils.to_categorical(y_test_mono, 2)
    # np_utils.to_categorical : one-hot encoding시킴.
    # y_train_mono : 1차원. [1,1,1,1,1, ... ,0,0,0,0,0 ... ]
    # y_train : 2차원. [  [1,0], [1,0], [1,0], ... ,  [0,1], [0,1], [0,1], ... ]

    X_ensemble_train = X_train_conjoint + X_train_conjoint_struct + X_train_conjoint_cnn + X_train_conjoint_struct_cnn
    #print('X_train_conjoint : {0} - X_train_conjoint_len : {1}'.format(X_train_conjoint, len(X_train_conjoint)))
    #print('X_train_conjoint_struct : {0} - X_train_conjoint_struct_len : {1}'.format(X_train_conjoint_struct, len(X_train_conjoint)))
    #print('X_train_conjoint_cnn : {0} - X_train_conjoint_cnn_len : {1}'.format(X_train_conjoint_cnn, len(X_train_conjoint)))
    #print('X_train_conjoint_struct_cnn : {0} - X_train_conjoint_struct_cnn_len : {1}'.format(X_train_conjoint_struct_cnn, len(X_train_conjoint)))
    #print('X_ensemble_train : {0} - X_ensemble_train_len : {1}'.format(X_ensemble_train, len(X_train_conjoint)))
    
    #print('len of X_train_conjoint[0][0] : {}'.format(len(X_train_conjoint[0][0])))
    #print('len of X_train_conjoint_struct[0][0] : {}'.format(len(X_train_conjoint_struct[0][0])))
    X_ensemble_test = X_test_conjoint + X_test_conjoint_struct + X_test_conjoint_cnn + X_test_conjoint_struct_cnn
    

    print(str('\n# ' + '=' * 10 + " Fold {} " + "=" * 10 + '\n').format(fold))
    out.write(str('\n# ' + '=' * 10 + " Fold {} " + "=" * 10 + '\n').format(fold))
    model_metrics = {'RPITER': np.zeros(6),
                     'Conjoint-SAE': np.zeros(6), 'Conjoint-Struct-SAE': np.zeros(6),
                     'Conjoint-CNN': np.zeros(6), 'Conjoint-Struct-CNN': np.zeros(6),
                     }
    model_weight_path = result_save_path + 'weights.hdf5'
    # os.path.dirname(script_dir) + '/result/data/RPI488/... 결과는 result폴더에 있다!

    module_index = 0

    # =================================================================
    # Conjoint-CNN module

    stage = 'Conjoint-CNN'
    print("\n# Module Conjoint-CNN part #\n")

    # create model
    model_conjoint_cnn = conjoint_cnn(PRO_CODING_LENGTH, RNA_CODING_LENGTH, VECTOR_REPETITION_CNN)
    # PRO_CODING_LENGTH = sum_power(7, 1, WINDOW_P_UPLIMIT) => 399
    # RNA_CODING_LENGTH = sum_power(4, 1, WINDOW_R_UPLIMIT) => 340
    # VECTOR_REPETITION_CNN = 1
    
    callbacks = get_callback_list(PATIENCES[0], result_save_path, stage, fold, X_test_conjoint_cnn,
                                  y_test)
    # PATIENCES = [10, 10, 10, 10, 10]
    # result_save_path : ~/result/data/RPI488/...
    # stage = 'Conjoint-CNN'
    # fold = 현재 for 문으로 순회하고 있는 값. range(0, 5)
    # X_test_conjoint_cnn : 3차원. [각 pN_conjoint의 각 정규화값에 VECTOR_REPETITION_CNN이 곱해짐. ] 중 test의 인덱스만. len : 2
    # y_test : kind 값들 중 test값들만 뽑인 것 일듯.
    
    



    # first train
    model_conjoint_cnn.compile(loss='categorical_crossentropy', optimizer=get_optimizer(FIRST_OPTIMIZER), metrics=['accuracy'])
    # model.compile() 모델 학습과정 설정
    # FIRST_OPTIMIZER = 'adam', SECOND_OPTIMIZER = 'sgd'
    
    callbacks[0].close_plt_on_train_end = False
    # callbacks = [acchistory, earlystopping]
    # close_plt_on_train_end 이게 False이면, 계속해서 밑에 subplot을 그려나가겠다는 의미.
    
    model_conjoint_cnn.fit(x=X_train_conjoint_cnn,
                           y=y_train,
                           epochs=FIRST_TRAIN_EPOCHS[0],
                           batch_size=BATCH_SIZE,
                           verbose=VERBOSE,
                           shuffle=SHUFFLE,
                           callbacks=[callbacks[0]])
    # model.fit() 모델 학습시키기

    # second train
    model_conjoint_cnn.compile(loss='categorical_crossentropy', optimizer=get_optimizer(SECOND_OPTIMIZER), metrics=['accuracy'])
    # FIRST_OPTIMIZER = 'adam', SECOND_OPTIMIZER = 'sgd'
    
    callbacks[0].close_plt_on_train_end = True
    # close_plt_on_train_end 이게 True이면, 지금 그릴게 마지막 subplot이라는 의미.
    
    model_conjoint_cnn.fit(x=X_train_conjoint_cnn,
                           y=y_train,
                           epochs=SECOND_TRAIN_EPOCHS[0],
                           batch_size=BATCH_SIZE,
                           verbose=VERBOSE,
                           shuffle=SHUFFLE,
                           callbacks=callbacks)

    # test
    y_test_predict = model_conjoint_cnn.predict(X_test_conjoint_cnn)
    model_metrics['Conjoint-CNN'] = np.array(calc_metrics(y_test[:, 1], y_test_predict[:, 1]))
    # model_metrics['Conjoint-CNN'] : Acc, Sn, Sp, Pre, MCC, AUC
    
    print('Best performance for module Conjoint-CNN:\n {}\n'.format(model_metrics['Conjoint-CNN'].tolist()))
    
    
    # =================================================================


    # =================================================================
    # Conjoint-Struct-CNN module

    stage = "Conjoint-Struct-CNN"
    print("\n# Module Conjoint-Struct-CNN part #\n")
    module_index += 1

    # create model
    model_conjoint_struct_cnn = conjoint_cnn(PRO_STRUCT_CODING_LENGTH, RNA_STRUCT_CODING_LENGTH, VECTOR_REPETITION_CNN)
    # PRO_CODING_LENGTH = sum_power(7, 1, WINDOW_P_UPLIMIT) + sum_power(3, 1, WINDOW_P_STRUCT_UPLIMIT) => 438
    # RNA_CODING_LENGTH = (4, 1, WINDOW_R_UPLIMIT) + sum_power(2, 1, WINDOW_R_STRUCT_UPLIMIT) => 340 + 30 = 370
    # VECTOR_REPETITION_CNN = 1

    
    callbacks = get_callback_list(PATIENCES[1], result_save_path, stage, fold,
                                  X_test_conjoint_struct_cnn, y_test)

    # first train
    model_conjoint_struct_cnn.compile(loss='categorical_crossentropy', optimizer=get_optimizer(FIRST_OPTIMIZER), metrics=['accuracy'])
    callbacks[0].close_plt_on_train_end = False
    model_conjoint_struct_cnn.fit(x=X_train_conjoint_struct_cnn,
                                  y=y_train,
                                  epochs=FIRST_TRAIN_EPOCHS[1],
                                  batch_size=BATCH_SIZE,
                                  verbose=VERBOSE,
                                  shuffle=SHUFFLE,
                                  callbacks=[callbacks[0]])

    # second train
    model_conjoint_struct_cnn.compile(loss='categorical_crossentropy', optimizer=get_optimizer(SECOND_OPTIMIZER), metrics=['accuracy'])
    callbacks[0].close_plt_on_train_end = True
    model_conjoint_struct_cnn.fit(x=X_train_conjoint_struct_cnn,
                                  y=y_train,
                                  epochs=SECOND_TRAIN_EPOCHS[1],
                                  batch_size=BATCH_SIZE,
                                  verbose=VERBOSE,
                                  shuffle=SHUFFLE,
                                  callbacks=callbacks)

    # test
    y_test_predict = model_conjoint_struct_cnn.predict(X_test_conjoint_struct_cnn)
    model_metrics['Conjoint-Struct-CNN'] = np.array(calc_metrics(y_test[:, 1], y_test_predict[:, 1]))
    # model_metrics['Conjoint-Struct-CNN'] : Acc, Sn, Sp, Pre, MCC, AUC
    
    print(
        'Best performance for module Conjoint-Struct-CNN:\n {}\n'.format(model_metrics['Conjoint-Struct-CNN'].tolist()))
    # =================================================================


    # =================================================================
    # Conjoint-SAE module

    stage = 'Conjoint-SAE'
    print("\n# Module Conjoint-SAE part #\n")
    module_index += 1

    # create model
    encoders_pro, encoders_rna = get_auto_encoders(X_train_conjoint, X_test_conjoint)
    # encoders_pro : keras.layers.core.Dense 객체. [Dense(256), Dense(128), Dense(128)]. 399->256 처럼 인코딩시켜서 수를 줄이는 객체가 담긴듯.
    # encoders_rna : keras.layers.core.Dense 객체. [Dense(256), Dense(128), Dense(128)]. 399->256 처럼 인코딩시켜서 수를 줄이는 객체가 담긴듯.
    
    
    # PRO_CODING_LENGTH = sum_power(7, 1, WINDOW_P_UPLIMIT) + sum_power(3, 1, WINDOW_P_STRUCT_UPLIMIT) => 438
    # RNA_CODING_LENGTH = (4, 1, WINDOW_R_UPLIMIT) + sum_power(2, 1, WINDOW_R_STRUCT_UPLIMIT) => 340 + 30 = 370
    model_conjoint_sae = conjoint_sae(encoders_pro, encoders_rna, PRO_CODING_LENGTH, RNA_CODING_LENGTH)
    # model_conjoint = Model(inputs=[xp_in_conjoint, xr_in_conjoint], outputs=y_conjoint)
    
    callbacks = get_callback_list(PATIENCES[2], result_save_path, stage, fold, X_test_conjoint,
                                  y_test)
    # PATIENCES = [10, 10, 10, 10, 10]
    # result_save_path : ~/result/data/RPI488/...
    # stage = Conjoint-SAE
    # fold = 현재 for 문으로 순회하고 있는 값. range(0, 5)
    # X_test_conjoint_cnn : 3차원. [각 pN_conjoint의 각 정규화값에 VECTOR_REPETITION_CNN이 곱해짐. ] 중 test의 인덱스만. len : 2
    # y_test : kind 값들 중 test값들만 뽑인 것 일듯.
    

    # first train
    

    model_conjoint_sae.compile(loss='categorical_crossentropy', optimizer=get_optimizer(FIRST_OPTIMIZER), metrics=['accuracy'])
    callbacks[0].close_plt_on_train_end = False
    model_conjoint_sae.fit(x=X_train_conjoint,
                           y=y_train,
                           epochs=FIRST_TRAIN_EPOCHS[2],
                           batch_size=BATCH_SIZE,
                           verbose=VERBOSE,
                           shuffle=SHUFFLE,
                           callbacks=[callbacks[0]])

    # second train
    model_conjoint_sae.compile(loss='categorical_crossentropy', optimizer=get_optimizer(SECOND_OPTIMIZER), metrics=['accuracy'])
    callbacks[0].close_plt_on_train_end = True
    model_conjoint_sae.fit(x=X_train_conjoint,
                           y=y_train,
                           epochs=SECOND_TRAIN_EPOCHS[2],
                           batch_size=BATCH_SIZE,
                           verbose=VERBOSE,
                           shuffle=SHUFFLE,
                           callbacks=[callbacks[0]])

    # test
    y_test_predict = model_conjoint_sae.predict(X_test_conjoint)
    model_metrics['Conjoint-SAE'] = np.array(calc_metrics(y_test[:, 1], y_test_predict[:, 1]))
    print('Best performance for module Conjoint-SAE:\n {}\n'.format(model_metrics['Conjoint-SAE'].tolist()))
    # =================================================================


    # =================================================================
    # Conjoint-Struct-SAE module

    stage = 'Conjoint-Struct-SAE'
    print("\n# Module Conjoint-Struct-SAE part #\n")
    module_index += 1

    # create model
    encoders_pro, encoders_rna = get_auto_encoders(X_train_conjoint_struct, X_test_conjoint_struct)
    model_conjoint_struct_sae = conjoint_sae(encoders_pro, encoders_rna, PRO_STRUCT_CODING_LENGTH,
                                             RNA_STRUCT_CODING_LENGTH)
    callbacks = get_callback_list(PATIENCES[3], result_save_path, stage, fold,
                                  X_test_conjoint_struct,
                                  y_test)

    # first train
    model_conjoint_struct_sae.compile(loss='categorical_crossentropy', optimizer=get_optimizer(FIRST_OPTIMIZER), metrics=['accuracy'])
    callbacks[0].close_plt_on_train_end = False
    model_conjoint_struct_sae.fit(x=X_train_conjoint_struct,
                                  y=y_train,
                                  epochs=FIRST_TRAIN_EPOCHS[3],
                                  batch_size=BATCH_SIZE,
                                  verbose=VERBOSE,
                                  shuffle=SHUFFLE,
                                  callbacks=[callbacks[0]])

    # second train
    model_conjoint_struct_sae.compile(loss='categorical_crossentropy', optimizer=get_optimizer(SECOND_OPTIMIZER), metrics=['accuracy'])
    callbacks[0].close_plt_on_train_end = True
    model_conjoint_struct_sae.fit(x=X_train_conjoint_struct,
                                  y=y_train,
                                  epochs=SECOND_TRAIN_EPOCHS[3],
                                  batch_size=BATCH_SIZE,
                                  verbose=VERBOSE,
                                  shuffle=SHUFFLE,
                                  callbacks=callbacks)

    # test
    y_test_predict = model_conjoint_struct_sae.predict(X_test_conjoint_struct)
    model_metrics['Conjoint-Struct-SAE'] = np.array(calc_metrics(y_test[:, 1], y_test_predict[:, 1]))
    print(
        'Best performance for module Conjoint-Struct-SAE:\n {}\n'.format(model_metrics['Conjoint-Struct-SAE'].tolist()))
    # =================================================================



    # =================================================================
    # module ensemble

    stage = 'RPITER'
    print("\n# Module Ensemble part #\n")
    module_index += 1

    # create model
    ensemble_in = concatenate([model_conjoint_sae.output, model_conjoint_struct_sae.output,
                               model_conjoint_cnn.output, model_conjoint_struct_cnn.output])
    ensemble_in = Dropout(0.25)(ensemble_in)
    ensemble = Dense(16, kernel_initializer='random_uniform', activation='relu')(ensemble_in)
    ensemble = BatchNormalization()(ensemble)
    # ensemble = Dropout(0.2)(ensemble)
    ensemble = Dense(8, kernel_initializer='random_uniform', activation='relu')(ensemble)
    ensemble = BatchNormalization()(ensemble)
    # ensemble = Dropout(0.2)(ensemble)
    ensemble_out = Dense(2, activation='softmax')(ensemble)
    model_ensemble = Model(
        inputs=model_conjoint_sae.input + model_conjoint_struct_sae.input + model_conjoint_cnn.input + model_conjoint_struct_cnn.input,
        outputs=ensemble_out)

    callbacks = get_callback_list(PATIENCES[4], result_save_path, stage, fold, X_ensemble_test,
                                  y_test)

    # first train
    # freeze sub-models
    if FREEZE_SUB_MODELS: # default : True
        for model in [model_conjoint_sae, model_conjoint_struct_sae, model_conjoint_cnn, model_conjoint_struct_cnn]:
            control_model_trainable(model, False) 
            # ?? model.layer.trainable 인자를 False로 바꾸는 것임.이해못함.
            # 학습된 값들을 그대로 사용할 것이기 때문에 trainable의 속성을 False로 함.
            
    model_ensemble.compile(loss='categorical_crossentropy', optimizer=get_optimizer(Ff metrics=['accuracy'])
    callbacks[0].close_plt_on_train_end = False
    model_ensemble.fit(x=X_ensemble_train,
                       y=y_train,
                       epochs=FIRST_TRAIN_EPOCHS[4],
                       batch_size=BATCH_SIZE,
                       verbose=VERBOSE,
                       shuffle=SHUFFLE,
                       callbacks=[callbacks[0]])

    # second train
    # if FREEZE_SUB_MODELS:
    #     for model in [model_conjoint_sae, model_conjoint_struct_sae, model_conjoint_cnn, model_conjoint_struct_cnn]:
    #         control_model_trainable(model, True)
    model_ensemble.compile(loss='categorical_crossentropy', optimizer=get_optimizer(SECOND_OPTIMIZER), metrics=['accuracy'])
    callbacks[0].close_plt_on_train_end = True
    model_ensemble.fit(x=X_ensemble_train,
                       y=y_train,
                       epochs=SECOND_TRAIN_EPOCHS[4],
                       batch_size=BATCH_SIZE,
                       verbose=VERBOSE,
                       shuffle=SHUFFLE,
                       callbacks=callbacks)

    # test
    y_test_predict = model_ensemble.predict(X_ensemble_test)
    model_metrics['RPITER'] = np.array(calc_metrics(y_test[:, 1], y_test_predict[:, 1]))
    print('Best performance for module ensemble:\n {}\n'.format(model_metrics['RPITER'].tolist()))

    # =================================================================
    for key in model_metrics:
        print(key + " : " + str(model_metrics[key].tolist()) + "\n")
        out.write(key + " : " + str(model_metrics[key].tolist()) + "\n")
    for key in model_metrics:
        metrics_whole[key] += model_metrics[key]

    # get rid of the model weights file
    # if os.path.exists(model_weight_path):
    #     os.remove(model_weight_path)

print('\nMean metrics in {} fold:\n'.format(K_FOLD))
out.write('\nMean metrics in {} fold:\n'.format(K_FOLD))
# metrics_whole = {'RPITER': np.zeros(6),
#                 'Conjoint-SAE': np.zeros(6), 'Conjoint-Struct-SAE': np.zeros(6),
#                 'Conjoint-CNN': np.zeros(6), 'Conjoint-Struct-CNN': np.zeros(6)}

for key in metrics_whole.keys():
    metrics_whole[key] /= K_FOLD
    print(key + " : " + str(metrics_whole[key].tolist()) + "\n")
    out.write(key + " : " + str(metrics_whole[key].tolist()) + "\n")
out.flush()
out.close()
