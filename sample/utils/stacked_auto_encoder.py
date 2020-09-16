import numpy as np
from keras.layers import Dense, Dropout
from keras.models import Sequential


def train_auto_encoder(X_train, X_test, layers, batch_size=100, nb_epoch=100, activation='sigmoid', optimizer='adam'):
    # X_train : 3차원. [p1_conjoint, p2_conjoint, p3_conjoint ... ] 중 train의 인덱스만 선택, RNA의 경우도 마찬가지. len : 2
    # X_test : 3차원. [p1_conjoint_struct, p2_conjoint_struct, ... ] 중 train의 인덱스만 선택, RNA의 경우도 마찬가지. len : 2
    # layers = [399, 256, 128, 64]
    # BATCH_SIZE = 150
    
    trained_encoders = []
    trained_decoders = []
    X_train_tmp = np.copy(X_train)
    X_test_tmp = np.copy(X_test)
    for n_in, n_out in zip(layers[:-1], layers[1:]): # zip([399, 256, 128], [256, 128, 64])
        # zip = 김밥자르기
        # (399, 256), (256, 128), (128, 64)
        
        print('Pre-training the layer: Input {} -> Output {}'.format(n_in, n_out))
        ae = Sequential(
            [Dense(n_out, input_dim=X_train_tmp.shape[1], activation=activation, ),
             Dense(n_in, activation=activation),
             Dropout(0.2)]
        )
        # Dense 레이어는 입력과 출력을 모두 연결해주는 과정. 예를 들어 입력 뉴런이 4개, 출력 뉴런이 8개있다면 총 연결선은 32개
        # Dense() 안에 숫자는 출력 뉴런 수. 현재는 n_out, n_in이 들어감.
        # Dense() 안의 input_dim은 입력차원. =1이면 입력 노드가 한개라고 생각. 배열의 데이터가 2개라면 2, 3이라면 3.
        #print('input_dim=X_train_tmp.shape[1] : {}'.format(X_train_tmp.shape[1]))
        
        ae.compile(loss='mean_squared_error', optimizer=optimizer)
        # loss : 손실함수. 얼마나 입력데이터가 출력데이터와 일치해주는지 평가해주는 함수. mean squared error : 평균제곱오차
        # optimizer : 손실함수를 기반으로 네트워크가 어떻게 업데이트될지 결정.
        
        ae.fit(X_train_tmp, X_train_tmp, batch_size=batch_size, epochs=nb_epoch, verbose=0, shuffle=True)
        # (중요) Auto Encoder임으로 (X, y) = (X_train_tmp, X_train_tmp) 이다.
        
        # store trained encoder
        trained_encoders.append(ae.layers[0])
        trained_decoders.append(ae.layers[1])
        #print('ae.layers[2] : {}'.format(ae.layers[2]))
        # ae.layers[0] : Dense(n_out, input_dim=X_train_tmp.shape[1], activation=activation, )
        # ae.layers[1] : Dense(n_in, activation=activation)
        print('X_train_tmp : {0} - len : {1} - len X_train_tmp[0] : {2}'.format(X_train_tmp, len(X_train_tmp), len(X_train_tmp[0])))
        

        
        # update training data
        encoder = Sequential([ae.layers[0]]) 
        # (399), (256), (128) 각각에 대해 "Dense(n_out, input_dim=X_train_tmp.shape[1], activation=activation, )"에 대한 Sequential 객체를 만듬.
        
        # encoder.evaluate(X_train_tmp, X_train_tmp, batch_size=batch_size) -> batch_size 크기만큼 batch를 잡아 손실 함수를 계산.
        X_train_tmp = encoder.predict(X_train_tmp) 
        X_test_tmp = encoder.predict(X_test_tmp)
        # n_out=(399) Dense(399) -> Dense(256) 으로 바꾸고, X_train_tmp을 Dense(399)의 출력물로 바꿈.
        # n_out=(256) Dense(256) -> Dense(128) 으로 바꾸고, X_train_tmp을 Dense(256)의 출력물로 바꿈.
        # n_out=(128) Dense(128) -> Dense(128) 으로 바꾸고,  X_train_tmp을 Dense(399)의 출력물로 바꿈.
        

    print('X_train_tmp : {0} - len : {1} - len X_train_tmp[0] : {2}'.format(X_train_tmp, len(X_train_tmp), len(X_train_tmp[0])))
    
    # train_encoders : keras.layers.core.Dense 객체. [Dense(256), Dense(128), Dense(128)]. 399->256 처럼 인코딩시켜서 수를 줄이는 객체가 담긴듯.
    # trained_decoders : keras.layers.core.Dense 객체. [Dense(399), Dense(256), Dense(128)]. 256->399 처럼 디코딩시켜서 수를 늘리는 객체가 담긴듯.
    # X_train_tmp : 2차원. [p1_conjoint, p2_conjoint, p3_conjoint ... ], 각각의 pN_conjoint는 64개의 원소로 압축됨.
    # X_test_tmp : 
    return trained_encoders, trained_decoders, X_train_tmp, X_test_tmp
