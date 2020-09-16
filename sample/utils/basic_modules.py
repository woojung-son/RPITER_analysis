from keras.layers import Dense, Conv1D, concatenate, BatchNormalization, MaxPooling1D
from keras.layers import Dropout, Flatten, Input
from keras.models import Model


def conjoint_cnn(pro_coding_length, rna_coding_length, vector_repeatition_cnn):
    # pro_coding_length = sum_power(7, 1, WINDOW_P_UPLIMIT) => 399
    # rna_coding_length = sum_power(4, 1, WINDOW_R_UPLIMIT) => 340
    # vector_repeatition_cnn = 1

    if type(vector_repeatition_cnn)==int:
        # VECTOR_REPETITION_CNN가 배열인지, int형인지 스캔함.
        vec_len_p = vector_repeatition_cnn
        vec_len_r = vector_repeatition_cnn
    else:
        vec_len_p = vector_repeatition_cnn[0]
        vec_len_r = vector_repeatition_cnn[1]

    # NN for protein feature analysis by one hot encoding
    xp_in_conjoint_cnn = Input(shape=(pro_coding_length, vec_len_p)) # shape=(399, 1) 399개의 원소가 있는 1차원 배열을 담은 Input 인스턴스.
    xp_cnn = Conv1D(filters=45, kernel_size=6, strides=1, activation='relu')(xp_in_conjoint_cnn) 
    # Conv1D는 feature들의 길이가 6이고, 그것이 한칸씩 오른쪽으로 움직이며, output으로 45개의 sequence(=배열)을 생성한다는 말인듯?
    # xp_cnn.shape : (None, 394, 45)
    #print('xp_cnn.shape : {}'.format(xp_cnn.shape))
    
    xp_cnn = MaxPooling1D(pool_size=2)(xp_cnn)
    # Conv1D에서 나온 특징벡터들을 맥스풀링(MaxPooling1D)를 통해 1/2로 줄여준다.
    # xp_cnn.shape : (None, 197, 45)
    #print('xp_cnn.shape : {}'.format(xp_cnn.shape))
    
    xp_cnn = BatchNormalization()(xp_cnn)
    #  입력값을 평균 0, 분산 1로 정규화하는 과정.
    
    xp_cnn = Dropout(0.2)(xp_cnn)
    # Dropout 하는 과정.
    #print('xp_cnn.shape : {}'.format(xp_cnn.shape))
    
    xp_cnn = Conv1D(filters=64, kernel_size=5, strides=1, activation='relu')(xp_cnn)
    #print('after Conv1D - xp_cnn.shape : {}'.format(xp_cnn.shape))
    xp_cnn = MaxPooling1D(pool_size=2)(xp_cnn)
    #print('after MaxPooling1D - xp_cnn.shape : {}'.format(xp_cnn.shape))
    xp_cnn = BatchNormalization()(xp_cnn)
    #print('after BatchNormalization - xp_cnn.shape : {}'.format(xp_cnn.shape))
    xp_cnn = Dropout(0.2)(xp_cnn)
    #print('after Dropout - xp_cnn.shape : {}'.format(xp_cnn.shape))
    xp_cnn = Conv1D(filters=45, kernel_size=4, strides=1, activation='relu')(xp_cnn)
    xp_cnn = BatchNormalization()(xp_cnn)
    xp_cnn = Dropout(0.2)(xp_cnn)
    # xp_cnn = Bidirectional(LSTM(32,return_sequences=True))(xp_cnn)
    # xp_cnn = LSTM(32,return_sequences=True)(xp_cnn)
    xp_cnn = Flatten()(xp_cnn)
    xp_out_conjoint_cnn = Dense(64)(xp_cnn)
    # Dense 레이어는 입력과 출력을 모두 연결해주는 과정. 예를 들어 입력 뉴런이 4개, 출력 뉴런이 8개있다면 총 연결선은 32개
    # Dense() 안에 숫자는 출력 뉴런 수. 현재는 shape=(None, 64)이므로 64가 들어감
    
    xp_out_conjoint_cnn = Dropout(0.2)(xp_out_conjoint_cnn)
    # xp_out_conjoint_cnn.shape : (None, 64)
    #print('xp_out_conjoint_cnn.shape : {}'.format(xp_out_conjoint_cnn.shape))

    # NN for RNA feature analysis  by one hot encoding
    xr_in_conjoint_cnn = Input(shape=(rna_coding_length, vec_len_r))
    xr_cnn = Conv1D(filters=45, kernel_size=6, strides=1, activation='relu')(xr_in_conjoint_cnn)
    xr_cnn = MaxPooling1D(pool_size=2)(xr_cnn)
    xr_cnn = BatchNormalization()(xr_cnn)
    xr_cnn = Dropout(0.2)(xr_cnn)
    xr_cnn = Conv1D(filters=64, kernel_size=5, strides=1, activation='relu')(xr_cnn)
    xr_cnn = MaxPooling1D(pool_size=2)(xr_cnn)
    xr_cnn = BatchNormalization()(xr_cnn)
    xr_cnn = Dropout(0.2)(xr_cnn)
    xr_cnn = Conv1D(filters=45, kernel_size=4, strides=1, activation='relu')(xr_cnn)
    xr_cnn = BatchNormalization()(xr_cnn)
    xr_cnn = Dropout(0.2)(xr_cnn)
    # xr_cnn = Bidirectional(LSTM(32,return_sequences=True))(xr_cnn)
    # xr_cnn = LSTM(32,return_sequences=True)(xr_cnn)
    xr_cnn = Flatten()(xr_cnn)
    xr_out_conjoint_cnn = Dense(64)(xr_cnn)
    xr_out_conjoint_cnn = Dropout(0.2)(xr_out_conjoint_cnn)
    # xr_out_conjoint_cnn.shape : (None, 64)
    #print('xr_out_conjoint_cnn.shape : {}'.format(xr_out_conjoint_cnn.shape))

    x_out_conjoint_cnn = concatenate([xp_out_conjoint_cnn, xr_out_conjoint_cnn])
    # xp_out_conjoint_cnn와 xr_out_conjoint_cnn를 concatenate
    
    x_out_conjoint_cnn = Dense(128, kernel_initializer='random_uniform', activation='relu')(x_out_conjoint_cnn)
    # x_out_conjoint_cnn = Dropout(0.25)(x_out_conjoint_cnn)
    x_out_conjoint_cnn = BatchNormalization()(x_out_conjoint_cnn)
    # x_out_conjoint_cnn = Dropout(0.3)(x_out_conjoint_cnn)
    x_out_conjoint_cnn = Dense(64, kernel_initializer='random_uniform', activation='relu')(x_out_conjoint_cnn)
    y_conjoint_cnn = Dense(2, activation='softmax')(x_out_conjoint_cnn)
    # y_conjoint_cnn.shape : (None, 2)
    #print('y_conjoint_cnn.shape : {}'.format(y_conjoint_cnn.shape))


    model_conjoint_cnn = Model(inputs=[xp_in_conjoint_cnn, xr_in_conjoint_cnn], outputs=y_conjoint_cnn)



    return model_conjoint_cnn


def conjoint_sae(encoders_protein, encoders_rna, pro_coding_length, rna_coding_length):
    # encoders_protein : keras.layers.core.Dense 객체. [Dense(256), Dense(128), Dense(128)]. 399->256 처럼 인코딩시켜서 수를 줄이는 객체가 담긴듯.
    # encoders_rna : keras.layers.core.Dense 객체. [Dense(256), Dense(128), Dense(128)]. 399->256 처럼 인코딩시켜서 수를 줄이는 객체가 담긴듯.
    # pro_coding_length = sum_power(7, 1, WINDOW_P_UPLIMIT) + sum_power(3, 1, WINDOW_P_STRUCT_UPLIMIT) => 438
    # rna_coding_length = (4, 1, WINDOW_R_UPLIMIT) + sum_power(2, 1, WINDOW_R_STRUCT_UPLIMIT) => 340 + 30 = 370
    

    # NN for protein feature analysis
    xp_in_conjoint = Input(shape=(pro_coding_length,)) # 438
    xp_encoded = encoders_protein[0](xp_in_conjoint) # Dense(256)
    xp_encoded = Dropout(0.2)(xp_encoded)
    xp_encoded = encoders_protein[1](xp_encoded) # Dense(128)
    xp_encoded = Dropout(0.2)(xp_encoded)
    xp_encoder = encoders_protein[2](xp_encoded) # Dense(64)
    xp_encoder = Dropout(0.2)(xp_encoder)
    xp_encoder = BatchNormalization()(xp_encoder) # 평균 0, 분산 1로 정규화
    # xp_encoder = PReLU()(xp_encoder)
    xp_encoder = Dropout(0.2)(xp_encoder)

    # NN for RNA feature analysis
    xr_in_conjoint = Input(shape=(rna_coding_length,)) # 370
    xr_encoded = encoders_rna[0](xr_in_conjoint) # Dense(256)
    xr_encoded = Dropout(0.2)(xr_encoded)
    xr_encoded = encoders_rna[1](xr_encoded) # Dense(128)
    xr_encoded = Dropout(0.2)(xr_encoded)
    xr_encoded = encoders_rna[2](xr_encoded) # Dense(64)
    xr_encoder = Dropout(0.2)(xr_encoded)
    xr_encoder = BatchNormalization()(xr_encoder) # 평균 0, 분산 1로 정규화
    # xr_encoder = PReLU()(xr_encoder)
    xr_encoder = Dropout(0.2)(xr_encoder)

    x_out_conjoint = concatenate([xp_encoder, xr_encoder]) # xp_encoder, xr_encoder는 둘다 2차원. xp_encoder[0] len = 64. 두개 합치면 128
    x_out_conjoint = Dense(128, kernel_initializer='random_uniform', activation='relu')(x_out_conjoint)
    x_out_conjoint = BatchNormalization()(x_out_conjoint)
    x_out_conjoint = Dense(64, kernel_initializer='random_uniform', activation='relu')(x_out_conjoint)
    y_conjoint = Dense(2, activation='softmax')(x_out_conjoint)

    model_conjoint = Model(inputs=[xp_in_conjoint, xr_in_conjoint], outputs=y_conjoint)


    return model_conjoint