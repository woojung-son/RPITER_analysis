import math
import string
from functools import reduce

import numpy as np


# encoder for protein sequence
class ProEncoder:
    elements = 'AIYHRDC'
    structs = 'hec'

    element_number = 7
    # number of structure kind
    struct_kind = 3

    # clusters: {A,G,V}, {I,L,F,P}, {Y,M,T,S}, {H,N,Q,W}, {R,K}, {D,E}, {C}
    pro_intab = 'AGVILFPYMTSHNQWRKDEC'
    pro_outtab = 'AAAIIIIYYYYHHHHRRDDC'

    def __init__(self, WINDOW_P_UPLIMIT, WINDOW_P_STRUCT_UPLIMIT, CODING_FREQUENCY, VECTOR_REPETITION_CNN,
                 TRUNCATION_LEN=None, PERIOD_EXTENDED=None):
        #WINDOW_P_UPLIMIT : protein feature를 최대 몇자리까지 쓸것인가를 저장한 상수. 3
        #WINDOW_P_STRUCT_UPLIMIT : struct 정보의 protein feature를 최대 몇자리까지 쓸것인가를 저장한 상수. 3
        #CODING_FREQUENCY : 전역상수. True.
        #VECTOR_REPETITION_CNN : 전역상수. 1.
        
        self.WINDOW_P_UPLIMIT = WINDOW_P_UPLIMIT
        self.WINDOW_P_STRUCT_UPLIMIT = WINDOW_P_STRUCT_UPLIMIT
        self.CODING_FREQUENCY = CODING_FREQUENCY
        self.VECTOR_REPETITION_CNN = VECTOR_REPETITION_CNN

        self.TRUNCATION_LEN = TRUNCATION_LEN
        self.PERIOD_EXTENDED = PERIOD_EXTENDED

        # list and position map for k_mer
        k_mers = ['']
        self.k_mer_list = []
        self.k_mer_map = {}
        for T in range(self.WINDOW_P_UPLIMIT): # 3
            temp_list = []
            for k_mer in k_mers:
                for x in self.elements:# AIYHRDC 
                    temp_list.append(k_mer + x)
            k_mers = temp_list
            self.k_mer_list += temp_list
        for i in range(len(self.k_mer_list)):
            self.k_mer_map[self.k_mer_list[i]] = i

        # list and position map for k_mer structure
        k_mers = ['']
        self.k_mer_struct_list = []
        self.k_mer_struct_map = {}
        for T in range(self.WINDOW_P_STRUCT_UPLIMIT):
            temp_list = []
            for k_mer in k_mers: 
                for s in self.structs:
                    temp_list.append(k_mer + s)
            k_mers = temp_list
            self.k_mer_struct_list += temp_list
        for i in range(len(self.k_mer_struct_list)):
            self.k_mer_struct_map[self.k_mer_struct_list[i]] = i

        # table for amino acid clusters
        self.transtable = str.maketrans(self.pro_intab, self.pro_outtab)
        
        #k_mer_map : feature들을 key값으로, 그것의 index를 value값으로 가지는 딕셔너리
        # k_mer_map 0~6 : 한 자리 알파벳으로 이루어진 원소
        # k_mer_map 7~55 : 두 자리 알파벳으로 이루어진 원소
        # k_mer_map 56~398 : 세 자리 알파벳으로 이루어진 원소
        
        #k_mer_list : feature들이 sorting되어 있는 리스트

        # print(len(self.k_mer_list))
        # print(self.k_mer_list)
        #print('self.k_mer_map : {}'.format(self.k_mer_map))
        # print(len(self.k_mer_struct_list))
        # print(self.k_mer_struct_list)


    def encode_conjoint_previous(self, seq):
        seq = seq.translate(self.transtable)
        seq = ''.join([x for x in seq if x in self.elements])
        seq_len = len(seq)
        if seq_len == 0:
            return 'Error'
        result = []
        K = self.WINDOW_P_UPLIMIT
        offset = reduce(lambda x, y: x + y, map(lambda x: self.element_number ** x, range(1, K)))
        vec = [0.0] * (self.element_number ** K)
        counter = seq_len - K + 1
        for i in range(seq_len - K + 1):
            k_mer = seq[i:i + K]
            vec[self.k_mer_map[k_mer] - offset] += 1
        vec = np.array(vec)
        if self.CODING_FREQUENCY:
            vec = vec / vec.max()
        result += list(vec)
        return np.array(result)

    def encode_conjoint(self, seq): 
        # sequence에서 각각의 feature들이 포함되는 횟수를 세서 정규화시킴. improved CTF. 우리 프로젝트랑 똑같음.
        # 정규화시키는 방법이 다름. min_max 정규화가 아니고, value를 최대값으로 나눔.
        
        seq = seq.translate(self.transtable) # seq는 문자열 # 'AGVILFPYMTSHNQWRKDEC' -> 'AAAIIIIYYYYHHHHRRDDC' 이렇게 바꿈.
        #print('seq before join : {}'.format(seq)) # 이건 AIYHRDC로만 이루어졌나 아닌가 체크하는 로직인 듯
        #seq = ''.join([x for x in seq if x in self.elements]) # seq는 문자열.
        #print('seq after_ join : {}'.format(seq))
        seq_len = len(seq)
        if seq_len == 0:
            return 'Error'
        result = []
        offset = 0
        for K in range(1, self.WINDOW_P_UPLIMIT + 1): # range(1, 4)
            # K는 feature의 길이임. 
            
            vec = [0.0] * (self.element_number ** K) 
            # vec배열을 7**K 개의 0.0 (float)가 담긴 배열로 초기화
            # element_number : 7
            
            counter = seq_len - K + 1
            for i in range(seq_len - K + 1): # K=1일 때는 sequence의 length만큼 순회. K=2일 때는 sequence의 length - 1만큼 순회. K=일때는 ...
                k_mer = seq[i:i + K] # feature를 순회하면서 K 길이의 문자열을 추출한거.
                vec[self.k_mer_map[k_mer] - offset] += 1 # vec 리스트에서 k_mer의 인덱스에 해당하는 자리에 카운트를 1 올림.
            vec = np.array(vec)
            offset += vec.size # K=1 일 때 vec.size = 7, K=2일 때 vec.size = 49, K=3일 때 vec.size = 343
            #print('self.k_mer_map[k_mer] : {}'.format(self.k_mer_map[k_mer]))
            #print('vec : {0} - vec.size : {1}'.format(vec, vec.size))
            if self.CODING_FREQUENCY:
                vec = vec / vec.max()
            result += list(vec)
            #print('len of result : {}'.format(len(result)))
        #print('result : {}'.format(result))
        
        # result 0~6 : 한 자리 알파벳으로 이루어진 원소
        # result 7~55 : 두 자리 알파벳으로 이루어진 원소
        # result 56~398 : 세 자리 알파벳으로 이루어진 원소
        return np.array(result)

    def encode_conjoint_struct(self, seq, struct):
        # seq length와 struct length는 같음. 헐.

        
        seq = seq.translate(self.transtable) # seq는 문자열 # 'AGVILFPYMTSHNQWRKDEC' -> 'AAAIIIIYYYYHHHHRRDDC' 이렇게 바꿈.
        seq_temp = []
        struct_temp = []
        for i in range(len(seq)):     
            if seq[i] in self.elements:
                # AIYHRDC 의 원소가 AIYHRDC안에 있으면, 0~len(seq)-1 의 모든 인덱스에 대해 translate된 seq[i]와 원본 struct[i]를 배열로 보관함.
                
                seq_temp.append(seq[i])
                struct_temp.append(struct[i])
        seq = ''.join(seq_temp) # 여기의 seq는 translate된 seq와 같음. (그냥 검증로직인듯)
        struct = ''.join(struct_temp) # 그냥 원본 struct와 같음.
        seq_len = len(seq)
        if seq_len == 0:
            return 'Error'

        
        # encode_conjoint의 sequence 인코딩 방식과 정확하게 동일함.
        result_seq = []
        offset_seq = 0
        for K in range(1, self.WINDOW_P_UPLIMIT + 1):
            vec_seq = [0.0] * (self.element_number ** K)
            counter = seq_len - K + 1
            for i in range(seq_len - K + 1):
                k_mer = seq[i:i + K]
                vec_seq[self.k_mer_map[k_mer] - offset_seq] += 1 # vec 리스트에서 k_mer의 인덱스에 해당하는 자리에 카운트를 1 올림.
            vec_seq = np.array(vec_seq)
            offset_seq += vec_seq.size
            if self.CODING_FREQUENCY:
                vec_seq = vec_seq / vec_seq.max()
            result_seq += list(vec_seq)


        result_struct = []
        offset_struct = 0
        for K in range(1, self.WINDOW_P_STRUCT_UPLIMIT + 1):
            vec_struct = [0.0] * (self.struct_kind ** K)
            # vec배열을 3^K 개의 0.0 (float)가 담긴 배열로 초기화
            # element_number : 3
            
            counter = seq_len - K + 1
            for i in range(seq_len - K + 1):
                k_mer_struct = struct[i:i + K]
                vec_struct[self.k_mer_struct_map[k_mer_struct] - offset_struct] += 1
            vec_struct = np.array(vec_struct)
            offset_struct += vec_struct.size
            if self.CODING_FREQUENCY:
                vec_struct = vec_struct / vec_struct.max()
            result_struct += list(vec_struct)
            
        # sequence를 정규화한 배열과 struct를 정규화한 배열을 concatenate시킴.
        # result_seq len : 399
        # result_struct len : 39 -> 3 + 9 + 27. feature의 알파벳 개수가 3개여서 그럼.
        # 결과 : 438 
        return np.array(result_seq + result_struct)

    def encode_conjoint_cnn(self, seq):
        result_t = self.encode_conjoint(seq)
        result = np.array([[x] * self.VECTOR_REPETITION_CNN for x in result_t])
        return result

    def encode_conjoint_struct_cnn(self, seq, struct):
        result_t = self.encode_conjoint_struct(seq, struct)
        result = np.array([[x] * self.VECTOR_REPETITION_CNN for x in result_t])
        return result

    def encode_onehot(self, seq):
        seq = seq.translate(self.transtable)
        seq = ''.join([x for x in seq if x in self.elements])
        seq_len = len(seq)
        if seq_len == 0:
            return 'Error'
        coding_len = min(len(seq), self.TRUNCATION_LEN)
        vec = [0] * coding_len
        for i in range(coding_len):
            pos = self.elements.index(seq[i])
            vec[i] = [0] * self.element_number
            vec[i][pos] = 1
        if coding_len < self.TRUNCATION_LEN:
            gap_len = (self.TRUNCATION_LEN - coding_len)
            if self.PERIOD_EXTENDED:
                gap_term = int(math.ceil(float(gap_len) / coding_len))
                gap_data = vec * gap_term
                vec += gap_data[0:gap_len]
            else:
                vec += [[0] * self.element_number] * gap_len
        return np.array(vec)

    def encode_onehot_struct(self, seq, struct):
        seq = seq.translate(self.transtable)
        seq_temp = []
        struct_temp = []
        for i in range(len(seq)):
            if seq[i] in self.elements:
                seq_temp.append(seq[i])
                struct_temp.append(struct[i])
        seq = ''.join(seq_temp)
        struct = ''.join(struct_temp)
        seq_len = len(seq)
        if seq_len == 0:
            return 'Error'
        coding_len = min(len(seq), self.TRUNCATION_LEN)
        vec = [0] * coding_len
        for i in range(coding_len):
            pos = self.elements.index(seq[i])
            pos = pos + self.element_number * self.structs.index(struct[i])
            vec[i] = [0] * (self.element_number * self.struct_kind)
            vec[i][pos] = 1
        if coding_len < self.TRUNCATION_LEN:
            gap_len = (self.TRUNCATION_LEN - coding_len)
            if self.PERIOD_EXTENDED:
                gap_term = int(math.ceil(float(gap_len) / coding_len))
                gap_data = vec * gap_term
                vec += gap_data[0:gap_len]
            else:
                vec += [[0] * self.element_number * self.struct_kind] * gap_len
        return np.array(vec)

    def encode_word2vec(self, seq, pro_word2vec, window_size, stride):
        seq = seq.translate(self.transtable)
        seq_len = len(seq)
        if seq_len == 0:
            return 'Error'
        words = pro_word2vec.keys()
        VEC_LEN_W2V = len(list(words)[0])
        MAX_PRO_W2V_LEN = self.TRUNCATION_LEN // stride
        vec = []
        p = 0
        while (len(vec) < MAX_PRO_W2V_LEN) and (p + window_size <= seq_len):
            word = seq[p:p + window_size]
            if word in words:
                vec.append(pro_word2vec[word])
            p += stride
        encoded_len = len(vec)
        if encoded_len == 0:
            return 'Error'
        elif encoded_len < MAX_PRO_W2V_LEN:
            gap_len = (MAX_PRO_W2V_LEN - encoded_len)
            if self.PERIOD_EXTENDED:
                gap_term = int(math.ceil(float(gap_len) / encoded_len))
                gap_data = vec * gap_term
                vec += gap_data[0:gap_len]
            else:
                vec += [[0.0] * VEC_LEN_W2V] * gap_len
        return np.array(vec)

    def encode_word2vec_struct(self, seq, struct, pro_word2vec, window_size, stride):
        seq = seq.translate(self.transtable)
        seq_temp = []
        struct_temp = []
        for i in range(len(seq)):
            if seq[i] in self.elements:
                seq_temp.append(seq[i])
                struct_temp.append(struct[i])
        seq = ''.join(seq_temp)
        struct = ''.join(struct_temp)
        seq_len = len(seq)
        if seq_len == 0:
            return 'Error'

        words = pro_word2vec.keys()
        VEC_LEN_W2V = len(list(words)[0])
        MAX_PRO_W2V_LEN = self.TRUNCATION_LEN // stride
        vec = []
        p = 0
        while (len(vec) < MAX_PRO_W2V_LEN) and (p + window_size <= seq_len):
            word = seq[p:p + window_size] + struct[p:p + window_size]
            if word in words:
                vec.append(pro_word2vec[word])
            p += stride
        encoded_len = len(vec)
        if encoded_len == 0:
            return 'Error'
        elif encoded_len < MAX_PRO_W2V_LEN:
            gap_len = (MAX_PRO_W2V_LEN - encoded_len)
            if self.PERIOD_EXTENDED:
                gap_term = int(math.ceil(float(gap_len) / encoded_len))
                gap_data = vec * gap_term
                vec += gap_data[0:gap_len]
            else:
                vec += [[0.0] * VEC_LEN_W2V] * gap_len
        return np.array(vec)


# encoder for RNA sequence
class RNAEncoder:
    elements = 'AUCG'
    structs = '.('

    element_number = 4
    struct_kind = 2

    def __init__(self, WINDOW_R_UPLIMIT, WINDOW_R_STRUCT_UPLIMIT, CODING_FREQUENCY, VECTOR_REPETITION_CNN,
                 TRUNCATION_LEN=None, PERIOD_EXTENDED=None):

        self.WINDOW_R_UPLIMIT = WINDOW_R_UPLIMIT
        self.WINDOW_R_STRUCT_UPLIMIT = WINDOW_R_STRUCT_UPLIMIT
        self.CODING_FREQUENCY = CODING_FREQUENCY
        self.VECTOR_REPETITION_CNN = VECTOR_REPETITION_CNN

        self.TRUNCATION_LEN = TRUNCATION_LEN
        self.PERIOD_EXTENDED = PERIOD_EXTENDED

        # list and position map for k_mer
        k_mers = ['']
        self.k_mer_list = []
        self.k_mer_map = {}
        for T in range(self.WINDOW_R_UPLIMIT):
            temp_list = []
            for k_mer in k_mers:
                for x in self.elements:
                    temp_list.append(k_mer + x)
            k_mers = temp_list
            self.k_mer_list += temp_list
        for i in range(len(self.k_mer_list)):
            self.k_mer_map[self.k_mer_list[i]] = i

        # list and position map for k_mer structure
        k_mers = ['']
        self.k_mer_struct_list = []
        self.k_mer_struct_map = {}
        for T in range(self.WINDOW_R_STRUCT_UPLIMIT):
            temp_list = []
            for k_mer in k_mers:
                for s in self.structs:
                    temp_list.append(k_mer + s)
            k_mers = temp_list
            self.k_mer_struct_list += temp_list
        for i in range(len(self.k_mer_struct_list)):
            self.k_mer_struct_map[self.k_mer_struct_list[i]] = i

        # print(len(self.k_mer_list))
        # print(self.k_mer_list)
        # print(len(self.k_mer_struct_list))
        # print(self.k_mer_struct_list)

    def encode_conjoint_previous(self, seq):
        seq = seq.replace('T', 'U')
        seq = ''.join([x for x in seq if x in self.elements])
        seq_len = len(seq)
        if seq_len == 0:
            return 'Error'
        result = []
        K = self.WINDOW_R_UPLIMIT
        offset = reduce(lambda x, y: x + y, map(lambda x: self.element_number ** x, range(1, K)))
        vec = [0.0] * (self.element_number ** K)
        counter = seq_len - K + 1
        for i in range(seq_len - K + 1):
            k_mer = seq[i:i + K]
            vec[self.k_mer_map[k_mer] - offset] += 1
        vec = np.array(vec)
        if self.CODING_FREQUENCY:
            vec = vec / vec.max()
        result += list(vec)
        return np.array(result)

    def encode_conjoint(self, seq):
        seq = seq.replace('T', 'U')
        seq = ''.join([x for x in seq if x in self.elements])
        seq_len = len(seq)
        if seq_len == 0:
            return 'Error'
        result = []
        offset = 0
        for K in range(1, self.WINDOW_R_UPLIMIT + 1):
            vec = [0.0] * (self.element_number ** K)
            counter = seq_len - K + 1
            for i in range(seq_len - K + 1):
                k_mer = seq[i:i + K]
                vec[self.k_mer_map[k_mer] - offset] += 1
            vec = np.array(vec)
            offset += vec.size
            if self.CODING_FREQUENCY:
                vec = vec / vec.max()
            result += list(vec)
        return np.array(result)

    def encode_conjoint_struct(self, seq, struct):
        seq = seq.replace('T', 'U')
        struct = struct.replace(')', '(')
        seq_temp = []
        struct_temp = []
        for i in range(len(seq)):
            if seq[i] in self.elements:
                seq_temp.append(seq[i])
                struct_temp.append(struct[i])
        seq = ''.join(seq_temp)
        struct = ''.join(struct_temp)
        seq_len = len(seq)
        if seq_len == 0:
            return 'Error'

        result_seq = []
        offset_seq = 0
        for K in range(1, self.WINDOW_R_UPLIMIT + 1):
            vec_seq = [0.0] * (self.element_number ** K)
            counter = seq_len - K + 1
            for i in range(seq_len - K + 1):
                k_mer = seq[i:i + K]
                vec_seq[self.k_mer_map[k_mer] - offset_seq] += 1
            vec_seq = np.array(vec_seq)
            offset_seq += vec_seq.size
            if self.CODING_FREQUENCY:
                vec_seq = vec_seq / vec_seq.max()
            result_seq += list(vec_seq)


        result_struct = []
        offset_struct = 0
        for K in range(1, self.WINDOW_R_STRUCT_UPLIMIT + 1):
            vec_struct = [0.0] * (self.struct_kind ** K)
            counter = seq_len - K + 1
            for i in range(seq_len - K + 1):
                k_mer_struct = struct[i:i + K]
                vec_struct[self.k_mer_struct_map[k_mer_struct] - offset_struct] += 1
            vec_struct = np.array(vec_struct)
            offset_struct += vec_struct.size
            if self.CODING_FREQUENCY:
                vec_struct = vec_struct / vec_struct.max()
            result_struct += list(vec_struct)
        return np.array(result_seq + result_struct)

    def encode_conjoint_cnn(self, seq):
        result_t = self.encode_conjoint(seq)
        result = np.array([[x] * self.VECTOR_REPETITION_CNN for x in result_t])
        return result

    def encode_conjoint_struct_cnn(self, seq, struct):
        result_t = self.encode_conjoint_struct(seq, struct)
        result = np.array([[x] * self.VECTOR_REPETITION_CNN for x in result_t])
        return result

    def encode_onehot(self, seq):
        seq = seq.replace('T', 'U')
        seq = ''.join([x for x in seq if x in self.elements])
        seq_len = len(seq)
        if seq_len == 0:
            return 'Error'
        coding_len = min(len(seq), self.TRUNCATION_LEN)
        vec = [0] * coding_len
        for i in range(coding_len):
            pos = self.elements.index(seq[i])
            vec[i] = [0] * self.element_number
            vec[i][pos] = 1
        if coding_len < self.TRUNCATION_LEN:
            gap_len = (self.TRUNCATION_LEN - coding_len)
            if self.PERIOD_EXTENDED:
                gap_term = int(math.ceil(float(gap_len) / coding_len))
                gap_data = vec * gap_term
                vec += gap_data[0:gap_len]
            else:
                vec += [[0] * self.element_number] * gap_len
        return np.array(vec)

    def encode_onehot_struct(self, seq, struct):
        seq = seq.replace('T', 'U')
        struct = struct.replace(')', '(')
        seq_temp = []
        struct_temp = []
        for i in range(len(seq)):
            if seq[i] in self.elements:
                seq_temp.append(seq[i])
                struct_temp.append(struct[i])
        seq = ''.join(seq_temp)
        struct = ''.join(struct_temp)
        seq_len = len(seq)
        if seq_len == 0:
            return 'Error'
        coding_len = min(len(seq), self.TRUNCATION_LEN)
        vec = [0] * coding_len
        for i in range(coding_len):
            pos = self.elements.index(seq[i])
            pos = pos + self.element_number * self.structs.index(struct[i])
            vec[i] = [0] * (self.element_number * self.struct_kind)
            vec[i][pos] = 1
        if coding_len < self.TRUNCATION_LEN:
            gap_len = (self.TRUNCATION_LEN - coding_len)
            if self.PERIOD_EXTENDED:
                gap_term = int(math.ceil(float(gap_len) / coding_len))
                gap_data = vec * gap_term
                vec += gap_data[0:gap_len]
            else:
                vec += [[0] * self.element_number * self.struct_kind] * gap_len
        return np.array(vec)

    def encode_word2vec(self, seq, rna_word2vec, window_size, stride):
        seq = seq.replace('T', 'U')
        seq = ''.join([x for x in seq if x in self.elements])
        seq_len = len(seq)
        if seq_len == 0:
            return 'Error'
        words = rna_word2vec.keys()
        VEC_LEN_W2V = len(list(words)[0])
        MAX_RNA_W2V_LEN = self.TRUNCATION_LEN // stride
        vec = []
        p = 0
        while (len(vec) < MAX_RNA_W2V_LEN) and (p + window_size <= seq_len):
            word = seq[p:p + window_size]
            if word in words:
                vec.append(rna_word2vec[word])
            p += stride
        encoded_len = len(vec)
        if encoded_len == 0:
            return 'Error'
        elif encoded_len < MAX_RNA_W2V_LEN:
            gap_len = (MAX_RNA_W2V_LEN - encoded_len)
            if self.PERIOD_EXTENDED:
                gap_term = int(math.ceil(float(gap_len) / encoded_len))
                gap_data = vec * gap_term
                vec += gap_data[0:gap_len]
            else:
                vec += [[0.0] * VEC_LEN_W2V] * gap_len
        return np.array(vec)

    def encode_word2vec_struct(self, seq, struct, rna_word2vec, window_size, stride):
        seq = seq.replace('T', 'U')
        struct = struct.replace(')', '(')
        seq_temp = []
        struct_temp = []
        for i in range(len(seq)):
            if seq[i] in self.elements:
                seq_temp.append(seq[i])
                struct_temp.append(struct[i])
        seq = ''.join(seq_temp)
        struct = ''.join(struct_temp)
        seq_len = len(seq)
        if seq_len == 0:
            return 'Error'

        words = rna_word2vec.keys()
        VEC_LEN_W2V = len(list(words)[0])
        MAX_RNA_W2V_LEN = self.TRUNCATION_LEN // stride
        vec = []
        p = 0
        while (len(vec) < MAX_RNA_W2V_LEN) and (p + window_size <= seq_len):
            word = seq[p:p + window_size] + struct[p:p + window_size]
            if word in words:
                vec.append(rna_word2vec[word])
            p += stride
        encoded_len = len(vec)
        if encoded_len == 0:
            return 'Error'
        elif encoded_len < MAX_RNA_W2V_LEN:
            gap_len = (MAX_RNA_W2V_LEN - encoded_len)
            if self.PERIOD_EXTENDED:
                gap_term = int(math.ceil(float(gap_len) / encoded_len))
                gap_data = vec * gap_term
                vec += gap_data[0:gap_len]
            else:
                vec += [[0.0] * VEC_LEN_W2V] * gap_len
        return np.array(vec)
