import struct
import numpy as np


# 1. Read phntable and phngroup and fead_train

def file_to_list(input_file):
    """
    
    :param input_file: txt file or raw file
    :return: python list
    """
    with open(input_file) as f:
        output = f.readlines()
    output = [x.strip() for x in output]
    return output    

def set_to_list(input_file):
    char_set = file_to_list(input_file)
    list = []
    for i,elem in enumerate(char_set):
        if i != 26:
            list.append(elem.split(' ')[1])
        else:
            list.append(' ')
    return list

def hash_table(input_file):
    read = file_to_list(input_file)
    dict = {}
    for elem in read:
        key = elem.split(' ')[0]
        trans = elem.split(' ')[1:]
        dict[key] = trans
    return dict


#2. Feature parsing

def parse_feat(input_file):
    """
    
    :param input_file: feat file (string type)
    :return: parsed feature np.ndarray. Shape is (nSample, 123).
    """
    with open(input_file, "rb") as f:
        header = f.read(12)
        nSamples, _, sampSize, _ = struct.unpack(">iihh", header)
        nFeatures = sampSize // 4
        data = []
        for x in range(nSamples):
            s = f.read(sampSize)
            frame = []
            for v in range(nFeatures):
                val = struct.unpack_from(">f", s, v * 4)
                frame.append(val[0])
            data.append(frame)
        return np.array(data)


def parse_char(feat,dic,char_list):
    
    key = feat.split('/')[-1].split('.')[0]
    trans = dic[key]
    trans_idx = []
    for word in trans:
        for char in word:
            trans_idx.append(char_list.index(char))
        if word != trans[-1]:
            trans_idx.append(26)
    return trans_idx

def load_save_data(size, feat_train_list, dic, char_list, files, SAVE_DATA):
    if SAVE_DATA:
        data_in = []
        data_trans = []
        feat_train_list= feat_train_list
        for i in range(size):
            data_in.append(parse_feat(feat_train_list[i]))
            data_trans.append(parse_char(feat_train_list[i],dic,char_list))
        data_in = np.array(data_in)
        data_trans = np.array(data_trans)
        np.save(files[0], data_in)
        np.save(files[1], data_trans)
    else:
        data_in = np.load(files[0]+'.npy')
        data_trans = np.load(files[1]+'.npy')
    return data_in,data_trans

def make_batch(batch_in_list, batch_out_list):
    """

        :param batch_in_list: list of features. Can obtain by subseting the data_in_list
        :param batch_out_list: list of phonemes. Can obtain by subsetting the data_out_list
        :return:
         length: list of each feature. We will use this for dynamic_rnn. numpy.ndarray type
         batch_in_pad: numpy of padded input for max_length
         batch_out_pad : numpy of padded output for max_length
        """
    feat_dim = batch_in_list[0].shape[1]
    length = [batch_in_list[i].shape[0] for i in range(batch_in_list.shape[0])]
    max_dim = np.max(length)
    length_label = [len(batch_out_list[i]) for i in range(batch_out_list.shape[0])]
    max_label_dim = np.max(length_label)
    shape = [batch_in_list.shape[0], max_label_dim]
    indices = []
    values = []
    for i in range(batch_in_list.shape[0]):
        for j in range(length_label[i]):
            indices.append([i,j])
            values.append(batch_out_list[i][j])
    batch_in_pad = np.zeros(shape=[batch_in_list.shape[0], max_dim, feat_dim], dtype=np.float32)
    batch_out = tf.SparseTensor(indices = indices, values = values, shape = shape)
    for i in range(batch_in_list.shape[0]):
        batch_in_pad[i, 0:length[i], :] = batch_in_list[i]
    return np.array(length), batch_in_pad, batch_out

def input_norm(train, valid, test):
    length = [train[i].shape[0] for i in range(train.shape[0])]
    data_num = np.sum(length); print(data_num)
    feat_dim = train[0].shape[1]
    max_dim = np.max(length)
    train_in_pad = np.zeros(shape=[train.shape[0], max_dim, feat_dim], dtype=np.float32)
    for i in range(train.shape[0]):
        train_in_pad[i, 0:length[i], :] = train[i]
    train_in_mean = np.sum(np.sum(train_in_pad, axis=1),axis=0)/data_num
    train_in_std = 0.
    for i in range(train.shape[0]):
        for j in range(len(train[i])):
            train_in_std += (train[i][j] - train_in_mean)**2
    for i in range(train.shape[0]):
        for j in range(len(train[i])):
            train[i][j] = (train[i][j] - train_in_mean)/np.sqrt(train_in_std)
    for i in range(valid.shape[0]):
        for j in range(len(valid[i])):
            valid[i][j] = (valid[i][j] - train_in_mean)/np.sqrt(train_in_std)
    for i in range(test.shape[0]):
        for j in range(len(test[i])):
            test[i][j] = (test[i][j] - train_in_mean) / np.sqrt(train_in_std)
    return train, valid, test
