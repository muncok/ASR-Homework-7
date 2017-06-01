import numpy as np
from data_processing import file_to_list, set_to_list, hash_table, load_save_data
import numpy as np

from data_processing import file_to_list, set_to_list, hash_table, load_save_data

#from train import train, optimizer
#from evaluation import evaluation
#from model.model import Model

char_list = set_to_list('char_set.txt')
key_to_trans = hash_table('train_all.trans')
feat_train_list = file_to_list('train_all.list')
feat_test_list = file_to_list('train_all.list')        # Set 'feat_train.list' -> 'feat_test.list' to test


####################################
###        LOAD DATA             ###
####################################
"""
NORMALIZE_INPUT = True
"""
files = ['data/data_in', 'data/data_trans']
files_wsj0 = ['data/data_wsj0_in', 'data/data_wsj0_trans']
files_wsj1 = ['data/data_wsj1_in', 'data/data_wsj1_trans']
files_toy_wsj = ['data/toy_wsj_in', 'data/toy_wsj0_trans']
SAVE_DATA = True#True
if SAVE_DATA:
    data_in, data_trans = load_save_data(10, feat_train_list, key_to_trans, char_list, files, SAVE_DATA)
    # data_wsj0_in = np.save_compressed(files_wsj0[0], data_in[0:4195])
    # data_wsj0_trans = np.savez_compressed(files_wsj0[1], data_trans[0:4195])
    # data_wsj1_in = np.savez_compressed(files_wsj1[0], data_in[4195:])
    # data_wsj1_trans = np.savez_compressed(files_wsj1[1], data_trans[4195:])
    data_toy_in = np.save(files_toy_wsj[0], data_in)
    data_toy_trans = np.save(files_toy_wsj[1], data_in)
else:
    data_wsj0_in, data_wsj0_trans = load_save_data(feat_train_list, key_to_trans, char_list, files_wsj0, SAVE_DATA)
    #data_in_wsj1 , data_trans_wsj1 = load_save_data(feat_train_list, key_to_trans, char_list, files_wsj1, SAVE_DATA)

"""
test_files = ['test_in_list', 'test_out_list']
SAVE_DATA = True #False : Load '.npy' Data
train_in_list, train_out_list, valid_in_list, valid_out_list = load_save_data(feat_train_list, phntable, files,
                                                                              SAVE_DATA)
test_in_list , test_out_list = load_save_test(feat_test_list, phntable, test_files, SAVE_DATA)

if NORMALIZE_INPUT:
    train_in_list, valid_in_list,test_in_list = input_norm(train_in_list, valid_in_list,test_in_list)

####################################
###        Hyperparameters       ###
####################################

learning_rate = 0.01
opt_name = 'Adam'
batch_size = 20
save_step = 1
num_of_features = 123  # input size
output_size = 61  # output size
rnn_hidden_neurons = 123  # hidden layer num of features
num_of_layers = 2  # the number of Stacked LSTM layers
rnn_type = 'LSTM'
recurrent_dropout = True
output_dropout = True
layernorm = False
LOAD_MODEL = False
SAVE_MODEL = True
MODEL_NAME = rnn_type+'_OUTPUT_RECURRENT_DROPOUT_ADAM'+str(learning_rate)
num_epoch = 3000
clip_norm = 5.

def main():
    with tf.variable_scope('PLACEHOLDER'):
        inputs = tf.placeholder(tf.float32, [None, None, num_of_features], 'inputs')  # (batch, max_dim, in)
        outputs = tf.placeholder(tf.int32, [None, None], 'outputs')  # (batch, max_dim)
        seq_length = tf.placeholder(tf.int32, [None], 'seq_length')
        mask = tf.placeholder(tf.float32, [None, None], 'mask')
        keep_prob = tf.placeholder(tf.float32, None, 'keep_prob')
    model = Model(batch_size, num_of_features, rnn_hidden_neurons, output_size, num_of_layers,  table_to_group, recurrent_dropout, output_dropout,
                  keep_prob, layernorm, rnn_type, MODEL_NAME)
    rnn_output = model.get_output(inputs, seq_length)
    cost, accuracy_61, accuracy_39 = model.get_cost_accuracy(output=rnn_output, target=outputs, length=seq_length, masking=mask)
    opt = optimizer(optimizer_name=opt_name, cost_=cost, lr=learning_rate, clip_norm=clip_norm)

    ###########################
    ###      Train          ###
    ###########################

    sess = tf.Session()
    #train(sess, inputs, outputs, seq_length, mask, keep_prob, model, cost, accuracy_61, accuracy_39, opt,
    #      batch_size, num_epoch, LOAD_MODEL, SAVE_MODEL, save_step,
    #      train_in_list, train_out_list, valid_in_list, valid_out_list, OVER_FITTING)
    evaluation(sess, inputs, outputs, seq_length, mask, keep_prob, model, cost, accuracy_61, accuracy_39,
               batch_size, num_epoch, LOAD_MODEL, train_in_list, train_out_list)
    evaluation(sess, inputs, outputs, seq_length, mask, keep_prob, model, cost, accuracy_61, accuracy_39,
               batch_size, num_epoch, LOAD_MODEL, valid_in_list, valid_out_list)
    evaluation(sess, inputs, outputs, seq_length, mask, keep_prob, model, cost, accuracy_61, accuracy_39,
               batch_size, num_epoch, LOAD_MODEL, test_in_list, test_out_list)
if __name__ == '__main__':
    main()
"""
