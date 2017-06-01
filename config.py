class config():
    dim = 123
    phngroup_filename = "phngroup"
    max_iter = None
    nepochs = 50
    batch_size = 5
    # dropout = 0.2
    lr = 0.001
    lr_decay = 0.95
    nepoch_no_imprv = 6
    hidden_size = 400
    output_path = "results/"
    model_output = output_path + "model.weights/"
    log_path = output_path + "log.txt"
    nb_layers = 2
    keep_prob = 0.5
    variational = True
