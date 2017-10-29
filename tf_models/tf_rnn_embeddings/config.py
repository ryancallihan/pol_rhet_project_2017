class DefaultConfig:
    n_epochs = 20
    start_lr = 0.001
    decay_steps = 177
    decay_rate = 0.95
    batch_size = 256
    max_timesteps = 30
    hidden_dropout = 0.8

    # Output sizes of the RNN layers.
    hidden_sizes = [128, 64]

    # Character embedding dropout
    input_dropout = 0.75

    # RNN output dropout
    rnn_output_dropout = 0.8

    # RNN state dropout
    rnn_state_dropout = 0.8
