class CFG():
    LEARNING_RATE = 0.001
    EPOCHS = 50
    BATCH_SIZE = 64
    MODEL ='GCN'

    HIDDEN_DIM = 128
    DROPOUT = 0.5
    GNN_LAYERS = 2
    
    DATASET = "Cora"
    TRAIN_SPLIT = 0.8
    VAL_SPLIT = 0.1

    SEED = 42
    DEVICE = "cpu"

    ANOMALY_TYPE = "node"
    FEATURE_OUT = 2
