import os

REGION = os.environ.get('REGION')

data_layers = {
    "static": {
        # /outputs/{region}/external/ + "static_layer_name": True/False,
        # modify input_dim_x calculation below if adding multichannel static files

    },
    "time": {
        # /outputs/{region}/tensors/ + "static_layer_name": True/False,

    },
}


hyperparameter_defaults_2D = dict(
    modeltype="2D",
    kernel_size=[(3, 3), (3, 3), (3, 3), (3, 3)],
    stride=[(2, 2), (1, 1), (1, 1), (1, 1)],
    padding=[0, 0, 0, 0],
    # Same as 3D
    region=REGION,
    size=34,  
    dropout=0.5081322909196353,  
    levels=[14],  
    batch_size=2048,  
    hidden_dim1=64,
    hidden_dim2=128,
    hidden_dim3=16,  
    hidden_dim4=64,
    lr=0.0014530946817692654,  
    weight_decay=0,  
    n_splits=5,  
    # Set criteria for Early stopping
    AUC=True,  
    BCE_Wloss=False,  
    FNcond=False,  
    n_epochs=25,  
    patience=7,  
    training_time=11.85,  
    # Weight on BCE loss
    pos_weight=2,  
    # set ratios of 0:1 labels in Train and Validation data sets
    train_times=4,  
    test_times=4,  
    # set parameters for the cost of the confusion matrix
    # weights on the False Negative Rate
    w=10,  
    # Batch params (debugging tools)
    stop_batch=None,  
    print_batch=10000,  
    # Set training time period
    # training data is all time series data 
    # from [start_year-1, end_year-1], predicting end_year deforestation
    start_year=18,  
    end_year=22,  
    # testing will see [start_year, end_year]
    # and be evaluated on end_year+1 (i.e., that year's labels)
    years_ahead=1,  
    data_layers=data_layers,  
)


hyperparameter_defaults_3D = dict(
    region=REGION,
    modeltype="3D",
    kernel_size=[(3, 3), (4, 3, 3), (3, 3), (2, 3, 3)],
    size=49,
    dropout=0.38651690390146254,
    levels=[14],
    batch_size=128,
    hidden_dim1=64,
    hidden_dim2=32,
    hidden_dim3=16,
    lr=7.635892111063769e-05,
    weight_decay=0,
    n_splits=5,
    AUC=True,
    BCE_Wloss=False,
    FNcond=False,
    n_epochs=5,
    patience=3,
    training_time=21,
    pos_weight=5,
    train_times=4,
    test_times=4,
    w=10,
    stop_batch=None,
    print_batch=10000,
    start_year=18,
    end_year=22,
    years_ahead=1,
    data_layers=data_layers,
    train_years=1,
)

hyperparameter_debug_3D = hyperparameter_defaults_3D.copy()
hyperparameter_debug_3D["print_batch"] = 10
hyperparameter_debug_3D["start_year"] = 17
hyperparameter_debug_3D["end_year"] = 20
hyperparameter_debug_3D["modeltype"] = "3D"
hyperparameter_debug_3D["data_layers"] = {
    "static": {
    }
}


# forecast_config_template = dict(
#     region="meghalaya_only",
#     modeltype="3D",
#     start_year = 19,
#     end_year = 23,
#     forecast_year = 24,

#     size = ,
#     hidden_dim1 = ,
#     hidden_dim2 = ,
#     hidden_dim3 = ,
#     kernel_size = [
#        [3, 3], 
#        [2, 3, 3], 
#        [3, 3], 
#        [2, 3, 3]
#     ],
#     levels = [],
#     dropout = ,
#     batch_size = ,

#     start_batch = 0,
#     stop_batch = ,
#     print_batch = ,
#     save_batch = ,
#     data_layers= ,
# )