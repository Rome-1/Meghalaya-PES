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

# 1 years ahead (3YA) model training and forecasting configs
# Stage 1: train and test using all available data
# Stage 2: train further on testing data
# Stage 3: Forecast for 2024
config_1YA_stage_1 = dict(
    region="meghalaya_only",
    modeltype="3D",
    kernel_size=[
        [3, 3],
        [2, 3, 3],
        [3, 3],
        [2, 3, 3]
    ],
    size=34,
    dropout=0.3,
    levels=[14],
    batch_size=2048,
    hidden_dim1=64,
    hidden_dim2=128,
    hidden_dim3=16,
    lr=0.0014530946817692654,
    weight_decay=0,
    n_splits=5,
    AUC=True,
    BCE_Wloss=False,
    FNcond=False,
    n_epochs=3,
    patience=3,
    training_time=21,
    pos_weight=2,
    train_times=4,
    test_times=4,
    w=10,
    stop_batch=None,
    print_batch=10000,
    start_year=15,
    end_year=20,
    years_ahead=1,
    data_layers=dict(
        static=dict(
            community_reserves_cropped=True,
            protected_areas_cropped=True,
            sacred_groves_cropped=True,
            slope_reproj_cr=True,
            meghalaya_DEM_reproj_cr=True,
            updated_block_boundaries_reproj_one_hot=True,
            updated_roads_reproj_one_hot=True,
            village_points_cropped=True
        ),
        time=dict(
            nightlight_median=True,
            nightlight_maximum=True
        )
    ),
    train_years=3
)

config_1YA_stage_2 = dict(
    region="meghalaya_only",
    modeltype="3D",
    kernel_size=[
        [3, 3],
        [2, 3, 3],
        [3, 3],
        [2, 3, 3]
    ],
    size=34,
    dropout=0.3,
    levels=[14],
    batch_size=512,
    hidden_dim1=64,
    hidden_dim2=128,
    hidden_dim3=16,
    lr=0.0014530946817692654,
    weight_decay=0,
    n_splits=5,
    AUC=True,
    BCE_Wloss=False,
    FNcond=False,
    n_epochs=1,
    patience=3,
    training_time=21,
    pos_weight=2,
    train_times=4,
    test_times=4,
    w=10,
    stop_batch=None,
    print_batch=10000,
    start_year=18,
    end_year=23,
    years_ahead=1,
    data_layers=dict(
        static=dict(
            community_reserves_cropped=True,
            protected_areas_cropped=True,
            sacred_groves_cropped=True,
            slope_reproj_cr=True,
            meghalaya_DEM_reproj_cr=True,
            updated_block_boundaries_reproj_one_hot=True,
            updated_roads_reproj_one_hot=True,
            village_points_cropped=True
        ),
        time=dict(
            nightlight_median=True,
            nightlight_maximum=True
        )
    ),
    train_years=1
)

forecast_config_1YA_stage_3 = dict(
    region="meghalaya_only",
    modeltype="3D",

    # Year settings
    start_year=20,
    end_year=23,
    forecast_year=24,

    # Model parameters
    size=34,
    hidden_dim1=64,
    hidden_dim2=128,
    hidden_dim3=16,
    kernel_size=[
        [3, 3], 
        [2, 3, 3], 
        [3, 3], 
        [2, 3, 3]
    ],
    levels=[14],
    dropout=0.3,
    batch_size=512,

    # Batch processing parameters
    start_batch=0,
    stop_batch=None,
    print_batch=500,
    save_batch=None,

    # Data Layers for forecasting
    data_layers=dict(
        static=dict(
            community_reserves_cropped=True,
            protected_areas_cropped=True,
            sacred_groves_cropped=True,
            slope_reproj_cr=True,
            meghalaya_DEM_reproj_cr=True,
            updated_block_boundaries_reproj_one_hot=True,
            updated_roads_reproj_one_hot=True,
            village_points_cropped=True
        ),
        time=dict(
            nightlight_median=True,
            nightlight_maximum=True
        )
    )
)

# 3 years ahead (3YA) model training and forecasting configs
# Stage 1: train and test using all available data
# Stage 2: train further on testing data
# Stage 3: Forecast for 2024
config_3YA_stage_1 = dict(
    region="meghalaya_only",
    modeltype="3D",
    kernel_size=[
        [3, 3],
        [2, 3, 3],
        [3, 3],
        [2, 3, 3]
    ],
    size=29,
    dropout=0.38651690390146254,
    levels=[6],
    batch_size=64,
    hidden_dim1=256,
    hidden_dim2=128,
    hidden_dim3=16,
    lr=7.635892111063769e-05,
    weight_decay=0,
    n_splits=5,
    AUC=True,
    BCE_Wloss=False,
    FNcond=False,
    n_epochs=1,
    patience=3,
    training_time=21,
    pos_weight=5,
    train_times=4,
    test_times=4,
    w=10,
    stop_batch=None,
    print_batch=10000,
    start_year=16,
    end_year=19,
    years_ahead=3,
    data_layers=dict(
        static=dict(
            community_reserves_cropped=True,
            protected_areas_cropped=True,
            sacred_groves_cropped=True,
            slope_reproj_cr=True,
            meghalaya_DEM_reproj_cr=True,
            updated_block_boundaries_reproj_one_hot=True,
            updated_roads_reproj_one_hot=True,
            village_points_cropped=True
        ),
        time=dict(
            nightlight_median=True,
            nightlight_maximum=True
        )
    ),
    train_years=3
)

config_3YA_stage_2 = dict(
    region="meghalaya_only",
    modeltype="3D",
    kernel_size=[
        [3, 3],
        [2, 3, 3],
        [3, 3],
        [2, 3, 3]
    ],
    size=29,
    dropout=0.38651690390146254,
    levels=[6],
    batch_size=64,
    hidden_dim1=256,
    hidden_dim2=128,
    hidden_dim3=16,
    lr=7.635892111063769e-05,
    weight_decay=0,
    n_splits=5,
    AUC=True,
    BCE_Wloss=False,
    FNcond=False,
    n_epochs=1,
    patience=3,
    training_time=21,
    pos_weight=5,
    train_times=4,
    test_times=4,
    w=10,
    stop_batch=None,
    print_batch=10000,
    start_year=18,
    end_year=21,
    years_ahead=3,
    data_layers=dict(
        static=dict(
            community_reserves_cropped=True,
            protected_areas_cropped=True,
            sacred_groves_cropped=True,
            slope_reproj_cr=True,
            meghalaya_DEM_reproj_cr=True,
            updated_block_boundaries_reproj_one_hot=True,
            updated_roads_reproj_one_hot=True,
            village_points_cropped=True
        ),
        time=dict(
            nightlight_median=True,
            nightlight_maximum=True
        )
    ),
    train_years=1
)

forecast_config_3YA_stage_3 = dict(
    region="meghalaya_only",
    modeltype="3D",
    start_year = 20,
    end_year = 23,
    forecast_year = 24,

    size = 29,
    hidden_dim1=256,
    hidden_dim2=128,
    hidden_dim3=16,
    kernel_size = [
        [3, 3], 
        [2, 3, 3], 
        [3, 3], 
        [2, 3, 3]
    ],
    levels = [6],
    dropout = 0.38651690390146254,
    batch_size = 512, # 64

    start_batch=0,
    stop_batch=None,
    print_batch=500, # 10000
    save_batch=None,

    data_layers=dict(
        static=dict(
            community_reserves_cropped=True,
            meghalaya_DEM_reproj_cr=True,
            protected_areas_cropped=True,
            sacred_groves_cropped=True,
            slope_reproj_cr=True,
            updated_block_boundaries_reproj_one_hot=True,
            updated_roads_reproj_one_hot=True,
            village_points_cropped=True
        ),
        time=dict(
            nightlight_maximum=True,
            nightlight_median=True
        )
    )
)


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