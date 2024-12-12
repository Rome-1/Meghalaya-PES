"""
SCRIPT FOR TESTING 2DCNN MODELS
"""
import time
import torch
import numpy as np
from models.CNN import CNNmodel
from models.Training import ImbalancedDatasetUnderSampler, test_model, visualize, write_report, train_model
from models.Testing import *
from preprocessing.Data_maker_loader import *
from sklearn.model_selection import train_test_split

def train_2DCNN(
    wherepath,
    modelpath,
    picspath,
    file,
    start_year,
    end_year,
    input_dim,
    hidden_dim1,
    hidden_dim2,
    hidden_dim3,
    hidden_dim4,
    kernel_size,
    stride,
    padding,
    levels,
    train_times,
    test_times,
    AUC,
    BCE_Wloss,
    FNcond,
    perc,
    w,
    pos_weight,
    lr,
    dropout,
    size,
    weight_decay,
    n_splits,
    n_epochs,
    patience,
    training_time,
    stop_batch,
    job_id,
    batch_size,
    print_batch,
    data_layers={},
    years_ahead=1,
    pretrained_path=None,
    model_type_name="2D",
):
    if BCE_Wloss:
        raise NotImplementedError()
    
    start = time.time()

    # Set up model
    hidden_dim = [
        hidden_dim1,
        hidden_dim2,
        hidden_dim3,
        hidden_dim4,
    ]

    model = CNNmodel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        kernel_size=kernel_size,
        levels=levels,
        dropout=dropout,
        stride=stride,
        padding=padding,
    )
    model = torch.nn.DataParallel(model)

    # If pretrained path is specified, load in model parameters
    if pretrained_path is not None and pretrained_path != "":
        print("Pretrained checkpoint: " + pretrained_path)

        # Load to GPU is possible, else CPU
        if torch.cuda.is_available() is False:
            pretrained_model = torch.load(pretrained_path, map_location=torch.device('cpu'))
        else:
            pretrained_model = torch.load(pretrained_path)

        model.load_state_dict(pretrained_model["model_state_dict"])
        
        print("Pretrained model loaded in as starting point")

    # Set loss criterion and optimiser type
    # criterion = torch.nn.CrossEntropyLoss(reduction='mean', weight = pos_weight)
    criterion = torch.nn.BCEWithLogitsLoss(
        reduction="mean", pos_weight=torch.tensor(pos_weight)
    )
    optimiser = torch.optim.Adam(
        params=model.parameters(), lr=lr, weight_decay=weight_decay
    )

    # Load data
    Data = with_DSM(
        size=int(size / 2),
        start_year=start_year,
        end_year=end_year,
        wherepath=wherepath,
        type="2D",
        data_layers=data_layers,
        years_ahead=years_ahead
    )

    if not (
        os.path.isfile(wherepath + "/" + "Train2D_idx%d.npy" % (end_year))
        & os.path.isfile(wherepath + "/" + "Test2D_idx%d.npy" % (end_year))
    ):
        print("Creating indexes split")
        train_idx, test_idx = train_test_split(
            np.arange(len(Data.labels)),
            test_size=0.2,
            random_state=42,
            shuffle=True,
            stratify=Data.labels,
        )
        np.save(wherepath + "/" + "Train2D_idx%d.npy" % (end_year), train_idx)
        np.save(wherepath + "/" + "Test2D_idx%d.npy" % (end_year), test_idx)
    else:
        print("loading: " + wherepath + "/" + "Train2D_idx%d.npy" % (end_year))
        train_idx = np.load(wherepath + "/" + "Train2D_idx%d.npy" % (end_year))
        test_idx = np.load(wherepath + "/" + "Test2D_idx%d.npy" % (end_year))

    # Set train and test samplers
    train_sampler = ImbalancedDatasetUnderSampler(
        labels=Data.labels, indices=train_idx, times=train_times
    )
    test_sampler = ImbalancedDatasetUnderSampler(
        labels=Data.labels, indices=test_idx, times=test_times
    )

    # Print model and training details
    print(
        "Model:",
        str(type(model))[8:-2],
        "\nPeriod 20%d-20%d -> 20%d"
        % (start_year, end_year, end_year + 1),
    )
    print(
        "\t% deforested pixels in train:",
        train_sampler.count[1] / sum(train_sampler.count),
    )
    print(
        "\t% deforested pixels in val:", test_sampler.count[1] / sum(test_sampler.count)
    )
    print("Job: ", job_id)
    print("\nHyperparameters: ")
    print("\tImage size: %d" % (size))
    print("\tHidden dim: ", hidden_dim)
    print("\tDropout: ", dropout)
    print(
        "\tTrain and Val ratios of 0:1 labels: 1:%d ; 1:%d "
        % (train_times, test_times)
    )
    print(
        "\tADAM optimizer parameters: lr=%.7f, weight decay=%.2f, batch size=%d"
        % (lr, weight_decay, batch_size)
    )
    print("\tBCEWithLogitsLoss pos_weights = %.2f" % (pos_weight))
    print(
        "\tn_epochs = %d with patience of %d epochs"
        % (n_epochs, patience)
    )
    print("\tCross Validation with n_splits = %d " % (n_splits))
    print(
        "\tIf to use BCEWithLogitsLoss as an early stop criterion :",
        ((not AUC) & (not FNcond)),
    )
    print("\tIf to use AUC as an early stop criterion :", AUC)
    print("\tIf to use cost = FP+w*FN / TP+FP+w*FN+TN as an early stop criterion")
    print(
        "\twith w = %d and treshhold = the %d percentile of the output"
        % (w, perc),
        FNcond,
    )
    print("\nModel: \n", model)
    print("\nCriterion: \n", criterion)
    print("\nOptimiser: \n", optimiser)

    # Initiate training routine
    (
        model,
        train_loss,
        valid_loss,
        AUCs_train,
        AUCs_val,
        costs_train,
        costs_val,
        name,
    ) = train_model(
        Data=Data,
        model=model,
        sampler=train_sampler,
        criterion=criterion,
        optimiser=optimiser,
        patience=patience,
        n_epochs=n_epochs,
        n_splits=n_splits,
        batch_size=batch_size,
        stop_batch=stop_batch,
        print_batch=print_batch,
        training_time=training_time,
        w=w,
        FNcond=FNcond,
        AUC=AUC,
        job=job_id,
        path=modelpath,
    )
    # Produce graphs
    visualize(
        train=train_loss,
        valid=valid_loss,
        name="BCEloss",
        modelname=name,
        best="min",
        path=picspath,
    )
    visualize(
        train=AUCs_train,
        valid=AUCs_val,
        name="AUC",
        modelname=name,
        best="max",
        path=picspath,
    )
    visualize(
        train=costs_train,
        valid=costs_val,
        name="Cost",
        modelname=name,
        best="min",
        path=picspath,
    )

    test_loss, test_AUC, test_cost = test_model(
        model=model,
        Data=Data,
        criterion=criterion,
        w=w,
        perc=perc,
        test_sampler=test_sampler,
        batch_size=batch_size,
        stop_batch=stop_batch,
        name=name,
        path=picspath,
    )

    write_report(
        name=name,
        job_id=job_id,
        train_loss=train_loss,
        valid_loss=valid_loss,
        test_loss=test_loss,
        AUCs_train=AUCs_train,
        AUCs_val=AUCs_val,
        test_AUC=test_AUC,
        costs_train=costs_train,
        costs_val=costs_val,
        test_cost=test_cost,
        file=file,
        FNcond=FNcond,
        AUC=AUC,
    )

    print("\n\nEND!Total time (in h):", (time.time() - start) / 3600)

    return name