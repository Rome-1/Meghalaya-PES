from models.ConvRNN import *

def parse_model_forward(model_output, target, data, criterion, reconstruction_criterion, recon_loss_weight):
    recon_loss_2D = recon_loss_3D = None

    # Check if the output contains auxiliary outputs (ie two reconstruction weights)
    if isinstance(model_output, tuple):
        output, auxiliary_outputs = model_output
        recon_2D, recon_3D = auxiliary_outputs

        # Testing / Forecasting
        if criterion is None:
            return output, None, None, None

        classification_loss = criterion(output, target.float())
        recon_loss_2D = reconstruction_criterion(recon_2D, data[0])  # 2D input
        recon_loss_3D = reconstruction_criterion(recon_3D, data[1])  # 3D input
        
        loss = classification_loss + recon_loss_weight * (recon_loss_2D + recon_loss_3D)

    else:
        output = model_output

        # Testing / Forecasting
        if criterion is None:
            return output, None, None, None

        loss = criterion(model_output, target.float())

    return output, loss, recon_loss_2D, recon_loss_3D

def get_model_type(model_type_name, start_year, end_year):
    modeltype = None
    if (start_year - end_year) % 2 == 0:
        if model_type_name == "3D":
            modeltype = Conv_3Dodd
        elif model_type_name == "3DAutoencoder":
            modeltype = Conv_3DoddWithAutoencoders
        elif model_type_name == "3DOnly":
            modeltype = Conv_3DoddOnly
    else:
        if model_type_name == "3D":
            modeltype = Conv_3Deven
        elif model_type_name == "3DAutoencoder":
            modeltype = Conv_3DevenWithAutoencoders
        elif model_type_name == "3DOnly":
            modeltype = Conv_3DevenOnly

    if model_type_name == "3DUNet":
        modeltype = Conv_3DUNet
    elif model_type_name == "3DBasicAtt":
        modeltype = Conv_3DBasicAttention
    
    if modeltype is None:
        raise ValueError(f"No model of type {model_type_name} configured.")

    return modeltype