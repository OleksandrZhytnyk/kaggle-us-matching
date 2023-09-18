from models.custom_model import DeBertav3Regressor
import torch

def load_checkpoint(model,filepath, device):
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint)
    for parameter in model.parameters():
        parameter.requires_grad = False

    model.eval()
    return model


def predict(test_dataloader,  checkpoint):

    use_cuda = torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")

    model = DeBertav3Regressor("microsoft/deberta-v3-large")

    checkpoint = torch.load(checkpoint)

    model.load_state_dict(checkpoint)

    model.eval()

    model.to(device)

    prediction = []

    for batch in test_dataloader:
        mask = batch[1].to(device)
        input_id = batch[0].to(device)

        with torch.no_grad():
            output = model(input_id, mask, device)
            prediction.append(output.reshape(-1).cpu().numpy()[0])

    return prediction
