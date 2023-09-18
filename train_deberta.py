import torch
import numpy as np
from torchmetrics.functional import pearson_corrcoef, r2_score
from tqdm import tqdm
from dataloader.dataloader import create_data_loaders
import transformers
from transformers import AutoTokenizer
import click
from torch.optim import Adam
from torch import nn
from models.custom_model import DeBertav3Regressor
from utils.model_training_utils import get_sheduler, train
from utils.displaying import display_values
from utils.data_loading import create_dataframe
import os

transformers.logging.set_verbosity_error()


@click.command()
@click.option('--train_path', type=str, default='data/train.csv', help="A path to the train csv file")
@click.option('--titles_path', type=str, default='data/titles.csv', help="A path to the titles csv file")
@click.option('--model_name', type=str, default='microsoft/deberta-v3-large', help="The model name from "
                                                                                   "the hugging face platform ")
@click.option('--batch_size', type=int, default=8, help="A batch size of the input in the model")
@click.option('--max_length', type=int, default=100, help="A path to the titles csv file")
@click.option('--learning_rate', type=float, default=1e-5, help="A learning rate for the optimizer")
@click.option('--weight_decay', type=float, default=5e-5, help="A weight decay parameters for the optimizer")
@click.option('--epochs', type=int, default=20, help="A number of training steps")
@click.option('--save_dir', type=str, default='./checkpoints', help="A path for saving the model checkpoints")
@click.option('--patience', type=int, default=4, help="Number of epochs with no improvement to wait before stopping")
@click.option('--images_dir', type=str, default='./images', help="A path for saving images of training metrics")
def main(train_path, titles_path, model_name, batch_size, max_length, learning_rate, weight_decay, epochs, save_dir,
         patience, images_dir):

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if not os.path.exists(images_dir):
        os.makedirs(images_dir)

    train_df = create_dataframe(train_path, titles_path)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_dataloader, val_dataloader = create_data_loaders(train_df, tokenizer, batch_size, max_length)

    use_cuda = torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = nn.MSELoss()

    model = DeBertav3Regressor(model_name)

    optimizer = Adam(model.parameters(), lr=learning_rate, eps=1e-6, weight_decay=weight_decay)

    scheduler = get_sheduler(optimizer, epochs)

    max_val_pearson = float('-inf')
    epochs_without_improvement = 0

    best_val_score = [-10]

    list_total_loss_train, list_train_pearson, list_train_r2_score = [], [], []
    list_val_loss_mean, list_val_pearson_mean, list_val_r2_score_mean = [], [], []

    for epoch in range(epochs):

        click.secho(f"Epoch №: {epoch + 1}", fg="green")
        model.train(True)

        total_loss_train, train_pearson, train_r2_score = train(model, train_dataloader, device,
                                                                optimizer, criterion, scheduler)

        model.eval()

        val_pearson = []
        val_r2_score = []
        val_loss = []

        val_dataloader = tqdm(val_dataloader, total=len(val_dataloader), desc="Validation")

        for batch in val_dataloader:
            val_label = batch[2].to(device)
            mask = batch[1].to(device)
            input_id = batch[0].to(device)

            with torch.no_grad():
                output = model(input_id, mask, device)

            batch_loss = criterion(output.view(-1), val_label)
            val_loss.append(batch_loss.item())

            pearson = pearson_corrcoef(output.view(-1), val_label)
            val_pearson.append(pearson.cpu().detach().numpy())

            r2 = r2_score(output.view(-1), val_label)
            val_r2_score.append(r2.cpu().detach().numpy())

            val_dataloader.set_description(
                f"Validation - Loss: {np.mean(val_loss):.4f},"
                f" Pearson: {np.mean(val_pearson):.4f},"
                f" R2: {np.mean(val_r2_score):.4f}"
            )

        val_loss_mean = np.mean(val_loss)
        val_pearson_mean = np.mean(val_pearson)
        val_r2_score_mean = np.mean(val_r2_score)

        if val_pearson_mean >= max(best_val_score):
            click.secho("Best model", fg="green")
            click.secho(f"Validation score is : {val_pearson_mean}", fg="green")
            click.secho("Saving model!", fg="green")
            torch.save(model.state_dict(), f'{save_dir}/checkpoint_epoch_{epoch + 1}.pth')
            click.secho("Model Saved!", fg="green")

            best_val_score.append(val_pearson_mean)

        click.secho(f"Training Loss after {epoch + 1} epoch: {total_loss_train}", fg="green")
        click.secho(f"Training Pearson score {epoch + 1} epoch: {train_pearson}", fg="green")
        click.secho(f"Training R2 score {epoch + 1} epoch: {train_r2_score}", fg="green")
        click.secho(f"Valid Loss after {epoch + 1} epoch: {val_loss_mean}", fg="green")
        click.secho(f"Valid Pearson score {epoch + 1} epoch: {val_pearson_mean}", fg="green")
        click.secho(f"Valid R2 score {epoch + 1} epoch: {val_r2_score_mean}", fg="green")

        list_total_loss_train.append(total_loss_train)
        list_train_pearson.append(train_pearson)
        list_train_r2_score.append(train_r2_score)
        list_val_loss_mean.append(val_loss_mean)
        list_val_pearson_mean.append(val_pearson_mean)
        list_val_r2_score_mean.append(val_r2_score_mean)

        if val_pearson_mean > max_val_pearson:
            max_val_pearson = val_pearson_mean
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            click.secho(f'Early stopping after {epoch + 1} epochs without improvement', fg="red")
            click.secho(f'The best model on {epoch + 1 - patience} epoch.', fg="red")
            break

    display_values(list_total_loss_train, list_val_loss_mean, type_of_plot="Loss")
    display_values(list_train_pearson, list_val_pearson_mean, type_of_plot="Pearson")
    display_values(list_train_r2_score, list_val_r2_score_mean, type_of_plot="R2")


if __name__ == '__main__':
    main()
