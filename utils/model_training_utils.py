import click
import torch
import numpy as np
from transformers import get_linear_schedule_with_warmup
from torchmetrics.functional import pearson_corrcoef, r2_score
from tqdm import tqdm


def get_sheduler(optimizer, total_steps):
    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    return scheduler


def train(model, train_dataloader, device, optimizer, criterion, scheduler):
    model.train()

    train_pearson = []
    train_r2_score = []
    total_loss_train = 0
    batch_loss_counter = 0
    batch_counts = 0
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    model = model.to(device)

    train_dataloader = tqdm(train_dataloader, total=len(train_dataloader), desc="Training")

    for step, batch in enumerate(train_dataloader):

        train_label = batch[2].to(device)
        mask = batch[1].to(device)
        input_id = batch[0].to(device)

        batch_counts += 1

        optimizer.zero_grad()

        output = model(input_id, mask, device)

        batch_loss = criterion(output.view(-1), train_label)

        batch_loss_counter += batch_loss.item()
        total_loss_train += batch_loss.item()

        pearson = pearson_corrcoef(output.view(-1), train_label)
        train_pearson.append(pearson.cpu().detach().numpy())

        r2 = r2_score(output.view(-1), train_label)
        train_r2_score.append(r2.cpu().detach().numpy())

        scaler.scale(batch_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1000)

        train_dataloader.set_description(
            f"Training - Loss: {batch_loss_counter / batch_counts:.4f},"
            f" Pearson: {np.mean(train_pearson):.4f},"
            f" R2: { np.mean(train_r2_score):.4f}"
        )

    scheduler.step()

    return batch_loss_counter/batch_counts, np.mean(train_pearson), np.mean(train_r2_score)


def valid(model, val_dataloader, best_val_score, device, criterion, epoch):
    model.eval()

    val_pearson = []
    val_r2_score = []
    val_loss = []
    model = model.to(device)

    val_dataloader = tqdm(val_dataloader, total=len(val_dataloader), desc="Validating")

    for batch in val_dataloader:
        val_label = batch[2].to(device)
        mask = batch[1].to(device)
        input_id = batch[0].to(device)

        with torch.no_grad():
            output = model(input_id, mask, device)

        batch_loss = criterion(output.view(-1), val_label)
        val_loss.append(batch_loss.item())

        pearson = pearson_corrcoef(output.view(-1), val_label)
        val_pearson.append(pearson.cpu().numpy())

        r2 = r2_score(output.view(-1), val_label)
        val_r2_score.append(r2.cpu().detach().numpy())

    val_loss_mean = np.mean(val_loss)
    val_pearson_mean = np.mean(val_pearson)
    val_r2_score_mean = np.mean(val_r2_score)
    val_dataloader.set_description(
        f"Training - Loss: {val_loss_mean:.4f},"
        f" Pearson: {val_pearson_mean:.4f},"
        f" R2: {np.mean(val_r2_score_mean):.4f}"
    )
    if val_pearson_mean >= max(best_val_score):
        checkpoint = {'state_dict': model.state_dict()}

        click.secho("Best model", fg="green")
        click.secho(f"Validation score is : {val_pearson_mean}", fg="green")
        click.secho("Saving model!", fg="green")
        torch.save(checkpoint, f'./checkpoint_epoch_{epoch + 1}.pth')
        click.secho("Model Saved!", fg="green")

        best_val_score.append(val_pearson_mean)

    return val_loss_mean, val_pearson_mean, val_r2_score_mean
