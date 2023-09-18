import pandas as pd
import transformers
from transformers import AutoTokenizer
from dataloader.dataloader import create_test_data_loaders
import click
from utils.data_loading import create_dataframe
from utils.model_predicting_utils import predict

transformers.logging.set_verbosity_error()


@click.command()
@click.option('--test_path', type=str, default='data/test.csv', help="A path to the test csv file")
@click.option('--titles_path', type=str, default='data/titles.csv', help="A path to the titles csv file")
@click.option('--max_length', type=int, default=100, help="A path to the titles csv file")
@click.option('--checkpoint', type=str, default='checkpoints/checkpoint_epoch_8.pth',
              help="A path to the checkpoints file")
@click.option('--subm_path', type=str, default='./data/sample_submission.csv', help="A path to the submission csv file")
def main(test_path, titles_path, max_length, checkpoint, subm_path):
    test_df = create_dataframe(test_path, titles_path)

    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")

    test_dataloader = create_test_data_loaders(test_df, tokenizer, max_length)
    click.secho("Start prediction ... ", fg="green")
    prediction = predict(test_dataloader, checkpoint)
    submission = pd.read_csv(subm_path)

    submission['score'] = prediction
    submission['score'] = submission['score'].apply(lambda x: 0 if x < 0 else x)

    click.secho("Save prediction ... ", fg='green')

    submission.to_csv('submission.csv', index=False)
    click.secho("Prediction saved!", fg='green')


if __name__ == '__main__':
    main()
