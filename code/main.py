import argparse
from random import choice
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import os
import shutil
from tqdm import tqdm
import numpy as np
from data.dataset import get_dataloaders
from models.GIT_model import GIT_Model, GIT_Model_with_Classification
from train_eval.eval import eval_epoch
from train_eval.train import train_epoch
import logging
from torch.utils.tensorboard import SummaryWriter
import time

logger = logging.getLogger(__name__)


def log_model_parameters_summary(model):
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    pytorch_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'Total parameters: {pytorch_total_params}')
    logger.info(f'Trainable params: {pytorch_trainable_params}')
    logger.info(f'Trainable params: {pytorch_trainable_params / pytorch_total_params * 100}%')

# Define and parse arguments.
def _args():
    
    # Define a parser.
    parser = argparse.ArgumentParser()

    # Arguments for the logs.
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval', 'resume'])
    parser.add_argument('--resume_ckpt', type=str, default='best_checkpoint.pth.tar', choices=['best_checkpoint.pth.tar', 'last_checkpoint.pth.tar', 'best_loss_checkpoint.pth.tar', 'best_average_checkpoint.pth.tar'])
    parser.add_argument('--log_level', type=str, default='INFO', choices=[
                        'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help='log level')
    
    # Arguments for the type of model.
    parser.add_argument('--model', type=str, default="microsoft/git-base-coco", choices=["microsoft/git-base-coco", "microsoft/git-base", "microsoft/git-large-coco", "microsoft/git-large", "microsoft/git-base-msrvtt-qa", "microsoft/git-large-msrvtt-qa"])
    parser.add_argument('--processor', type=str, default="microsoft/git-base-coco", choices=["microsoft/git-base-coco", "microsoft/git-base", "microsoft/git-large-coco", "microsoft/git-large"])
    parser.add_argument('--model_variation', type=str, default='default', choices=['default', 'with_classification'])
    parser.add_argument('--data_variation', type=str, default='single_view', choices=['single_view', 'multi_view', 'multi_view_temporal'])
    parser.add_argument('--num_views', type=int, default=2, help='The number of views; used only when data_variation is multi_view or multi_view_temporal')
    parser.add_argument('--use_pretrained', type=str, default='legacy', choices=['legacy', 'no', 'yes'])

    parser.add_argument('--curriculum_learning', type=str, default='none', choices=['none', 'linear', 'exponential'])
    parser.add_argument('--curriculum_learning_percent', type=float, default=0.2)
    parser.add_argument('--curriculum_learning_progress', type=float, default=None)
    parser.add_argument('--curriculum_learning_cls_method', type=str, default='none', choices=['none', 'update_labels_weights', 'add_correction_weights'])

    parser.add_argument('--out', type=str, default='checkpoints')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cuda')
    
    # Arguments for  hyperparameter tuning.    
    parser.add_argument('--lr', type=float, default=0.00005)
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adamw'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=None)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=2)
    # 160 computed to cover 99.2% of impressions, 98.4% of findings and 92.5% of impressions+findings
    # 192 to cover 94% of findings+impression+history+indication on processed text
    parser.add_argument('--max_len', type=int, default=-1, help='-1 for auto absed on context_format and target_format')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--img_scale_factor', type=float, default=1)
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--max_train_samples', type=int, default=None)
    parser.add_argument('--max_evaltrain_samples', type=int, default=100)
    parser.add_argument('--max_valid_samples', type=int, default=None)
    parser.add_argument('--max_test_samples', type=int, default=None)
    parser.add_argument("--max_epoch_samples", type=int, default=None)
    parser.add_argument('--validation_frequency', type=int, default=1)
    
    # Arguments for dataset metadata.
    parser.add_argument('--csv_path', type=str, default=r'E:\MIMIC\physionet.org\files\mimic-cxr\2.0.0\cxr-record-list.csv')
    parser.add_argument('--splits_path', type=str, default=r'E:\MIMIC\physionet.org\files\mimic-cxr\2.0.0\mimic-cxr-2.0.0-split.csv')
    parser.add_argument('--findings_path', type=str, default=r'E:\MIMIC\physionet.org\files\mimic-cxr\2.0.0\mimic-cxr-sections\mimic_cxr_sectioned.csv')
    parser.add_argument('--metadata_path', type=str, default=r'E:\MIMIC\physionet.org\files\mimic-cxr\2.0.0\mimic-cxr-2.0.0-metadata-processed-p19.csv')
    parser.add_argument('--images_path', type=str, default=r'E:\MIMIC\physionet.org\files\mimic-cxr\2.0.0\files\jpg')
    parser.add_argument('--labels_path', type=str, default='')
    parser.add_argument('--aux_loss_weight', type=float, default=1)
    parser.add_argument('--num_labels', type=int, default=-1, choices=[-1, 4, 2]) # -1 for auto (4)

    parser.add_argument('--context_format', type=str, default='', choices=['', 'history+indication'])
    parser.add_argument('--target_format', type=str, default='findings_legacy', choices=['findings_legacy', 'findings', 'impression', 'impression+findings', 'findings+impression'])
    parser.add_argument('--max_context_len', type=int, default=45)

    # Arguments for LoRA.
    parser.add_argument('--use_lora', default=False, action='store_true')
    parser.add_argument('--lora_r', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--lora_dropout', type=float, default=0.05)
    parser.add_argument('--lora_targets', type=str, default='all')

    args = parser.parse_args()
    
    if args.max_len == -1:
        if args.context_format == '' and args.target_format == 'findings_legacy':
            args.max_len = 160
        elif args.context_format in ['', 'history+indication'] and args.target_format in ['impression+findings', 'findings+impression']:
            args.max_len = 192
        else:
            raise NotImplementedError(f"Can't automatically determine args.max_len")

    if args.curriculum_learning_progress:
        assert args.curriculum_learning_progress >= 0 and args.curriculum_learning_progress < 1
    return args


# Function to save important checkpoints => in "state".
def save_checkpoint(state, is_best, is_best_loss, is_best_average, checkpoint, filename='last_checkpoint.pth.tar'):

    # Combine the directory path and the filename into a full file path =>"filepath". 
    # This function takes care of inserting the correct path separator ( / or \) between the parts.
    filepath = os.path.join(checkpoint, filename)

    torch.save(state, filepath)

    # Save the best checkpoint by accuracy meaning meteor value.
    if is_best:
        # shutil.copyfile(filepath, os.path.join(checkpoint, 'best_checkpoint.pth.tar'))
        torch.save(state, os.path.join(checkpoint, 'best_checkpoint.pth.tar'))
    # Save the best checkpoint by loss.
    # if is_best_loss:
    #     # shutil.copyfile(filepath, os.path.join(checkpoint, 'best_loss_checkpoint.pth.tar'))
    #     torch.save(state, os.path.join(checkpoint, 'best_loss_checkpoint.pth.tar'))
    if is_best_average:
        torch.save(state, os.path.join(checkpoint, 'best_average_checkpoint.pth.tar'))


# Use a writer to be used to plot the values of the metrics.
def plot_metrics(writer, metrics, epoch, prefix):

    # "metrics" is a dictionary: "metric_name" - key and "metric_value" - value.
    for metric_name, metric_value in metrics.items():
        # The isinstance() function returns True if the specified object is of the specified type, otherwise False.
        if isinstance(metric_value, list):
            continue
        # In order to plot the values we neet a dictionary/xy coordinates.
        elif isinstance(metric_value, dict):
            plot_metrics(writer, metric_value, epoch, f'{prefix}/{metric_name}')
            # for suffix, value in metric_value.items():
            #     writer.add_scalar(f'{prefix}/{metric_name}/{suffix}', value, epoch)
        else:
            # TODO
            writer.add_scalar(f'{prefix}/{metric_name}', metric_value, epoch)


def main():
    
    # Call the arguments.
    args = _args()

    # Method that creates a directory recursively.
    # If the specified directory already exists and value is set to False an OSError is raised, else not.
    os.makedirs(args.out, exist_ok=True)

    if args.mode == 'train':
        assert not os.path.isfile(os.path.join(args.out, args.resume_ckpt))

    # Configuration for the logging system.
    # "--log_level" - the importance of the error that will be displayed in the logger.
    # "level=args.log_level" is therefore like a threshold.
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%d/%m/%Y %H:%M:%S",
                        level=args.log_level, filename=os.path.join(args.out, f'logfile_{args.mode}.log'))
    
    # Create a summary writer with "--out" argument folder name.
    writer = SummaryWriter(args.out)
    logger.info(f'Args: {args}')
    
    logger.info(f'CPU count: {os.cpu_count()}')
    logger.info(f'Current device: {torch.cuda.current_device()} ({torch.cuda.device_count()} available devices)')
    logger.info(f'Current device name: {torch.cuda.get_device_name(torch.cuda.current_device())}')

    # Initialize dataloaders.
    train_dataloader, evaltrain_dataloader, valid_dataloader, test_dataloader = get_dataloaders(args)
    logger.info(f'Datasets created successfully.')


    # GIT_cls 
    if args.model_variation == 'with_classification':
        args.labels_weights = train_dataloader.dataset.labels_weights
        model = GIT_Model_with_Classification(args).to(args.device)
        print('Using GIT_Model_with_Classification')
    # GIT classic - put it on device.
    else:
        model = GIT_Model(args).to(args.device)
        print('Using GIT_Model')
    logger.info(f'Model initialized successfully.')

    log_model_parameters_summary(model)

    # Use LoRA.
    if args.use_lora is True:
        from models.LoRA import add_lora_to_model
        model = add_lora_to_model(model, args)
        logger.info('Added LoRA to model.')
        model = model.to(args.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    start_epoch = 0
    best_metric = -1
    best_epoch = 0
    best_loss = np.inf
    best_loss_epoch = 0
    best_average = 0
    best_average_epoch = 0

    # Evaluate or Resume the model:
    if args.mode in ['eval', 'resume']:
        # The checkpoint file path is the joined paths of the output file,
        # and the resumed checkpoint file.
        ckpt_file = os.path.join(args.out, args.resume_ckpt)
        # Check if file exists.
        assert os.path.isfile(ckpt_file)
        # checkpoint = torch.load(ckpt_file, map_location='cpu')
        # Load the checkpoint file and all the best results.
        checkpoint = torch.load(ckpt_file)
        last_metric = checkpoint['metric']
        best_metric = checkpoint.get('best_metric', last_metric)
        last_epoch = checkpoint['epoch']
        best_epoch = checkpoint.get('best_epoch', last_epoch)
        best_loss = checkpoint.get('best_loss', np.inf)
        best_loss_epoch = checkpoint.get('best_loss_epoch', 0)
        best_average = checkpoint.get('best_average', 0)
        best_average_epoch = checkpoint.get('best_average_epoch', 0)
        start_epoch = last_epoch + 1
        # The learnable parameters (i.e. weights and biases) of an torch.nn.Module model
        # are contained in the model’s parameters (accessed with model.parameters()).
        # A state_dict is simply a Python dictionary object that maps each layer to its parameter tensor.
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info(f'Loaded checkpoint from epoch {last_epoch}')

    # Evaluation only.
    if args.mode == 'eval':
        metrics = eval_epoch(model, test_dataloader, args, last_epoch)
        return

    # Training.
    # Pentru fiecare epoca de antrenare:
    for epoch in range(start_epoch, args.epochs):
        train_loss = train_epoch(model, train_dataloader, optimizer, args, epoch,
                                 max_epoch_samples=args.max_epoch_samples)
        writer.add_scalar('train/train_loss', train_loss, epoch)
        
        # validate once every other <args.validation_frequency> epochs, otherwise skip to the next training epoch
        if epoch % args.validation_frequency != args.validation_frequency - 1:
            continue

        # eval on a subset of the train set
        metrics = eval_epoch(model, evaltrain_dataloader, args, epoch)
        plot_metrics(writer, metrics, epoch, 'evaltrain')

        # eval on the validation set
        metrics = eval_epoch(model, valid_dataloader, args, epoch)
        plot_metrics(writer, metrics, epoch, 'valid')

        metric = metrics.get('meteor', {}).get('meteor', -1)
        loss = metrics.get('loss', -1)
        # compute an average score, used for selecting the best checkpoint
        average = 0.25 * metrics.get('meteor', {}).get('meteor', 0) \
                + 0.25 * metrics.get('rouge', {}).get('rougeL', 0) \
                + 0.125 * metrics.get('bleu1', {}).get('bleu', 0) \
                + 0.125 * metrics.get('bleu2', {}).get('bleu', 0) \
                + 0.125 * metrics.get('bleu3', {}).get('bleu', 0) \
                + 0.125 * metrics.get('bleu4', {}).get('bleu', 0)
        is_best = metric > best_metric
        is_best_loss = loss < best_loss
        is_best_average = average > best_average
        if is_best:
            best_metric = metric
            best_epoch = epoch
        if is_best_loss:
            best_loss = loss
            best_loss_epoch = epoch
        if is_best_average:
            best_average = average
            best_average_epoch = epoch
        # write tensorboard summaries
        writer.add_scalar('train/best_metric', best_metric, epoch)
        writer.add_scalar('train/best_epoch', best_epoch, epoch)
        writer.add_scalar('train/start_epoch', start_epoch, epoch)
        writer.add_scalar('train/best_loss', best_loss, epoch)
        writer.add_scalar('train/best_loss_epoch', best_loss_epoch, epoch)
        writer.add_scalar('train/best_average', best_average, epoch)
        writer.add_scalar('train/best_average_epoch', best_average_epoch, epoch)
            
        print(f'Current epoch {epoch} with metric {metric} and loss {loss} and average {average}')
        print(f'Best epoch {best_epoch} with metric {best_metric}')
        print(f'Best loss epoch {best_loss_epoch} with loss {best_loss}')
        print(f'Best average epoch {best_average_epoch} with average {best_average}')
        
        # save checkpoint
        save_checkpoint({
            'epoch': epoch,
            'best_epoch': best_epoch,
            'best_loss_epoch': best_loss_epoch,
            'metric': metric,
            'best_metric': best_metric,
            'loss': loss,
            'best_loss': best_loss,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'average': average,
            'best_average': best_average,
            'best_average_epoch': best_average_epoch,
        }, is_best, is_best_loss, is_best_average, args.out)

        # use early stopping
        if args.patience and (epoch - best_epoch >= args.patience) and (epoch - best_loss_epoch >= args.patience) and (epoch - best_average_epoch >= args.patience):
            break


if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    logger.info(f'Finished in {(end_time - start_time) / 60} minutes.')
