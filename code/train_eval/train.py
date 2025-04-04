
from tqdm import tqdm
import logging
import torch

logger = logging.getLogger(__name__)


"""Trains the model for one epoch using the train_dataloader.
    Args:
        - model: the model
        - train_dataloader: the train dataloader
        - optimizer: the optimizer 
        - args: the main arguments
        - epoch: the current epoch
        - max_epoch_samples: faster evaluation - only a definite number of samples per epoch

    Returns:

"""
def train_epoch(model, train_dataloader, optimizer, args, epoch, max_epoch_samples=None):
    model.train()
    
    if args.curriculum_learning != 'none':
        cl_progress = args.curriculum_learning_progress or epoch/args.epochs
        train_dataloader.dataset.set_curriculum_learning_progress(cl_progress)

        if (args.curriculum_learning_cls_method == 'update_labels_weights') and (args.model_variation == 'with_classification'):
            model.update_labels_weights(train_dataloader.dataset.labels_weights)

    epoch_loss = 0
    # In practice we need batches of the dataset.
    num_batches = len(train_dataloader)
    # Progress bar.
    p_bar = tqdm(range(num_batches))

    # For every batch we also want the index => enumerate.
    for idx, batch in enumerate(train_dataloader):

        # TODO - we don't use max_epoch_samples now.
        if max_epoch_samples is not None and idx * args.batch_size > max_epoch_samples:
            num_batches = idx
            break

        input_ids = batch["input_ids"].to(args.device)
        pixel_values = batch["pixel_values"].to(args.device)
        attention_mask = batch["attention_mask"].to(args.device)
        labels = (input_ids * attention_mask + (-100) * (1 - attention_mask)).to(args.device)

        clf_labels = batch["clf_labels"].to(args.device) if args.model_variation == 'with_classification' else None

        if idx == 0:
            logger.info(f'pixel_values shape {pixel_values.shape}')
            logger.info(f'input_ids shape {input_ids.shape}')
            logger.info(f'attention_mask shape {attention_mask.shape} and sum {attention_mask.sum()}')
            logger.info(f'labels shape {labels.shape}')
            logger.info(f'labels == inpu_ids : {(labels == input_ids).sum()}')
            logger.info(f'labels == -100 : {(labels == -100).sum()}')

        inputs_dict = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'pixel_values': pixel_values,
            'labels': labels
        }
        
        if args.model_variation == 'with_classification':
            inputs_dict['clf_labels'] = clf_labels

        outputs = model(**inputs_dict)

        loss = outputs.loss
        loss_value = loss.item()
        epoch_loss += loss_value
        
        if idx == 0 and args.labels_path:
            logger.info(f'original_loss: {outputs.get("original_loss")}')
            logger.info(f'clf_loss: {outputs.get("clf_loss")}')
            logger.info(f'loss: {outputs.loss}')

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        p_bar.set_description(f'Epoch {epoch}, loss {loss_value}, mean_loss {epoch_loss/(idx+1)}')
        p_bar.update()

    epoch_loss /= num_batches
    print('Train loss:', epoch_loss)
    return epoch_loss
