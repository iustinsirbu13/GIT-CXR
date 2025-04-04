import json
import torch
import os
import evaluate
import pandas as pd
from tqdm import tqdm
from functools import partial
import time
import logging


logger = logging.getLogger(__name__)


def hf_score(name, **kwargw):
    fn = evaluate.load(name)
    return fn.compute(**kwargw)


METRICS = {
    'bleu1': partial(hf_score, name='bleu', max_order=1),
    'bleu2': partial(hf_score, name='bleu', max_order=2),
    'bleu3': partial(hf_score, name='bleu', max_order=3),
    'bleu4': partial(hf_score, name='bleu', max_order=4),
    'rouge': partial(hf_score, name='rouge'),
    'meteor': partial(hf_score, name='meteor'),
}


def remove_context_from_text(text, context):
    idx = len(context)
    return text[idx:].strip()


def _preds_list_to_df(texts, preds, cols, text_col='Report Impression'):
    data = {text_col: texts}
    for i, col in enumerate(cols):
        data[col] = [elem[i] for elem in preds]
        
    logger.info(f'lens {len(texts), len(preds), len(cols)}')
    # logger.info(f'{data}')
    return pd.DataFrame(data)


def eval_epoch(model, dataloader, args, epoch, max_samples=None):

    model.eval()
    # all_ids = []
    all_target_texts_full = []
    all_paths = []
    all_generated_texts_full = []
    all_contexts = []
    total_loss = 0
    original_loss = 0
    aux_loss = 0
    all_clf_outputs = []
    all_clf_predictions = []
    all_clf_labels = []
    
    with torch.no_grad():

        for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            if idx == 0:
                logger.info(f'Batch with idx==0: {batch}')

            if max_samples and (idx >= max_samples):
                break

            pixel_values = batch["pixel_values"].to(args.device)
            input_ids = batch["input_ids"].to(args.device)
            prompt_input_ids = batch["prompt_input_ids"].to(args.device)
            attention_mask = batch["attention_mask"].to(args.device)
            labels = (input_ids * attention_mask + (-100) * (1 - attention_mask)).to(args.device)
            clf_labels = batch["clf_labels"].to(args.device) if args.model_variation == 'with_classification' else None

            inputs_dict = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'pixel_values': pixel_values,
                'labels': labels
            }

            if args.model_variation == 'with_classification':
                inputs_dict['clf_labels'] = clf_labels

            outputs = model(**inputs_dict)

            total_loss += outputs.loss.item()

            if idx == 0:
                try:
                    logger.info(f'Outputs for batch 0: {outputs}')
                    logger.info(f'Outputs keys: {outputs.to_dict()}')
                    logger.info(f'Outputs keys: {outputs.to_dict().keys()}')
                except Exception as e:
                    logger.error(e)

            if 'clf_loss' in outputs.keys():
                original_loss += outputs.original_loss.item()
                aux_loss += outputs.clf_loss.item()
                all_clf_outputs += outputs.clf_outputs.tolist()
                all_clf_predictions += outputs.clf_predictions.tolist()
                all_clf_labels += [e.tolist() for e in clf_labels]
                if idx == 0:
                    logger.info(f'CLF outputs:{outputs.clf_outputs.shape} {outputs.clf_outputs}')
                    logger.info(f'CLF predictions: {outputs.clf_predictions.shape} {outputs.clf_predictions}')
                    logger.info(f'CLF labels: {[e.tolist() for e in clf_labels]}')

            generated_ids = model.generate(pixel_values=pixel_values, input_ids=prompt_input_ids, max_length=args.max_len)
            generated_texts = dataloader.dataset.processor.batch_decode(generated_ids, skip_special_tokens=True)
            # print(generated_texts)
            all_target_texts_full += batch['text']
            all_paths += batch['path']
            all_contexts += batch['context']
            # all_ids.append(generated_ids.flatten())
            all_generated_texts_full += generated_texts

    all_target_texts = [remove_context_from_text(text, context) for text, context in zip(all_target_texts_full, all_contexts)]
    all_generated_texts = [remove_context_from_text(text, context) for text, context in zip(all_generated_texts_full, all_contexts)]
    all_target_texts_lists = [[text] for text in all_target_texts]

    # all_ids = torch.stack(all_ids)
    # all_generated_texts = dataloader.dataset.processor.batch_decode(all_ids, skip_special_tokens=True)
    
    # metrics = evaluate.combine(["meteor",])
    # try:
        # scores = metrics.compute(predictions=all_generated_texts, references=all_target_texts_lists)
    # except:
    #     scores = {}

    scores = {
        'num_samples': len(all_generated_texts),
        'loss': total_loss / len(all_generated_texts),
        'original_loss': original_loss / len(all_generated_texts),
        'aux_loss': aux_loss / len(all_generated_texts),
    }
    for metric_name, metric_fn in METRICS.items():
        scores[metric_name] = metric_fn(predictions=all_generated_texts, references=all_target_texts_lists)

    # eval f1

    os.makedirs(os.path.join(args.out, 'scores'), exist_ok=True)
    os.makedirs(os.path.join(args.out, 'predictions'), exist_ok=True)

    with open(os.path.join(args.out, 'scores', f'{dataloader.dataset.mode}_{epoch}.json'), 'w') as fp:
        json.dump(scores, fp)
    
    df = pd.DataFrame({
        'path': all_paths,
        'target': all_target_texts,
        'prediction': all_generated_texts,
        'context': all_contexts,
        'target_full': all_target_texts_full,
        'prediction_full': all_generated_texts_full,
    })
    df.to_csv(os.path.join(args.out, 'predictions', f'{dataloader.dataset.mode}_{epoch}.csv'), index=False)
    
    if args.model_variation == 'with_classification':
        try:
            clf_scores = {
                'clf_labels': all_clf_labels,
                'clf_predictions': all_clf_predictions,
                'clf_outputs': all_clf_outputs,
                'labels_columns': dataloader.dataset.labels_columns,
            }
            # logger.info(f'Clf scores: {clf_scores}')
            with open(os.path.join(args.out, 'predictions', f'{dataloader.dataset.mode}_{epoch}_clf.json'), 'w') as fp:
                json.dump(clf_scores, fp)
        except Exception as e:
            logger.error(e)
        try:
            df_preds = _preds_list_to_df(all_target_texts, all_clf_predictions, dataloader.dataset.labels_columns)
            df_labels = _preds_list_to_df(all_target_texts, all_clf_labels, dataloader.dataset.labels_columns)
            df_preds.to_csv(os.path.join(args.out, 'predictions', f'{dataloader.dataset.mode}_{epoch}_clf_predictions.csv'), index=False)
            df_labels.to_csv(os.path.join(args.out, 'predictions', f'{dataloader.dataset.mode}_{epoch}_clf_labels.csv'), index=False)
        except Exception as e:
            logger.error(e)
    return scores
