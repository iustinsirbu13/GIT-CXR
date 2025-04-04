import json
import os
import evaluate
import pandas as pd
from tqdm import tqdm
from functools import partial
import time
import argparse
import re


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


def eval_metrics(args):
    
    df = pd.read_csv(args.input_path)

    all_generated_texts = df[args.prediction_column].tolist()
    _all_target_texts = df[args.target_column].tolist()
    all_target_texts_lists = [[x] for x in _all_target_texts]

    assert len(all_generated_texts) == len(all_target_texts_lists)
    print('Example of processed target:', all_target_texts_lists[0])
    print('Example of processed prediction:', all_generated_texts[0])

    scores = {
        'num_samples': len(all_generated_texts),
    }
    for metric_name, metric_fn in METRICS.items():
        scores[metric_name] = metric_fn(predictions=all_generated_texts, references=all_target_texts_lists)

    with open(args.output_path, 'w') as fp:
        json.dump(scores, fp)

    return scores


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Compute metrics using the pretrained bert predictions.')
    
    parser.add_argument('--input_path', required=True, type=str)
    parser.add_argument('--output_path', required=True, type=str)
    parser.add_argument('--target_column', default='target', type=str)
    parser.add_argument('--prediction_column', default='prediction', type=str)
    args = parser.parse_args()

    eval_metrics(args)
