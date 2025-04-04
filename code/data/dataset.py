import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from PIL import Image
import os
from transformers import AutoProcessor
import logging
import re
import random
import torchvision
from functools import cached_property


logger = logging.getLogger(__name__)

def process_text(text):
    if pd.isna(text):
        return ''
    text = text.lower()
    for c in [':', ';']:
        text = text.replace(c, '.')

    # replace numbers by _
    text = re.sub('\d+\.?\d*', '_', text)
    # replace extra characters
    text = re.sub('[^a-z .,_/]', ' ', text)
    # replace __ by _
    text = re.sub('_+', '_', text)
    text = re.sub('/+', '/', text)
    text = re.sub('\.+', '.', text)
    text = re.sub(',+', ',', text)
    text = text.replace('/', ' / ')
    # replace consecutive spaces by one space
    text = re.sub(' +', ' ', text)
    return text.strip()


def create_target(row, target_format='impression+findings', only_header=False):
    if target_format == 'impression+findings':
        if only_header:
            return 'impression : '
        # impression = '' if pd.isna(row.impression) else row.impression
        impression = process_text(row.impression)
        # findings = '' if pd.isna(row.findings) else row.findings
        findings = process_text(row.findings)
        if impression == '' and findings == '':
            return None
        return f'impression : {impression} findings : {findings}'
    elif target_format == 'findings+impression':
        if only_header:
            return 'findings : '
        impression = process_text(row.impression)
        findings = process_text(row.findings)
        if impression == '' and findings == '':
            return None
        return f'findings : {findings} impression : {impression}'
    elif target_format == 'findings':
        if only_header:
            return 'findings : '
        findings = process_text(row.findings)
        if findings == '':
            return None
        return f'findings : {findings}'
    elif target_format == 'impression':
        if only_header:
            return 'impression : '
        impression = process_text(row.impression)
        if impression == '':
            return None
        return f'impression : {impression}'
    elif target_format == 'findings_legacy':
        if only_header:
            return ''
        return row.findings
    else:
        raise NotImplementedError(f'Invalid target_format {target_format}')


def create_context(row, context_format='history+indication'):
    context_columns = context_format.split('+') if '+' in context_format else []
    texts = []
    for col in context_columns:
        texts.append(process_text(row[col]))
    return ' '.join(texts).strip()


def _compute_correction_weights(df, cols, progress_value):
    df = df.copy()
    # df['cl_weight'] = df['TARGET_LEN_QCUT10'].map(lambda x: 1 / (1 + abs(x - progress_value)))

    dfg = df.groupby(by='TARGET_LEN_QCUT10', as_index=True).agg({c: (lambda x: x.mean()) for c in cols})
    q_weight = dfg.index.map(lambda x: 1 / (1 + abs(x - progress_value))).values
    q_weight = q_weight.reshape(-1, 1) / q_weight.sum() # the normalized weight of each bin
    actual_frequency_per_disease = df[cols].describe().loc['mean']
    sampling_frequency_per_disease = (dfg * q_weight).sum(axis=0)
    dis_weights = actual_frequency_per_disease / sampling_frequency_per_disease # frequency of a disease over the expected frequency of it in the sampled 

    df['dis_weight'] = 0.0
    df['dis_weight_count'] = 0
    for col in cols:
        df.loc[df[col] == 1, 'dis_weight'] += dis_weights[col]
        df.loc[df[col] == 1, 'dis_weight_count'] += 1
    df['dis_weight'] = (df['dis_weight'] / df['dis_weight_count']).fillna(1)
    df['orig_cl_weight'] = df['cl_weight']
    df['cl_weight'] = df['dis_weight'] * df['orig_cl_weight']

    logger.info(f'add_correction_weights with factors \n{dis_weights}')
    return df


class MimicDataset(Dataset):
    def __init__(self, csv_path, splits_path, findings_path, images_path, processor, args, mode='train', use_p=['p10', 'p11', 'p12', 'p13', 'p14', 'p15', 'p16', 'p17', 'p18', 'p19']):
        super().__init__()
        self.images_path = images_path
        self.processor = processor
        self.mode = mode
        self.args = args
        
        df = pd.read_csv(csv_path)
        df['p'] = df['path'].map(lambda elem: elem.split('/')[1])
        self.df = df.loc[df['p'].isin(use_p)]
        # self.df = df

        logger.info(f'Dataset size from {csv_path}: {self.df.shape}')
        
        df_splits = pd.read_csv(splits_path)
        self.df = self.df.merge(df_splits, how='inner', on=['dicom_id', 'study_id', 'subject_id'])
        
        logger.info(f'Dataset size after adding split column: {self.df.shape}')

        if mode == 'train' or mode == 'evaltrain':
            self.df = self.df.loc[self.df['split'] == 'train']
        elif mode == 'valid':
            self.df = self.df.loc[self.df['split'] == 'validate']
        elif mode == 'test':
            self.df = self.df.loc[self.df['split'] == 'test']
        else:
            raise NotImplementedError()
        
        logger.info(f'Dataset size after filtering the split {mode}: {self.df.shape}')
        
        # logger.info('Filtering PA and AP images using csv with images metadata')
        if args.data_variation == 'single_view':
            metadata_df = pd.read_csv(args.metadata_path, usecols=['dicom_id', 'ViewPosition'], low_memory=False)
            metadata_df = metadata_df.loc[metadata_df['ViewPosition'].isin(['AP', 'PA'])]
        elif args.data_variation in ['multi_view', 'multi_view_temporal']:
            metadata_df = pd.read_csv(args.metadata_path, usecols=['dicom_id', 'study_id', 'ViewPosition'], low_memory=False)
            _ap_pa_studies = metadata_df.loc[metadata_df['ViewPosition'].isin(['AP', 'PA']), 'study_id']
            metadata_df = metadata_df.loc[metadata_df['study_id'].isin(_ap_pa_studies), ['dicom_id', 'ViewPosition']]
        else:
            raise NotImplementedError('data_variation not available')
        self.df = self.df.merge(metadata_df, how='inner', on=['dicom_id'])

        logger.info(f'Dataset size after filtering AP and PA positions: {self.df.shape}')

        findings_df = pd.read_csv(findings_path, usecols=['study_id', 'findings', 'impression', 'history', 'indication'])
        findings_df['study_id'] = findings_df['study_id'].map(lambda row: int(row[1:]))
        findings_df['CONTEXT'] = findings_df.apply(lambda row: create_context(row, args.context_format), axis=1)
        findings_df['TARGET'] = findings_df.apply(lambda row: create_target(row, args.target_format), axis=1)
        findings_df['TARGET_HEADER'] = findings_df.apply(lambda row: create_target(row, args.target_format, only_header=True), axis=1)
        logger.info(f'findings_df shape initial {findings_df.shape}')
        findings_df.dropna(subset=['TARGET'], inplace=True)
        logger.info(f'findings_df shape after dropna {findings_df.shape}')

        if args.labels_path:
            logger.info(f'Adding clf labels from args.labels_path={args.labels_path}')
            labels_df = pd.read_csv(args.labels_path)
            labels_df, self.num_labels = self._process_labels(labels_df, drop_columns=['subject_id'], num_labels=args.num_labels)

            self.labels_columns = labels_df.columns.tolist()[1:]
            assert 'study_id' not in self.labels_columns, f'cols: {self.labels_columns}'
            # self.labels_columns_encodings = self.processor(text=self.labels_columns, padding='do_not_pad', add_special_tokens=False)
            # logger.info(f'labels_columns_encodings: {self.labels_columns_encodings}')
            assert self.labels_columns == 'Atelectasis,Cardiomegaly,Consolidation,Edema,Enlarged Cardiomediastinum,Fracture,Lung Lesion,Lung Opacity,No Finding,Pleural Effusion,Pleural Other,Pneumonia,Pneumothorax,Support Devices'.split(',')

            self.labels_weights = self._compute_label_weights(labels_df, self.labels_columns, self.num_labels)
            findings_df = findings_df.merge(labels_df, how='inner', on='study_id')
            logger.info(f'findings_df shape after adding labels {findings_df.shape}')

        
        self.findings_df = findings_df
        logger.info(f'findings_df shape final {self.findings_df.shape}')

        logger.info(f'Unique study_id in self.df: {self.df.study_id.unique().shape}')
        logger.info(f'Unique study_id in self.findings_df: {self.findings_df.study_id.unique().shape}')
        logger.info(f'self.df study_id in self.findings_df: {self.df.study_id.isin(self.findings_df.study_id).value_counts()}')
        logger.info(f'self.df.columns={self.df.columns}')
        logger.info(f'self.findings_df.columns={self.findings_df.columns}')
        
        self.df = self.df.merge(self.findings_df, how='inner', on=['study_id'])
        self.df['jpg_path'] = self.df['path'].map(lambda row: row.replace('files/', '').replace('.dcm', '.jpg'))
        # self.df = self.df.loc[~self.df['path'].isin(to_remove)]

        logger.info(f'Dataset size after filtering findings: {self.df.shape}')

        if mode == 'train' and args.max_train_samples:
            logger.info(f'args.max_train_samples=={args.max_train_samples}')
            self.df = self.df.iloc[:args.max_train_samples].copy()

        elif mode == 'evaltrain' and args.max_evaltrain_samples:
            logger.info(f'args.max_evaltrain_samples=={args.max_evaltrain_samples}')
            self.df = self.df.iloc[:args.max_evaltrain_samples].copy()

        elif mode == 'valid' and args.max_valid_samples:
            logger.info(f'args.max_valid_samples=={args.max_valid_samples}')
            self.df = self.df.iloc[:args.max_valid_samples].copy()

        elif mode == 'test' and args.max_test_samples:
            logger.info(f'args.max_test_samples=={args.max_test_samples}')
            self.df = self.df.iloc[:args.max_test_samples].copy()

        self.unique_studies = self.df.study_id.unique()
        logger.info(f'Created dataset in mode {mode} with size {len(self.df)} ({len(self.unique_studies)} unique studies)')

        if args.curriculum_learning != 'none':
            self.df['TARGET_LEN'] = self.df['TARGET'].map(
                lambda x: len(processor(text=[x], padding='do_not_pad', add_special_tokens=False, max_length=self.args.max_len, truncation=True)['input_ids'][0]))
            self.df['TARGET_LEN_QCUT10'] = pd.qcut(self.df['TARGET_LEN'], 10, labels=range(10)).astype(int)
            self.full_df = self.df
            self.full_unique_studies = self.unique_studies
            self.full_labels_weights = self.labels_weights
        
        os.makedirs(os.path.join(args.out, 'dataset'), exist_ok=True)
        self.df.to_csv(os.path.join(args.out, 'dataset', f'{mode}.csv'))

        self.error_indexes = set()
        self.max_allowed_errors = 2

    def set_curriculum_learning_progress(self, progress):
        if progress is None:
            self.df = self.full_df
            self.unique_studies = self.full_unique_studies

        elif self.args.data_variation == 'single_view':
            assert progress >= 0 and progress <=1
            value = int(progress * 10)
            self.df = self.full_df.copy()
            if self.args.curriculum_learning == 'linear':
                self.df['cl_weight'] = self.df['TARGET_LEN_QCUT10'].map(lambda x: 1 / (1 + abs(x - value)))
            elif self.args.curriculum_learning == 'exponential':
                self.df['cl_weight'] = self.df['TARGET_LEN_QCUT10'].map(lambda x: 1 / (2 ** abs(x - value)))

            if self.args.curriculum_learning_cls_method == 'add_correction_weights':
                self.df = _compute_correction_weights(self.df, self.labels_columns, value)

            self.df = self.df.sample(frac=self.args.curriculum_learning_percent, replace=False, weights='cl_weight')
            self.unique_studies = self.df.study_id.unique()
            logger.info(f'Set curriculum learning for SV with value {value} and sampled {len(self.df)} examples with qcut10 distribution \n{self.df.TARGET_LEN_QCUT10.value_counts(sort=False)}')

            if  self.args.curriculum_learning_cls_method == 'update_labels_weights':
                self.labels_weights = self._compute_label_weights(self.df, self.labels_columns, self.num_labels)
                logger.info(f'update_labels_weights to \n{self.labels_weights}')

        elif self.args.data_variation == 'multi_view_temporal':
            assert progress >= 0 and progress <=1
            value = int(progress * 10)
            self.df = self.full_df.copy()
            self.df.drop_duplicates(subset=['study_id'], inplace=True)
            if self.args.curriculum_learning == 'linear':
                self.df['cl_weight'] = self.df['TARGET_LEN_QCUT10'].map(lambda x: 1 / (1 + abs(x - value)))
            elif self.args.curriculum_learning == 'exponential':
                self.df['cl_weight'] = self.df['TARGET_LEN_QCUT10'].map(lambda x: 1 / (2 ** abs(x - value)))

            if self.args.curriculum_learning_cls_method == 'add_correction_weights':
                self.df = _compute_correction_weights(self.df, self.labels_columns, value)

            self.df = self.df.sample(frac=self.args.curriculum_learning_percent, replace=False, weights='cl_weight')
            self.unique_studies = self.df.study_id.unique()
            logger.info(f'Set curriculum learning for MV with value {value} and sampled {len(self.df)} examples with qcut10 distribution \n{self.df.TARGET_LEN_QCUT10.value_counts(sort=False)}')
            
            if  self.args.curriculum_learning_cls_method == 'update_labels_weights':
                self.labels_weights = self._compute_label_weights(self.df, self.labels_columns, self.num_labels)
                logger.info(f'update_labels_weights to \n{self.labels_weights}')
            
            self.df = self.full_df
        else:
            raise NotImplementedError()

    def __len__(self):
        if self.args.data_variation == 'single_view':
            return len(self.df)
        elif self.args.data_variation in ['multi_view', 'multi_view_temporal']:
            return len(self.unique_studies)
        else:
            raise NotImplementedError('data_variation not available')
    
    
    def _process_labels(self, df, labels5=False, drop_columns=['Report Impression'], num_labels=-1):
        if drop_columns:
            df = df.drop(columns=drop_columns)
        if labels5:
            df = df[['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']]
        if num_labels == 2:
            df = df.replace({-1.0: 0.0})
            df = df.fillna(0.0)
            num_labels = 2
        else:
            df = df.replace({-1.0: 2.0})
            df = df.fillna(3.0)
            num_labels = 4
        df = df.astype(int)
        return df, num_labels
    
    def _compute_label_weights(self, df, columns, num_labels, eps=1e-7):
        weights = []
        for col in columns:
            values = df[col].value_counts().reindex(range(num_labels), fill_value=0).values
            normalized_values = (values + eps).max() / (values + eps)
            weights.append(normalized_values.tolist())
        return weights
    
    def _encode_prompt(self, prompt_text):
        encoding_prompt = self.processor(
            text=prompt_text,
            padding=False,
            max_length=self.args.max_len,
            truncation=True,
            add_special_tokens=False
        )
        prompt_input_ids = [self.processor.tokenizer.cls_token_id] + encoding_prompt.input_ids
        return {
            'prompt_input_ids': torch.tensor(prompt_input_ids).unsqueeze(0)
        }

    def _sample_image(self, rows):
        if self.mode == 'train':
            return rows.sample(n=1).jpg_path.iloc[0]
        else:
            return rows.jpg_path.iloc[0]

    def _read_mv_images(self, df, study_id):
        rows = df.loc[df.study_id == study_id]
        assert len(rows) > 0

        img_path1 = self._sample_image(rows.loc[rows['ViewPosition'].isin(['AP', 'PA'])])

        rows_lat = rows.loc[~rows['ViewPosition'].isin(['AP', 'PA'])]
        if len(rows_lat) > 0:
            img_path2 = self._sample_image(rows_lat)
        else:
            rows_other = rows.loc[rows['jpg_path'] != img_path1]
            if len(rows_other) > 0:
                img_path2 = self._sample_image(rows_other)
            else:
                img_path2 = img_path1
        image_paths = [img_path1, img_path2]
        # random.shuffle(image_paths)

        row = rows.loc[rows['jpg_path'] == img_path1].iloc[0]
        return image_paths, row

    def _read_sv_images(self, df, index):
        row = df.iloc[index]
        image_path = row['jpg_path']
        return [image_path], row

    def _scale_image(self, image):
        image = torchvision.transforms.functional.pil_to_tensor(image)
        image = torchvision.transforms.Resize(int(self.args.img_size * self.args.img_scale_factor))(image)
        if self.mode == 'train':
            image = torchvision.transforms.RandomCrop(self.args.img_size)(image)
        else:
            image = torchvision.transforms.CenterCrop(self.args.img_size)(image)
        image = torchvision.transforms.functional.to_pil_image(image)
        return image

    def __getitem__(self, index):
                
        if self.args.data_variation == 'single_view':
            image_paths, row = self._read_sv_images(self.df, index)
        elif self.args.data_variation in ['multi_view', 'multi_view_temporal']:
            study_id = self.unique_studies[index]
            image_paths, row = self._read_mv_images(self.df, study_id)
            
        else:
            raise NotImplementedError('data_variation not available')

        image_paths = [os.path.join(self.images_path, jpg_path) for jpg_path in image_paths]
        try:
            images = [Image.open(image_path) for image_path in image_paths]
        except Exception as e:
            logger.warning(f"Error while reading image {row['jpg_path']}.")
            logger.warning(f'{e}')
            self.error_indexes.add(index)
            if len(self.error_indexes) <= self.max_allowed_errors:
                # return self.__getitem__(index + 1)
                return self.__getitem__(random.randint(0, self.__len__()-1))
            else:
                raise e

        if self.args.img_scale_factor != 1:
            assert self.args.img_scale_factor > 1
            images = [self._scale_image(image) for image in images]

        # encoding = self.processor(images=image, text=text, padding="max_length", return_tensors="pt")
        tokenized_context = self.processor.tokenizer.tokenize(row['CONTEXT'], max_length=self.args.max_context_len, truncation=True)
        truncated_context = self.processor.tokenizer.convert_tokens_to_string(tokenized_context)

        text = ' '.join([truncated_context, row['TARGET']]).lstrip()
        prompt_text = ' '.join([truncated_context, row['TARGET_HEADER']]).lstrip()

        encoding_txt = self.processor(
            text=text,
            padding="max_length",
            max_length=self.args.max_len,
            truncation=True,
            return_tensors="pt"
        )
        encoding_img = self.processor(
            images=images,
            return_tensors="pt",
            size={'shortest_edge': self.args.img_size},
            crop_size=self.args.img_size
        )
        encoding = encoding_img | encoding_txt
        if self.mode != 'train':
            encoding = encoding | self._encode_prompt(prompt_text)
        # logger.debug(f'Encoding img: {encoding_img.keys()}')
        # logger.debug(f'Encoding txt: {encoding_txt.keys()}')
        # logger.debug(f'Encoding: {encoding.keys()}')
        encoding = {k:v.squeeze() for k,v in encoding.items()}

        if self.args.data_variation == 'multi_view':
            assert encoding['pixel_values'].shape == (self.args.num_views, 3, self.args.img_size, self.args.img_size)
            encoding['pixel_values'] = encoding['pixel_values'].reshape(-1, self.args.img_size, self.args.img_size)
        elif self.args.data_variation == 'multi_view_temporal':
            assert encoding['pixel_values'].shape == (self.args.num_views, 3, self.args.img_size, self.args.img_size)

        encoding['context'] = truncated_context
        encoding['text'] = text
        encoding['prompt_text'] = prompt_text
        encoding['path'] = row['path']
        if self.args.labels_path:
            encoding['clf_labels'] = torch.tensor(row[self.labels_columns].values.tolist())
        
        return encoding


def get_datasets(args):
    processor = AutoProcessor.from_pretrained(args.processor)
    
    train_dataset = MimicDataset(
        args.csv_path,
        args.splits_path,
        args.findings_path,
        args.images_path,
        processor,
        args,
        'train'
    )

    evaltrain_dataset = MimicDataset(
        args.csv_path,
        args.splits_path,
        args.findings_path,
        args.images_path,
        processor,
        args,
        'evaltrain'
    )
    
    valid_dataset = MimicDataset(
        args.csv_path,
        args.splits_path,
        args.findings_path,
        args.images_path,
        processor,
        args,
        'valid'
    )
    
    test_dataset = MimicDataset(
        args.csv_path,
        args.splits_path,
        args.findings_path,
        args.images_path,
        processor,
        args,
        'test'
    )

    return train_dataset, evaltrain_dataset, valid_dataset, test_dataset


def get_dataloaders(args):
    train_dataset, evaltrain_dataset, valid_dataset, test_dataset = get_datasets(args)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                    drop_last=True, num_workers=args.num_workers)
    evaltrain_dataloader = DataLoader(evaltrain_dataset, batch_size=1, shuffle=False,
                                    drop_last=False, num_workers=args.num_workers)
    valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False,
                                    drop_last=False, num_workers=args.num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                                    drop_last=False, num_workers=args.num_workers)

    return train_dataloader, evaltrain_dataloader, valid_dataloader, test_dataloader
