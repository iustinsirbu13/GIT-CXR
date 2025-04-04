#### GETTING THE GIT BASE AND GIT LARGE MODELS ####

from huggingface_hub import hf_hub_download

from transformers import AutoProcessor
from transformers import AutoModelForCausalLM
from transformers import GitConfig

import torch
import logging

logger = logging.getLogger(__name__)


class GIT_Model(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        if args.data_variation == 'single_view':
            self.model = self._init_singleview_model()
        elif args.data_variation == 'multi_view':
            self.model = self._init_multiview_model()
        elif args.data_variation == 'multi_view_temporal':
            self.model = self._init_multiview_temporal_model()
        else:
            raise NotImplementedError(f'data_variation={args.data_variation} is not available')
        
    def _init_multiview_temporal_model(self):
        args = self.args
        logger.info(f'Initializing multiview temporal model with num_views={args.num_views}, img_size={args.img_size}, patch_size={args.patch_size}')

        config = GitConfig.from_pretrained(args.model)
        config.vision_config.image_size = args.img_size
        config.vision_config.patch_size = args.patch_size
        config.num_image_with_embedding = args.num_views
        
        if args.use_pretrained == 'yes':
            logger.info('Loading PRETRAINED model weights.')
            model = AutoModelForCausalLM.from_pretrained(args.model, config=config)
        else:
            logger.info('Loading RANDOM model weights.')
            assert args.use_pretrained in ['legacy', 'no']
            model = AutoModelForCausalLM.from_config(config)
        return model

    def _init_multiview_model(self):
        args = self.args
        logger.info(f'Initializing multiview model with num_views={args.num_views}, img_size={args.img_size}, patch_size={args.patch_size}')

        config = GitConfig.from_pretrained(args.model)
        config.vision_config.image_size = args.img_size
        config.vision_config.patch_size = args.patch_size
        config.vision_config.num_channels = 3 * args.num_views
        
        assert args.use_pretrained in ['legacy', 'no']
        model = AutoModelForCausalLM.from_config(config)
        model.git.image_encoder.vision_model.embeddings.patch_embedding = torch.nn.Conv2d(
            config.vision_config.num_channels,
            config.vision_config.hidden_size,
            kernel_size=(args.patch_size, args.patch_size),
            stride=(args.patch_size, args.patch_size),
            bias=False,
            # groups=args.num_views
        )
        return model

    def _init_singleview_model(self):
        args = self.args

        config = GitConfig.from_pretrained(args.model)
        config.vision_config.image_size = args.img_size
        config.vision_config.patch_size = args.patch_size
        config.num_image_with_embedding = None

        if args.use_pretrained in ['yes', 'legacy']:
            logger.info(f'Initializing pretrained model from {args.model}')
            model = AutoModelForCausalLM.from_pretrained(args.model, config=config)
        else:
            assert args.use_pretrained == 'no'
            logger.info(f'Initializing model with custom image size {args.img_size} and patch_size {args.patch_size}')
            model = AutoModelForCausalLM.from_config(config)
        return model

    # All layers already in the huggingface model
    def forward(self, *argv, **kwargs):
        outputs = self.model(*argv, **kwargs)
        return outputs

    def generate(self, *argv, **kwargs):
        return self.model.generate(*argv, **kwargs)


class GIT_Model_with_Classification(GIT_Model):
    def __init__(self, args):
        super().__init__(args)
        assert args.model_variation == 'with_classification'
        self._init_cls_heads()

    def _init_cls_heads(self):
        args = self.args
        clf_heads = []
        clf_loss_functions = []

        in_features = 768*197*args.num_views if args.data_variation == 'multi_view_temporal' else 768*197

        for clf_weights in args.labels_weights:
            head = torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Dropout(p=0.2),
                torch.nn.Linear(in_features=in_features, out_features=len(clf_weights))
            )
            clf_heads.append(head)
            clf_loss_functions.append(
                torch.nn.CrossEntropyLoss(weight=torch.tensor(clf_weights).to(args.device))
            )

        self.clf_heads = torch.nn.ModuleList(clf_heads)
        self.clf_loss_functions = clf_loss_functions

    def _extract_visual_features(self, pixel_values):
        if pixel_values.ndim == 4:
            # here we assume pixel_values is of shape (batch_size, num_channels, height, width)
            encoded_image = self.model.git.image_encoder(pixel_values)
            visual_features = encoded_image.last_hidden_state
        elif pixel_values.ndim == 5:
            # here we assume pixel_values is of shape (batch_size, num_frames, num_channels, height, width)
            visual_features = []
            for frame_idx in range(pixel_values.shape[1]):
                visual_features_frame = self.model.git.image_encoder(pixel_values[:, frame_idx, :, :]).last_hidden_state
                visual_features_frame += self.model.git.img_temperal_embedding[frame_idx]
                visual_features.append(visual_features_frame)
            # finally, concatenate all features along sequence dimension
            visual_features = torch.cat(visual_features, dim=1)
        else:
            raise ValueError("pixel_values must be of rank 4 or 5")
        
        projected_visual_features = self.model.git.visual_projection(visual_features)
        return projected_visual_features

    def update_labels_weights(self, labels_weights):
        clf_loss_functions = []
        for old_ce in self.clf_loss_functions:
            old_ce.cpu()
        for clf_weights in labels_weights:
            clf_loss_functions.append(
                torch.nn.CrossEntropyLoss(weight=torch.tensor(clf_weights).to(self.args.device))
            )
        assert len(clf_loss_functions) == len(self.clf_loss_functions)
        self.clf_loss_functions = clf_loss_functions
        logger.info(f'Updated CE weights using \n{labels_weights}')

    # All layers already in the huggingface model
    def forward(self, *argv, **kwargs):
        clf_labels = kwargs.pop('clf_labels')
        outputs = self.model(*argv, **kwargs)
        projected_visual_features = self._extract_visual_features(kwargs['pixel_values'])

        outputs['clf_outputs'] = []
        outputs['clf_losses'] = []
        outputs['clf_predictions'] = []
        clf_loss = 0
        
        for i in range(len(self.clf_heads)):
            clf_output = self.clf_heads[i](projected_visual_features)

            loss_fn = self.clf_loss_functions[i]
            clf_loss_i = loss_fn(clf_output, clf_labels[:, i])

            clf_loss += clf_loss_i
            outputs['clf_outputs'].append(clf_output)
            outputs['clf_losses'].append(clf_loss_i)
        
        outputs['clf_outputs'] = torch.stack(outputs['clf_outputs']).transpose(0, 1)
        outputs['clf_predictions'] = torch.argmax(outputs['clf_outputs'], dim=2)
        outputs['clf_loss'] = clf_loss / len(self.clf_heads)

        outputs['original_loss'] = outputs['loss']
        outputs['loss'] = (outputs['original_loss'] + self.args.aux_loss_weight * outputs['clf_loss']) / (1 + self.args.aux_loss_weight)

        return outputs
