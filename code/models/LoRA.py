from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType

output_modules = ["output"]
vision_encoder_modules = ["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"]
decoder_modules = ["query", "key", "value", "dense"]
visual_projection_modules = ["0"]
vision_embedding_modules = ["patch_embedding"] # "position_embedding"
word_embedding_modules = ["word_embeddings"] # "position_embeddings"
# position_embeddings_modules = ["position_embedding", "position_embeddings"]

targets_dict = {
    'conf1': ["query", "value", "q_proj", "v_proj"],
    'att': output_modules + vision_encoder_modules + decoder_modules,
    'all': output_modules + vision_encoder_modules + decoder_modules + visual_projection_modules + vision_embedding_modules + word_embedding_modules,
}


def add_lora_to_model(model, args):
    # Define LoRA Config
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=targets_dict[args.lora_targets],
        lora_dropout=args.lora_dropout,
        bias="none",
        # task_type=TaskType.SEQ_2_SEQ_LM
    )
    # lora_model = prepare_model_for_int8_training(model)
    lora_model = get_peft_model(model.model, lora_config)
    lora_model.print_trainable_parameters()
    return lora_model
