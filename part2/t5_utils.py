import os
import torch
from transformers import T5ForConditionalGeneration, T5Config, T5TokenizerFast


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_wandb(args):
    # You can optionally add wandb init here, but it is not required.
    pass


# def initialize_model(args):
#     """
#     Initialize either:
#         - Pretrained T5-small (if --finetune)
#         - Randomly initialized T5-small (config only)

#     Returns:
#         model (nn.Module) on device
#     """
#     if args.finetune:
#         # Load pretrained model
#         model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-small")
#     else:
#         # Train from scratch using the same config as t5-small
#         config = T5Config.from_pretrained("google-t5/t5-small")
#         model = T5ForConditionalGeneration(config)

#     model = model.to(DEVICE)
#     return model


def initialize_model(args):
    """
    Initialize pretrained or scratch T5 and attach tokenizer.
    """
    tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-small")
    args.tokenizer = tokenizer   # <--- IMPORTANT!

    if args.finetune:
        model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-small")
    else:
        config = T5Config.from_pretrained("google-t5/t5-small")
        model = T5ForConditionalGeneration(config)

    model.tokenizer = tokenizer  # <--- attach tokenizer to model too

    return model.to(DEVICE)


def mkdir(dirpath):
    if not os.path.exists(dirpath):
        try:
            os.makedirs(dirpath)
        except FileExistsError:
            pass


def save_model(checkpoint_dir, model, best=False):
    """
    Save model weights.
    Two versions:
        checkpoint_dir/model.pt
        checkpoint_dir/model_best.pt
    """
    mkdir(checkpoint_dir)

    filename = "model_best.pt" if best else "model.pt"
    save_path = os.path.join(checkpoint_dir, filename)

    torch.save(model.state_dict(), save_path)
    print(f"[Saved] {save_path}")


def load_model_from_checkpoint(args, best=True):
    """
    Load saved model weights.
    Must reinitialize the model first using initialize_model(args).
    """
    model = initialize_model(args)

    checkpoint_dir = args.checkpoint_dir
    filename = "model_best.pt" if best else "model.pt"
    load_path = os.path.join(checkpoint_dir, filename)

    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Checkpoint not found: {load_path}")

    state_dict = torch.load(load_path, map_location=DEVICE)
    model.load_state_dict(state_dict)

    print(f"[Loaded] {load_path}")
    return model

import transformers

def initialize_optimizer_and_scheduler(args, model, epoch_length):
    optimizer = initialize_optimizer(args, model)
    scheduler = initialize_scheduler(args, optimizer, epoch_length)
    return optimizer, scheduler


def initialize_optimizer(args, model):
    decay_parameters = get_parameter_names(model, transformers.pytorch_utils.ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if (n in decay_parameters and p.requires_grad)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if (n not in decay_parameters and p.requires_grad)
            ],
            "weight_decay": 0.0,
        },
    ]

    if args.optimizer_type == "AdamW":
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            eps=1e-8,
            betas=(0.9, 0.999),
        )
    else:
        raise NotImplementedError

    return optimizer


def initialize_scheduler(args, optimizer, epoch_length):
    num_training_steps = epoch_length * args.max_n_epochs
    num_warmup_steps = epoch_length * args.num_warmup_epochs

    if args.scheduler_type == "none":
        return None
    elif args.scheduler_type == "cosine":
        return transformers.get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps, num_training_steps
        )
    elif args.scheduler_type == "linear":
        return transformers.get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps, num_training_steps
        )
    else:
        raise NotImplementedError


def get_parameter_names(model, forbidden_layer_types):
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model-specific parameters
    result += list(model._parameters.keys())
    return result
