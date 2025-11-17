import torch
from transformers import T5ForConditionalGeneration, T5TokenizerFast
from load_data import get_dataloader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load tokenizer
tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-small")

# 2. Create model architecture
model = T5ForConditionalGeneration.from_pretrained(
    "google-t5/t5-small"
)

# 3. Load your fine-tuned weights
state_dict = torch.load(
    "checkpoints/ft_experiments/experiment/model_best.pt",
    map_location=DEVICE
)
model.load_state_dict(state_dict)
model = model.to(DEVICE)
model.eval()

# 4. Load dev dataloader
dev_loader = get_dataloader(4, "dev")

# 5. Show predictions
for batch in dev_loader:
    encoder_input, encoder_mask, _, _, _ = batch

    encoder_input = encoder_input.to(DEVICE)
    encoder_mask = encoder_mask.to(DEVICE)

    with torch.no_grad():
        pred_ids = model.generate(
            input_ids=encoder_input,
            attention_mask=encoder_mask,
            max_length=256,
            num_beams=1,
        )

    for i in range(len(pred_ids)):
        nl_text = tokenizer.decode(
            encoder_input[i], skip_special_tokens=True
        )
        pred_text = tokenizer.decode(
            pred_ids[i], skip_special_tokens=True
        )

        print("NL:", nl_text)
        print("PRED:", pred_text)
        print("-" * 50)

    break

