import torch
from transformers import AutoModel

MODEL_NAME = "Maxlegrec/ChessBot"

print(f"Loading model '{MODEL_NAME}'...")
model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)

print("\n=== Model Named Modules ===")
for name, module in model.named_modules():
    # Only print modules that actually have parameters (to avoid too much clutter)
    if len(list(module.parameters(recurse=False))) > 0:
        print(f"Layer Name: {name}")
        print(f"Layer Type: {type(module).__name__}")
        print("-" * 30)

print("\nTIP: Look for Linear or Conv1d layers to use as 'target_modules' in LoRA.")
