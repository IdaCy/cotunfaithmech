import torch

def query_model(model_and_tokenizer, prompt, max_new_tokens=50):
    """
    Generate a response from the model given a prompt.

    Args:
        model_and_tokenizer: Tuple of (model, tokenizer)
        prompt (str): The input prompt
        max_new_tokens (int): Max tokens to generate

    Returns:
        str: The generated response (excluding the original prompt)
    """
    model, tokenizer = model_and_tokenizer

    # Tokenize and move to the model's device
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Generate with no gradients (inference only)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False  # use greedy decoding for consistency
        )

    # Decode and strip the prompt
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded[len(prompt):].strip()
