import torch
from sample import sample

def generate(
    model, tokenizer, prompt, max_context, length, temperature=0.5, verbose=False
):
    sequence = prompt.detach().clone()

    if verbose:
        print("Prompt:")
        print(tokenizer.decode(prompt))

    print("Generation:")
    for _ in range(length):
        input = sequence[-max_context:].unsqueeze(0)
        output = model(input)
        c = sample(output[0, -1, :], temperature)

        if verbose:
            print(tokenizer.decode(c), end="", flush=True)
        
        sequence = torch.cat([sequence, c.unsqueeze(0)])
    return sequence