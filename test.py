import torch
import transformers

tokenized = torch.load("test_213.pt")

print(len(tokenized[0]))

tokenizer = transformers.GPT2TokenizerFast.from_pretrained("gpt2")
print(tokenizer.decode(tokenized[0][2040]))
