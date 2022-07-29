import torch.nn.functional as F
import torch.distributions as distributions

def sample(lnprobs, temperature=0.8):
    if temperature == 0.0:
        return lnprobs.argmax()

    p = F.softmax(lnprobs / temperature, dim=0)
    cd = distributions.Categorical(p)
    return cd.sample()