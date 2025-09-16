from cs336_basics.tokenizer import Tokenizer
from cs336_basics.transformer import TransformerLM
from cs336_basics.AdamW import AdamW
from cs336_basics.funciton import softmax, load_checkpoint
import torch

def generate(
    model: TransformerLM,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int, 
    temperature: float, 
    top_p: float,
    end_token: str,
    device: torch.device | None = None
) -> str:
    tokens = tokenizer.encode(prompt)
    tokens = torch.tensor([tokens], device=device) #(1, seq_len)
    end_token_id = tokenizer.encode(end_token)[0]
    
    model.eval()
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            if tokens.size(-1) > model.context_length:
                tokens = tokens[:, -model.context_length:]
            logits = model(tokens)[:, -1:, :] #(1, 1, vocab_size)
            
            #temprature scaling
            logits = logits / temperature
            
            probs = softmax(logits, -1)
            
            #top-p
            sorted_probs, sorted_indices = torch.sort(probs, descending=True) #(1, 1, vocab_size)
            cum_probs = torch.cumsum(sorted_probs, dim=-1)
            
            Vpmask = cum_probs < top_p
            Vpmask[..., 0] = True
            
            masked_porbs = sorted_probs.clone()
            masked_porbs[~Vpmask] = 0
            
            masked_porbs = masked_porbs / torch.sum(masked_porbs, keepdim=True, dim=-1)
            
            sample_indices = torch.multinomial(masked_porbs.squeeze(), num_samples=1) #(1, 1)
            next_token = sorted_indices.squeeze().gather(-1, sample_indices)
            next_token = next_token.unsqueeze(0)
            
            tokens = torch.cat([tokens, next_token], dim=-1) #(1, seq_len)
            
            if end_token_id is not None and next_token.item() == end_token_id:
                break
    
    generate_tokens = tokens.squeeze().tolist()
    return tokenizer.decode(generate_tokens)

if __name__ == '__main__':
    t = Tokenizer.from_files("data/tinystory/vocab", "data/tinystory/merges", ["<|endoftext|>"])
    
    model = TransformerLM(10000, 256, 4, 512, 16, 1344, torch.device("cuda"))
    opt = AdamW(model.parameters(), 0.001, (0.9, 0.995), 1e-8, 0.1)
    
    load_checkpoint("model/checkpoint-final", model, opt)
    
    prompt = input("prompt:")
    maxnewtokens = int(input("max_new_tokens:"))
    gen = generate(model, t, prompt, maxnewtokens, 0.3, 0.8, "<|endoftext|>", torch.device("cuda"))
    print(gen)