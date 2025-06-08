from transformers import GPT2LMHeadModel, GPT2Tokenizer

def generate_text(prompt, max_length=100):
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(
        input_ids,
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        early_stopping=True
    )
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

if __name__ == "__main__":
    prompt = input("Enter a prompt: ")
    print(generate_text(prompt))
