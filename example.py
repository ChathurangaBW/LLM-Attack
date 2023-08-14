import random
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained GPT-2 model and tokenizer
model_name = "gpt2"  # You can choose a different model size if needed
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Seed input text for manipulation
seed_text = "Climate change is"

# Generate a manipulated response
def generate_manipulated_response(seed_text, max_length=100, manipulation_factor=0.5):
    # Tokenize input text
    input_ids = tokenizer.encode(seed_text, return_tensors="pt")

    # Generate text using the model
    output = model.generate(
        input_ids,
        max_length=max_length,
        temperature=manipulation_factor,  # Higher values make output more random
        num_return_sequences=1
    )

    # Decode generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

# Generate manipulated content and print
manipulated_content = generate_manipulated_response(seed_text, manipulation_factor=1.5)
print("Manipulated Content:")
print(manipulated_content)
