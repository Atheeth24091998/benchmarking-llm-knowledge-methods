from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model_name = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    device_map="auto",
    torch_dtype=torch.float16
)

def generate_answer(query, retrieved_chunks):
    context = "\n\n".join([c['text'] for c in retrieved_chunks])
    
    # Create messages in chat format
    messages = [
        {
            "role": "system",
            "content": "You are a technical support assistant for industrial machinery. Extract troubleshooting information from the provided manual context and structure it clearly."
        },
        {
            "role": "user",
            "content": f"""Based on this machine manual excerpt:

{context}

Question: {query}

Extract and format the answer as:

Problem:
- <describe the identified problem>

Possible Causes:
- <list specific causes from the manual>
- <another cause>
- <another cause>

Recommended Actions:
- <list specific actions from the manual>
- <another action>
- <another action>

Only use information from the manual context. If information is missing, write "Not specified in manual"."""
        }
    ]
    
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False, 
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=300,
        temperature=0.1,
        do_sample=True,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )
    
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extracting only the assistant's response
    if "assistant" in answer:
        parts = answer.split("assistant")
        if len(parts) > 1:
            answer = parts[-1].strip()
    
    # Removing any remaining special tokens or headers
    answer = answer.replace("<|end_header_id|>", "").strip()
    
    return answer
