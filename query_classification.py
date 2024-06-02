import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def get_query_classifier(model_name='microsoft/phi-1_5'):
    # torch.set_default_device("cuda")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True).cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def get_logit(prompt, model, tokenizer):
    inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=False)
    inputs["input_ids"] = inputs["input_ids"].to("cuda")
    # print(inputs, model)
    outputs = model(**inputs)
    labels = inputs.input_ids[0]
    logits = outputs.logits[0]
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss_fct = torch.nn.CrossEntropyLoss()
    shift_logits = shift_logits.view(-1, model.config.vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    loss = loss_fct(shift_logits, shift_labels)
    return -loss

def get_prompts(query, action_list, 
                prompt="Consider it is a conversation between patient and doctor,"
                "what is the intention of the patient. Patient:{}. The intention of the patient:{}"):
    return [prompt.format(query, action) for action in action_list]

def get_classification_result(query, action_list, prompt, model, tokenizer):
    logit_list = [get_logit(p, model, tokenizer) 
                  for p in get_prompts(query, action_list, prompt)]
    return action_list[torch.argmax(torch.tensor(logit_list))]

if __name__ == '__main__':
    model, tokenizer = get_query_classifier(model_name='microsoft/phi-1_5')
    result = get_classification_result(query='I want to play cards', 
                                        action_list=['medical query', 'not relates to health or medical treatment'], 
                                        prompt="Consider it is a conversation between patient and doctor,"
                                        "what is the intention of the patient. Patient:{}. The intention of the patient:{}", 
                                        model=model, 
                                        tokenizer=tokenizer)
    print(result)
    result = get_classification_result(query='I have a fever', 
                                        action_list=['medical query', 'not relates to health or medical treatment'], 
                                        prompt="Consider it is a conversation between patient and doctor,"
                                        "what is the intention of the patient. Patient:{}. The intention of the patient:{}", 
                                        model=model, 
                                        tokenizer=tokenizer)
    print(result)
    