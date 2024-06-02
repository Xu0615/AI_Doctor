import gradio as gr
import transformers
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

model = None
tokenizer = None
generator = None
history = []
embedding = None
retriever = None


def load_model(model_name="path_to_your_model", device="cuda", use_rag=True):
    global model, tokenizer, generator, embedding, retriever
    print("Loading model: " + model_name)
    tokenizer = transformers.LLaMATokenizer.from_pretrained(model_name)
    model = transformers.LLaMAForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        load_in_8bit=False,
        cache_dir="cache"
    ).cuda()
    generator = model.generate
    if use_rag:
        pass                   #TODO: restore here
    #     model_name = "BAAI/bge-base-en-v1.5"
    #     model_kwargs = {'device': 'cuda'}
    #     encode_kwargs = {'normalize_embeddings': True}
    #     print("load Embedding: " + model_name)
    #     embedding = HuggingFaceBgeEmbeddings(
    #         model_name=model_name,
    #         model_kwargs=model_kwargs,
    #         encode_kwargs=encode_kwargs,
    #         query_instruction="Index for text"
    #     )
    #     persist_directory = "./data/bge1"
    #     print("load vectorDB")
    #     db = Chroma(persist_directory=persist_directory, embedding_function=embedding)
    #     retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    # # else:
    # #     model = transformers.LlamaForCausalLM.from_pretrained(
    # #         model_name,
    # #         torch_dtype=torch.float16,
    # #         low_cpu_mem_usage=True,
    # #         load_in_8bit=False,
    # #         cache_dir="cache"
    # #     ).cuda()
    # #     generator = model.generate
    # print("Model loaded successfully.")


load_model("./pretrained/")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def chat_with_doctor(user_input, use_rag, max_length, temperature, top_p, top_k, repetition_penalty, num_samples,
                     use_history):
    global history, tokenizer, model, generator, embedding, retriever
    invitation = "ChatDoctor: "
    human_invitation = "Patient: "
    num_samples = int(num_samples)  # debug: ensure int

    # Add the generated text and user input to the history and return this history
    # if user_input.strip() != "":
    #     history.append(human_invitation + user_input.strip())

    # add history into input
    if use_history == "keep history":
        if user_input.strip() != "":
            history.append(human_invitation + user_input.strip())
        fulltext = "\n\n".join(history) + "\n\n" + invitation
    else:
        history = []
        fulltext = "\n\n" + human_invitation + user_input.strip() + "\n\n" + invitation

    # Options
    # if use_rag == "Use RAG" and (retriever is None or embedding is None):
    #     load_model("./pretrained/", use_rag=True)
    # elif use_rag == "Don't Use RAG" and (model is None or generator is None):
    #     load_model("./pretrained/", use_rag=False)

    if use_rag == "Use RAG":
        if len(history) == 0:
            retrieved_docs = retriever.invoke(user_input)
        else:
            retrieved_docs = retriever.invoke("\n".join(history[-10:]))
        context = format_docs(retrieved_docs)
        fulltext = f"Context information is below. \n---------------------\n{context}\n---------------------\nGiven the context information and prior knowledge, answer the question: {fulltext}"

    gen_in = tokenizer(fulltext, return_tensors="pt").input_ids.to("cuda")
    print("\n---" + fulltext + "---\n")
    with torch.no_grad():
        generated_ids = generator(
            gen_in,
            max_length=max_length,
            max_new_tokens=512,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=num_samples,
            do_sample=True,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            #   early_stopping=True
        )
        generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        response = generated_text[len(fulltext):]
        response = response.split(human_invitation)[0].strip()

    # history.append(invitation + response)
    response = invitation + response
    if use_history == "keep history":
        history.append(response)
        display_history = "\n".join(history[-12:])
    else:
        display_history = human_invitation + user_input.strip() + '\n' + response

    return display_history


css = """
body { font-family: 'Montserrat', sans-serif; background: radial-gradient(#e66465, #9198e5); }
h1 { color: #4a56e2; }
p { color: #555; }
label { font-weight: 600; color: #555; }
input[type='text'], textarea, select, input[type=number] {
    border-radius: 20px; border: 1px solid #ccc; padding: 10px; width: 100%;
    box-sizing: border-box; margin-bottom: 10px;
}
.slider_container { margin-bottom: 10px; }
.textbox_input { border: none; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
.radio { display: block; position: relative; padding-left: 35px; margin-bottom: 12px; cursor: pointer; font-size: 18px; user-select: none; }
.radio input { position: absolute; opacity: 0; cursor: pointer; }
.checkmark { position: absolute; top: 0; left: 0; height: 25px; width: 25px; background-color: #eee; border-radius: 50%; }
.radio:hover input ~ .checkmark { background-color: #ccc; }
.radio input:checked ~ .checkmark { background-color: #2196F3; }
.checkmark:after {
    content: ""; position: absolute; display: none;
}
.radio input:checked ~ .checkmark:after {
    display: block;
}
.radio .checkmark:after {
    top: 9px; left: 9px; width: 8px; height: 8px; border-radius: 50%; background: white;
}
.button, .gr-button { background-color: #4a56e2; color: white; padding: 15px 20px; border: none; border-radius: 5px; text-transform: uppercase; font-weight: bold; letter-spacing: 1px; }
.button:hover, .gr-button:hover { background-color: #3b44b6; }
"""

iface = gr.Interface(
    fn=chat_with_doctor,
    inputs=[
        gr.Textbox(lines=2, placeholder="Please enter your medical question."),
        gr.Radio(choices=["Don't Use RAG", "Use RAG"], value="Don't Use RAG", label="Select Mode"),
        gr.Slider(100, 2048, step=1, value=1200, label="Maximum Length"),
        gr.Slider(0.1, 1.0, step=0.1, value=1.0, label="Temperature"),
        gr.Slider(0.1, 1.0, step=0.1, value=0.9, label="Top P"),
        gr.Slider(0, 100, step=1, value=30, label="Top K"),
        gr.Slider(1.0, 2.0, step=0.1, value=1.0, label="Repetition Penalty"),
        gr.Number(label="Number of Samples", value=1),
        gr.Radio(choices=["keep history", "not keep history"], value="not keep history", label="Select Mode"),
    ],
    outputs=gr.Textbox(label="Conversation History"),
    title="AIDoctor",
    description="I am AIDoctor created by 7102 group. If you have any disease questions, you can ask me.",
    css=css,  # CSS
    # theme="huggingface" ,
    layout="vertical"
)

# iface.launch(server_port=18044, server_name="0.0.0.0")
iface.launch()   # TODO