import gettext
from history_tool import Display_History
from sound.sound_classification import load_model, get_health_result
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import gradio as gr
import transformers
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma
import os
from query_classification import get_query_classifier, get_classification_result
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from typing import Iterable
from typing import Union
from gradio.themes.base import Base
from gradio.themes.utils import colors, fonts, sizes
import time
import argparse

model = None
tokenizer = None
generator = None
history = []
embedding = None
retriever = None
sound_model = load_model('sound/resnet.pth')

display_history = Display_History()
def get_query_classifier(local_model_path):
    """
    Load a query classifier model and its tokenizer from a local path.
    
    Parameters:
    local_model_path (str): The path to the local directory containing the model files.

    Returns:
    model: The loaded model.
    tokenizer: The loaded tokenizer.
    """
    model = AutoModelForSequenceClassification.from_pretrained(local_model_path)
    tokenizer = AutoTokenizer.from_pretrained(local_model_path)
    return model, tokenizer


def get_classification_result(query, action_list, prompt, model, tokenizer):
    """
    Classify the query based on the given action list and prompt using the provided model and tokenizer.

    Parameters:
    query (str): The input query to classify.
    action_list (list): The list of possible actions or classifications.
    prompt (str): The prompt template to use for classification.
    model: The loaded classification model.
    tokenizer: The loaded tokenizer.

    Returns:
    str: The classification result.
    """
    inputs = tokenizer(prompt.format(query, ""), return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_id = torch.argmax(logits, dim=1).item()
    return action_list[predicted_class_id]

def load_model(model_name="path_to_your_model", device="cuda", use_rag=False):
    global model, tokenizer, generator, embedding, retriever
    print("Loading model: " + model_name)
    tokenizer = transformers.LlamaTokenizer.from_pretrained(model_name)
    model = transformers.LlamaForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            load_in_8bit=False,
            cache_dir="cache"
        ).cuda()
    generator = model.generate
    if use_rag:
        model_name = "BAAI/bge-base-en-v1.5"
        model_kwargs = {'device': 'cuda'}
        encode_kwargs = {'normalize_embeddings': True}
        print("load Embedding: "+ model_name)
        embedding = HuggingFaceBgeEmbeddings(
                    model_name=model_name,
                    model_kwargs=model_kwargs,
                    encode_kwargs=encode_kwargs,
                    query_instruction="Index for text"
                )
        persist_directory = "./data/bge1"
        print("load vectorDB")
        db = Chroma(persist_directory=persist_directory, embedding_function=embedding)
        retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    # else:
    #     model = transformers.LlamaForCausalLM.from_pretrained(
    #         model_name,
    #         torch_dtype=torch.float16,
    #         low_cpu_mem_usage=True,
    #         load_in_8bit=False,
    #         cache_dir="cache"
    #     ).cuda()
    #     generator = model.generate
    print("Model loaded successfully.")

parser = argparse.ArgumentParser(description="Load the model with optional RAG")
parser.add_argument('--use_rag', action='store_true', help='Activate RAG by including this flag')

# Parse arguments
args = parser.parse_args()
use_rag = args.use_rag
print(use_rag)
# Use the argument in the function call
load_model("./pretrained/", use_rag=use_rag)
#qc_model, qc_tokenizer = get_query_classifier(model_name='microsoft/phi-1_5')
local_model_path = "./pretrained/microsoft/phi-1_5"
qc_model, qc_tokenizer = get_query_classifier(local_model_path)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)



def chat_with_doctor(user_input, use_rag, max_length, temperature, top_p, top_k, repetition_penalty, num_samples, use_history, use_next_prediction):
    global history, tokenizer, model, generator, embedding, retriever, display_history
    response = ""
    
    if len(history) <= 0:
        result = get_classification_result(
            query=user_input, 
            action_list=['medical query', 'not relates to health or medical treatment'],
            prompt="Consider it is a conversation between patient and doctor, what is the intention of the patient. Patient:{}. The intention of the patient:{}", 
            model=qc_model, 
            tokenizer=qc_tokenizer
        )
        if result != "medical query":
            bot_response = "Please ask a medical related question"
            display_history.append(user_input, bot_response)
            return

    if use_history == "keep history" and user_input.strip() != "":
        history.append("Patient: " + user_input.strip())
        
    fulltext = "You should only act as the AIDoctor\n\n".join(history) + "\n\nAIDoctor: " if use_history == "keep history" else "\n\nPatient: " + user_input.strip() + "\n\nAIDoctor: "
    
    if use_rag == "Use RAG":
        retrieved_docs = retriever.invoke("\n".join(history[-5:])) if len(history) > 0 else retriever.invoke(user_input)
        context = format_docs(retrieved_docs)
        fulltext = "Context information is below:\n" + context + "\nGiven the context information and prior knowledge, answer the question: \n" + fulltext
    
    print("\n------------\n" + fulltext + "\n----------------\n")
    # Generate the response to the current user input
    gen_in = tokenizer(fulltext, return_tensors="pt").input_ids.to("cuda")
    with torch.no_grad():
        generated_ids = generator(
            gen_in,
            max_length=max_length + len(fulltext),
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=1,
            do_sample=True,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        response = generated_text[len(fulltext):].strip()

    display_history.add(user_input, response)

    # Append response to history
    if use_history == "keep history":
        history.append("AIDoctor: " + response)

    # Optionally generate predicted next question
    next_question = " "
    if use_next_prediction == "Use Next Prediction":
        conversation = "\n\n".join(history[-5:]) + "\n\nAIDoctor: " if use_history == "keep history" else "\n\nPatient: " + user_input.strip() + "\n\nAIDoctor: " + response
        next_question_input = "Conversation is below. \n" + conversation + "\nGiven the conversation and prior knowledge, What might patient ask next based on the conversation?  Give me at most 3 questions in order."
        
        # '''Here is a sample: 1. What are the possible causes of my cough and headache? 2. What are the possible tests that I need to do to find out the cause of my cough and headache? 3. What are the possible treatments for my cough and headache? Keywords: cough, headache, treatment "'''
        
  
        gen_in_next_question = tokenizer(next_question_input, return_tensors="pt").input_ids.to("cuda")
        with torch.no_grad():
            generated_ids_next_question = generator(
                gen_in_next_question,
                max_new_tokens=512,
                pad_token_id=tokenizer.eos_token_id,
                num_return_sequences=1,
                do_sample=True,
                repetition_penalty=repetition_penalty,
                temperature=0.1,
                top_k=3,
                top_p=0.9,
                length_penalty=0.9
            )
            generated_next_question = tokenizer.decode(generated_ids_next_question[0], skip_special_tokens=True)
            next_question = generated_next_question[len(next_question_input):]
        print(next_question)
        display_history.add(None, f'Possible questions:\n {next_question}')
        user_input_box.value = ""
        user_input_box.placeholder = next_question
    # Construct final display history
    d_history = "\n".join(history[-12:]) if use_history == "keep history" else "Patient: " + user_input.strip() + '\n' + "AIDoctor: " + response
    d_history += "\nNext predicted question: " + next_question
    return d_history


def format_health_result(dict_):
    result = ''
    for key in dict_:
        result += f" The probability of {key} is {dict_[key]:.4f}."

    return result
import soundfile as sf
i = 0
#  save audio
def save_audio(audio_data):
    global i
    global tokenizer
    sr, data = audio_data
    print(sr)
    print(data)
    voice_dir = "./Voice"
    if not os.path.exists(voice_dir):
        os.makedirs(voice_dir, exist_ok=True)  #

        # timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"audio.wav"
    file_path = os.path.join(voice_dir, filename)
    sound_result = "According to the patient's current sound file." + " The probability of health is 0.9151. The probability of unhealth is 0.0849."

    if i != 0:
        with open(file_path, "wb") as f:
            sf.write(f, data, sr)

        sound_result = "According to the patient's current sound file." + format_health_result(get_health_result(sound_model, file_path))
    i += 1

    input_text = "\n".join(history[-3:]) + sound_result + "You should act as an AIDoctor analyse the results and give a diagnose or suggestions."
    # input_text = "\n".join(history[-5:]) + sound_result

    gen_in_next_question = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")
    with torch.no_grad():
        generated_ids_next_question = generator(
            gen_in_next_question,
            max_new_tokens=512,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=1,
            do_sample=True,
            repetition_penalty=1.2,
            temperature=0.3,
            top_k=3,
            top_p=0.9,
            length_penalty = 0.95
        )
        generated_next_question = tokenizer.decode(generated_ids_next_question[0], skip_special_tokens=True)
        next_question = generated_next_question[len(input_text):]
    print("input_text:", input_text)
    print("next_question", next_question)
    display_history.add(None, sound_result+next_question)
    return display_history.export()


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

'''
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
#gr.cleanButton.add(history)

iface.launch(server_port=18044, server_name="0.0.0.0")
'''

user_avatar_path = "./Avata/user.jpg"
doctor_avatar_path = "./Avata/doctor.png"

combined_css = ".container { max-width: 1700px; margin: auto; }" #+ additional_css


class Seafoam(Base):
    def __init__(
        self,
        *,
        primary_hue: Union[colors.Color, str] = colors.emerald,
        secondary_hue: Union[colors.Color, str] = colors.blue,
        neutral_hue: Union[colors.Color, str] = colors.blue,
        spacing_size: Union[sizes.Size, str] = sizes.spacing_md,
        radius_size: Union[sizes.Size, str] = sizes.radius_md,
        text_size: Union[sizes.Size, str] = sizes.text_lg,
        font: Union[fonts.Font, str, Iterable[Union[fonts.Font, str]]] = (
            fonts.GoogleFont("Quicksand"),
            "ui-sans-serif",
            "sans-serif",
        ),
        font_mono: Union[fonts.Font, str, Iterable[Union[fonts.Font, str]]] = (
            fonts.GoogleFont("IBM Plex Mono"),
            "ui-monospace",
            "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            spacing_size=spacing_size,
            radius_size=radius_size,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )
        super().set(
            # gradio default version
#             body_background_fill="repeating-linear-gradient(45deg, *primary_200, *primary_200 10px, *primary_50 10px, *primary_50 20px)",
#             body_background_fill_dark="repeating-linear-gradient(45deg, *primary_800, *primary_800 10px, *primary_900 10px, *primary_900 20px)",
            
#             button_primary_background_fill="linear-gradient(90deg, *primary_300, *secondary_400)",
#             button_primary_background_fill_hover="linear-gradient(90deg, *primary_200, *secondary_300)",
#             button_primary_text_color="white",
#             button_primary_background_fill_dark="linear-gradient(90deg, *primary_600, *secondary_800)",
            
            # slider_color="*secondary_300",
            # slider_color_dark="*secondary_600",
            
            # Our 7102 group further beautified version
            body_background_fill="linear-gradient(135deg, #e0f2f1 10%, #26a69a 100%)",
            body_background_fill_dark="linear-gradient(135deg, #26a69a 10%, #00796b 100%)",
            
    
            button_primary_background_fill="linear-gradient(90deg, #80cbc4, #26a69a)",
            button_primary_background_fill_hover="linear-gradient(90deg, #4db6ac, #009688)",
            button_primary_text_color="#ffffff",  
            button_primary_background_fill_dark="linear-gradient(90deg, #00796b, #004d40)",
            
            slider_color="#26a69a",
            slider_color_dark="#00796b",
            
            
            block_title_text_weight="600",
            block_border_width="3px",
            block_shadow="*shadow_drop_lg",
            button_shadow="*shadow_drop_lg",
            button_large_padding="32px",
        )


seafoam = Seafoam()


with gr.Blocks(theme=seafoam, css=combined_css) as demo:
    gr.Markdown("AI Doctor created by 7102 group")
    
    with gr.Row():
        # Control elements
        with gr.Column(scale=1):
            use_rag = gr.Radio(choices=["Don't Use RAG", "Use RAG"], value="Don't Use RAG", label="Select Mode")
            max_length = gr.Slider(100, 2048, step=1, value=1200, label="Maximum Length")
            temperature = gr.Slider(0.1, 1.0, step=0.1, value=1.0, label="Temperature")
            top_p = gr.Slider(0.1, 1.0, step=0.1, value=0.9, label="Top P")
            top_k = gr.Slider(0, 100, step=1, value=3, label="Top K")
            repetition_penalty = gr.Slider(1.0, 2.0, step=0.1, value=1.0, label="Repetition Penalty")
            num_samples = gr.Number(label="Number of Samples", value=1)
            use_history = gr.Radio(choices=["keep history", "not keep history"], value="keep history", label="Keep History")
            use_next_prediction = gr.Radio(choices=["Use Next Prediction", "Do Not Use Next Prediction"],
                                           value="Do Not Use Next Prediction",
                                           label="Next Question Prediction")
            audio_input = gr.Audio(source="microphone", type="numpy", label="Record or Upload Audio")
            save_button = gr.Button("Save Audio")

            
        # Chatbot component for the conversation
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Conversation History", avatar_images=(user_avatar_path, doctor_avatar_path)).style(height=575) # Avatar customization
            user_input_box = gr.Textbox(lines=2, placeholder="What would you like to ask the AIDoctor?", label="Your Question")
            submit_button = gr.Button("Submit")
            clean_button = gr.Button("Clean")


    def process_inputs(user_input, use_rag, max_length, temperature, top_p, top_k, repetition_penalty, num_samples, use_history, use_next_prediction):

        conversation_string = chat_with_doctor(user_input, use_rag, max_length, temperature, top_p, top_k, repetition_penalty, num_samples, use_history, use_next_prediction)
        print(display_history)
        user_input_box.update(value=None)
        return display_history.export()

   
    def clean_chat():
        global history, display_history
        # Check if history is empty
        if not history:  # If history is empty
            return [("Clean", "Please submit a question before using the 'clean' button.")]  # Display a message
        else:
            history = []  # Reset history
            display_history.clean()
            global i
            i = 0
            return []  # Clean the conversation in the chatbot

    
    save_button.click(fn=save_audio, inputs=[audio_input], outputs=chatbot)

    submit_button.click(
        fn=process_inputs,
        inputs=[user_input_box, use_rag, max_length, temperature, top_p, top_k, repetition_penalty, num_samples, use_history, use_next_prediction],
        outputs=chatbot
    )
    submit_button.click(lambda x: gr.update(value=""), None, user_input_box, queue=False)
    clean_button.click(
        fn=clean_chat,
        inputs=[],
        outputs=chatbot
    )
    
    
demo.launch(server_name="0.0.0.0", server_port=18024, share=True, inbrowser=True, ssl_verify=False)
