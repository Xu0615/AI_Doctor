# AI_Doctor
## Overview

This application uses advanced language models(Llama2) to provide medical advice and answers to health-related queries. 
This project is an improvement and innovation based on the chatdoctor project. 
Special thanks to the chatdoctor project (https://github.com/Kent0n-Li/ChatDoctor). 
This project is suitable for people who are learning large models, rags, and chatbot visualization.

## Innovations
- **Medical Query Classification**: Identifies and responds to health-related questions.(Phi-1.5)
- **Contextual Answers**: Utilizes historical conversation context to provide relevant answers.
- **RAG (Retrieval-Augmented Generation)**: Optionally includes retrieved context from a vector database(chroma) to enhance answer accuracy.
- **Audio Analysis**: Analyzes audio recordings to assess health conditions and provide suggestions based on sound inputs.(resnet)
- **Customizable Settings**: Users can adjust parameters like maximum response length, temperature, top-p, top-k, repetition penalty, and more.

## Install:

1. Clone the repository:
    ```sh
    git clone https://github.com/your-repo/ai-doctor.git
    cd ai-doctor
    ```

2. Create and activate a virtual environment:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

4. Download and place the necessary models:
    - Ensure the pretrained models are placed in the specified directories (e.g., `./pretrained/`).

## Data and model:

### 1. ChatDoctor Dataset:

You can download the following training dataset

100k real conversations between patients and doctors from HealthCareMagic.com [HealthCareMagic-100k](https://drive.google.com/file/d/1lyfqIwlLSClhgrCutWuEe_IACNq6XNUt/view?usp=sharing).

10k real conversations between patients and doctors from icliniq.com [icliniq-10k](https://drive.google.com/file/d/1ZKbqgYqWc7DJHs3N9TQYQVPdDQmZaClA/view?usp=sharing).

5k generated conversations between patients and physicians from ChatGPT [GenMedGPT-5k](https://drive.google.com/file/d/1nDTKZ3wZbZWTkFMBkxlamrzbNz0frugg/view?usp=sharing) and [disease database](https://github.com/Kent0n-Li/ChatDoctor/blob/main/format_dataset.csv).

Our model was firstly be fine-tuned by Stanford Alpaca's data to have some basic conversational capabilities. [Alpaca link](https://github.com/Kent0n-Li/ChatDoctor/blob/main/alpaca_data.json)

**You can also download all the  dataset from** [all data](https://drive.google.com/file/d/1xdN9ItfpswiPE7V9RqRiGBWPrrIiOQZ5/view?usp=sharing)

### 2. Model Weights

Place the model weights file [checkpoint](https://drive.google.com/drive/folders/11-qPzz9ZdHD6pc47wBSOUSU61MaDPyRh?usp=sharing) in the ./pretrained folder.

Place the sound model file [sound checkpoint](https://drive.google.com/file/d/1Ow9s9Ld4VOBbSaUyr9P6igh9MScLWUkV/view?usp=sharing) in the ./sound folders

### 3. Build the RAG

You should follow the instruction of data/bge.ipynb to build our Vector Database or you can build your own.


## File Structure
- `web.py`: Chatbot code based on gradio visualization.
- `requirements.txt`: List of required Python packages.
- `pretrained/`: Directory for pretrained models.
- `Voice/`: Directory for storing audio files.
- `Avata/`: Directory for user and doctor avatar images.

## Parameters and Customization
- **Select Mode**: Choose whether to use RAG (`Don't Use RAG` or `Use RAG`).
- **Maximum Length**: Set the maximum length of the generated response (100 to 2048).
- **Temperature**: Adjust the randomness of the generated text (0.1 to 1.0).
- **Top P**: Set the cumulative probability for nucleus sampling (0.1 to 1.0).
- **Top K**: Limit the number of highest probability tokens to consider (0 to 100).
- **Repetition Penalty**: Control the likelihood of repeating tokens (1.0 to 2.0).
- **Number of Samples**: Number of response samples to generate.
- **Keep History**: Option to retain conversation history (`keep history` or `not keep history`).
- **Next Question Prediction**: Option to predict the patient's next question based on the conversation (`Use Next Prediction` or `Do Not Use Next Prediction`).

## How to fine-tuning

```python
torchrun --nproc_per_node=4 --master_port=<your_random_port> train.py \
    --model_name_or_path <your_path_to_hf_converted_llama_ckpt_and_tokenizer> \
    --data_path ./HealthCareMagic-100k.json \
    --bf16 True \
    --output_dir pretrained \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-6 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LLaMADecoderLayer' \
    --tf32 True
```

Fine-tuning with Lora

```python
WORLD_SIZE=6 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 torchrun --nproc_per_node=6 --master_port=4567 train_lora.py \
  --base_model './weights-alpaca/' \
  --data_path 'HealthCareMagic-100k.json' \
  --output_dir './lora_models/' \
  --batch_size 32 \
  --micro_batch_size 4 \
  --num_epochs 1 \
  --learning_rate 3e-5 \
  --cutoff_len 256 \
  --val_set_size 120 \
  --adapter_name lora
```

## How to inference

 You can build a AIDoctor model on your own machine and communicate with it.

```python
python chat.py
```

```python
python chat_rag.py
```

You can build a web interface to communicate with it.

```python
python web.py
```

Or you can use RAG if you have buile the Vector Database

```python
python web.py --use_rag
```
## Tips (Important!)
1. For Llama2 checkpoints, refer to the following Google Drive link: https://drive.google.com/drive/folders/11-qPzz9ZdHD6pc47wBSOUSU61MaDPyRh, please 
download the pytorch_model-00001/2/3-of-00003 in the link, and put the model in the pretrained folder.

2. If you want to use rag, please run the ./data/bge.ipynb file first to build your own vector database. 
If you are interested in our medical vector database(conversation1.txt), please contact me (sunny615527@gmail.com).

3. If you want to implement the function of determining whether the user's question is a medical question, you can add pytorch_model.bin to the local path of .\pretrained\microsoft\phi-1_5, refer to download.py. 
Or you can load it directly from huggingface (https://huggingface.co/microsoft/phi-1_5)

4. If uploading to the server is slow, you can download it to Alibaba Cloud Disk first, and then download it from Alibaba Cloud Disk to the server. 
Here is the github address for using Alibaba Cloud Disk in Linux: https://github.com/tickstep/aliyunpan

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Contact
For questions or support, please contact at [sunny615527@gmail.com](mailto:email@example.com).

## Contributing
We encourage everyone to use more cutting-edge open source big models and the latest LLM-related technologies to improve 
and innovate AI_Doctor! We look forward to and thank everyone for their contributions.
