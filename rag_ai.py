import sys
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextIteratorStreamer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from langchain_community.llms import HuggingFacePipeline # Kept for warning suppression, but not used in streaming
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter
import concurrent.futures
import threading
import logging
import traceback
import gc
import re

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.basicConfig(level=logging.WARNING)

MODEL_MAX_LENGTH = 2048 
DEFAULT_K_DOCUMENTS = 3
CHATML_TEMPLATE = [
    # 1. System Role: Takes your entire '### Instruction' section
    {
        "role": "system",
        "content": (
            "You are a helpful and expert legal assistant. Your task is to analyze the contract provided in the Context section and answer the user's Question. "
            "Answer the question clearly and concisely using ONLY the provided context. Do not use external knowledge or make inferences. "
            "***GROUNDING INSTRUCTION:*** Before providing any answer, first determine if the relevant information is explicitly present in the Context. "
            "If the Context states that the information is in an Exhibit (e.g., Exhibit A, Exhibit B) OR if the required information is left blank (e.g., \"$_____\", \"(DATE)\"), you **MUST** respond with the following EXACT phrase: \"I'm sorry, I don't have an answer for that.\" "
            "For all other questions, provide a clear, concise answer. Maintain a strictly professional, legal, and contractual tone. Do not generate jokes, response examples, or conversational fillers. "
            "Output must use strictly standard ASCII characters. For example, do not use \"’\" but instead use \"'\"."
        )
    },
    # 2. User Role: Uses the placeholders
    {
        "role": "user",
        "content": "Context: \n{context}\n\nQuestion: {question}"
    }
]

def use_gpu(gpu_index):
    """_summary_
    
        Although the tiny llama model that I am using is alredy quite performant, I wanted to use as much optimization as possible
        in order to compensate for my limited GPU that I was using at the time (6 GB of v-ram). 
        Another important note is that while tiny llama is highly optimized, it doesn't always abide by the rules
        when using prompt engineering and it also is NOT as accurate as other larger models like mistral 7B.
        However, for demonstration purposes, this will do as it does usually output good answers for 
        content that is found within its defined context that it is using for RAG AI.
    """
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4", 
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype 
    )
    
    model_name = "microsoft/Phi-3-mini-128k-instruct"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16, #Using 16 bit floats to have SOME balance of accuracy/speed ratio
        device_map=f"cuda:{gpu_index}",
        quantization_config=bnb_config,
        low_cpu_mem_usage=True, #Went ahead and limited cpu memory usage to rectify the model weight loading bottleneck
    )
    
    model = prepare_model_for_kbit_training(model)
    
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["qkv_proj", "o_proj"],
        lora_dropout=0.05, 
        bias="none", 
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    
    model.print_trainable_parameters()
    
    if getattr(model.config, "use_cache", False):
        model.config.use_cache = False
    
    return model, tokenizer

def use_cpu():
    model_name = "microsoft/Phi-3-mini-128k-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer

def generate_with_streamer(messages, device, tokenizer, model, streamer, event):
    """
        Handles tokenization in the correct ChatML format and streams the model's output.
        
        NOTE: The 'messages' argument is now the fully concrete list of dictionaries 
        with context and question substituted.
    """
    try:
        input_text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False,
            add_generation_prompt=True # Inserts the final <|assistant|> token
        )

        inputs = tokenizer(
            input_text, 
            return_tensors="pt"
        )
        
        input_ids = inputs.input_ids.to(device) 
        attention_mask = inputs.attention_mask.to(device)

        event.set()

        model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=4000,
            temperature=0.65,
            top_p=0.9,
            do_sample=True,
            streamer=streamer,
            eos_token_id=tokenizer.eos_token_id, 
            pad_token_id=tokenizer.pad_token_id
        )
    except Exception as e:
        sys.stderr.write(f"!!! GENERATION FAILED !!!\nError in generate_with_streamer: {e}\n")
        sys.stderr.write(traceback.format_exc())
        sys.stderr.flush()

def run_rag_ai(query, session_id, user_id):
    try:
        model = None
        tokenizer = None
        file_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(file_dir, "SampleContract-Shuttle.pdf")
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024,     
            chunk_overlap=256 
        )
        
        split_documents = text_splitter.split_documents(documents)
        
        cuda_is_available = torch.cuda.is_available()
        device = torch.device("cpu") 
        
        if cuda_is_available:
            model, tokenizer = use_gpu(gpu_index=0)
            device = next(model.parameters()).device 
            if hasattr(tokenizer, 'device') and tokenizer.device != device:
                tokenizer.device = device
            elif hasattr(tokenizer, 'to'):
                tokenizer.to(device)
        else:
            model, tokenizer = use_cpu()

        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2", 
            model_kwargs={"device": "cuda" if cuda_is_available else "cpu"},
            encode_kwargs={
                "convert_to_tensor":cuda_is_available,
                "batch_size": 16
            },
        )
        vectorstore = FAISS.from_documents(split_documents, embeddings)

        retriever = vectorstore.as_retriever(search_kwargs={"k": DEFAULT_K_DOCUMENTS})
        
        context_retriever_chain = itemgetter("question") | retriever

        located_docs = vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": 0.4, "k": 5},
        ).invoke(query)
        
        has_relevant_document = len(located_docs) > 0

        if not has_relevant_document:
            sys.stdout.write(f"{user_id}--{session_id}::I'm sorry, I don't have an answer for that.\n".replace("\n"," "))
            sys.stdout.flush()
            return
        
        context_docs = context_retriever_chain.invoke({"question": query})
        
        
        actual_context_text = "\n\n".join([d.page_content for d in context_docs])
        final_messages = [
            CHATML_TEMPLATE[0],
            {
                "role": "user",
                "content": CHATML_TEMPLATE[1]['content'].format(
                    context=actual_context_text, 
                    question=query.strip()
                )
            }
        ]
        generation_started_event = threading.Event()

        thread = threading.Thread(target=generate_with_streamer, args=(final_messages, device, tokenizer, model, streamer, generation_started_event,))
        thread.start()

        generation_started_event.wait() 
        
        for chunk in streamer: 
            sys.stdout.write(user_id+"--"+session_id + "::" + chunk)
            sys.stdout.flush()
            
        thread.join()

        sys.stdout.write("<<END_OF_STREAM>>\n")
        sys.stdout.flush()
            
    except Exception as err:
        sys.stderr.write("!!! PYTHON SCRIPT FAILED !!!\n")
        sys.stderr.write(f"Error: {err}\n")
        sys.stderr.write("Traceback:\n")
        sys.stderr.write(traceback.format_exc())
        sys.stderr.flush()

def clean_memory_states():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def main(): 
    clean_memory_states()
    
    while True:
        try:
            string_to_use = sys.stdin.readline()
            query_string = string_to_use
            
            query = re.search(r"(?<=::)(.*)", query_string).group()
            user_id = re.search(r"(.*)(?=\-\-)", query_string).group()
            session_id = re.search(r"(?<=\-\-)(.*)(?=::)", query_string).group()  
            
            if not query_string:
                break
            
            if len(query) > 0:
                run_rag_ai(query, session_id, user_id)
            
        except RuntimeError as err:
            sys.stderr.write(f"Runtime Error: {user_id}--{session_id}::{err}\n")
            sys.stderr.flush()
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True) 
    sys.stderr.reconfigure(line_buffering=True)
    main()
