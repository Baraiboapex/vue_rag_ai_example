import sys
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextIteratorStreamer, StoppingCriteria, StoppingCriteriaList
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
import unidecode

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.basicConfig(level=logging.WARNING)

import torch

class StopAfterPhraseCriteria(StoppingCriteria):
    """
    Custom stopping criterion to terminate generation once the STOP_PHRASE
    has been generated.
    """
    def __init__(self, stop_token_ids: torch.LongTensor):
        self.stop_token_ids = stop_token_ids
        self.len = len(stop_token_ids)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if input_ids.shape[-1] >= self.len:
            last_tokens = input_ids[0, -self.len:]
            if torch.equal(last_tokens, self.stop_token_ids):
                return True 
        return False

STOP_PHRASE = "I'm sorry, I don't have an answer for that."
MODEL_MAX_LENGTH = 768 
DEFAULT_K_DOCUMENTS = 3
CHATML_TEMPLATE = [
    # 1. System Role: Takes your entire '### Instruction' section
    {
        "role": "system",
        "content": (
            """
                You are an **Expert Legal Assistant** Your task is to analyze the contract provided in the Context section and answer the user's Question. 

                ### **CORE INSTRUCTIONS:**

                1.  **GROUNDING:** Your response MUST be based **EXCLUSIVELY** on the content in the **CONTEXT** section. Do not use external knowledge or make inferences.
                2.  **BREVITY & TONE:** Make all answers **EXTREMELY concise and brief** and strictly professional. **DO NOT** use conversational filler, greetings, or long-winded answers.

                ---

                ### **CRITICAL OUTPUT EXCLUSIVITY RULES (MANDATORY):**

                You have only **TWO** possible output formats. Your entire response **MUST** be one of these, and **NEVER** a combination.

                **A. IF THE ANSWER IS PRESENT:**
                    * **ACTION:** Provide **ONLY the factual answer.** **STOP GENERATION.**

                **B. IF THE ANSWER IS MISSING** (e.g., marked as an Exhibit, is a blank field, or is not addressed):
                    * **ACTION:** you must inform the user that the requested information is not provided in the contract **STOP GENERATION.**

                ---

                ### **ENFORCEMENT AND FORMATTING:**

                * **RULE VIOLATION:** You **MUST NOT** combine the factual answer with the negative constraint phrase.
                * **ASCII COMPLIANCE:** The output must use **strictly standard ASCII characters**. (e.g., use ' instead of ’; use " instead of “ ”).
            """
        )
    },
    # 2. User Role: Uses the placeholders
    {
        "role": "user",
        "content": "Context: \n{context}\n\nQuestion: {question}"
    }
]

def load_llm_resources(model_name):
    """
    Loads the LLM, Tokenizer, and sets up quantization/PEFT once at startup.
    This function replaces the previous use_gpu/use_cpu calls.
    """
    global LLM_MODEL, LLM_TOKENIZER, LLM_DEVICE

    cuda_is_available = torch.cuda.is_available()
    LLM_DEVICE = torch.device("cuda:0" if cuda_is_available else "cpu")
    
    try:
        if cuda_is_available:
            print("INFO::Attempting to load model using 4-bit quantization and device offloading.")
            
            compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4", 
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=compute_dtype 
            )
            
            LLM_TOKENIZER = AutoTokenizer.from_pretrained(model_name)
            
            # This is essential for 6GB VRAM cards with models like Phi-3-mini-128k-instruct.
            # Though this is a small model, this is still quite large for a 6GB V-RAM card.
            LLM_MODEL = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=compute_dtype,
                device_map={'': 0},
                quantization_config=bnb_config,
                low_cpu_mem_usage=True, 
                trust_remote_code=False,
                attn_implementation='eager'
            )
            
            LLM_MODEL = prepare_model_for_kbit_training(LLM_MODEL)
            
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["qkv_proj", "o_proj"],
                lora_dropout=0.05, 
                bias="none", 
                task_type="CAUSAL_LM"
            )
            
            LLM_MODEL = get_peft_model(LLM_MODEL, lora_config)
            
            # Ensure the use_cache fix is applied to the config 
            if getattr(LLM_MODEL.config, "use_cache", False):
                LLM_MODEL.config.use_cache = False
            
        else:
            print("WARNING::CUDA not available. Loading model on CPU (inference will be slow).")
            LLM_TOKENIZER = AutoTokenizer.from_pretrained(model_name)
            LLM_MODEL = AutoModelForCausalLM.from_pretrained(model_name)
            
        if LLM_TOKENIZER.pad_token is None:
            LLM_TOKENIZER.pad_token = LLM_TOKENIZER.eos_token
            
        print(f"INFO::Model loading complete. Model device map: {getattr(LLM_MODEL, 'hf_device_map', LLM_DEVICE)}")
        
    except Exception as e:
        sys.stderr.write(f"CRITICAL ERROR::Model Loading Failed: {e}\n")
        sys.stderr.write(traceback.format_exc())
        sys.stderr.flush()
        # Set to None if failure occurs
        LLM_MODEL = None
        LLM_TOKENIZER = None

def generate_with_streamer(messages, device, tokenizer, model, streamer, event):
    """
    Handles tokenization in the correct ChatML format and streams the model's output.
    """
    try:
        input_text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False,
            add_generation_prompt=True 
        )

        inputs = tokenizer(
            input_text, 
            return_tensors="pt"
        )
        
        input_ids = inputs.input_ids.to(model.device) 
        attention_mask = inputs.attention_mask.to(model.device)

        event.set()

        stop_token_ids = tokenizer.encode(STOP_PHRASE, return_tensors='pt')[0]
        custom_stop = StopAfterPhraseCriteria(stop_token_ids.to(model.device))
        stopping_criteria = StoppingCriteriaList([custom_stop])
        
        model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=MODEL_MAX_LENGTH,
            temperature=0.65,
            top_p=0.9,
            do_sample=True,
            streamer=streamer,
            eos_token_id=tokenizer.eos_token_id, 
            pad_token_id=tokenizer.pad_token_id,
            stopping_criteria=stopping_criteria,
            repetition_penalty=1.1
        )
    except Exception as e:
        sys.stderr.write(f"!!! GENERATION FAILED !!!\nError in generate_with_streamer: {e}\n")
        sys.stderr.write(traceback.format_exc())
        sys.stderr.flush()

def clean_to_ascii(text: str):
    ascii_text = unidecode.unidecode(text)
    
    filtered_text = "".join(
        char for char in ascii_text if 32 <= ord(char) <= 126 or char in ('\n', '\t', '\r')
    )
    
    return filtered_text

def run_rag_ai(query, session_id, user_id):
    """
    The main RAG execution function. Now only initializes RAG components and
    uses the globally loaded LLM.
    """
    global LLM_MODEL, LLM_TOKENIZER, LLM_DEVICE
    
    if LLM_MODEL is None or LLM_TOKENIZER is None:
        sys.stderr.write(f"ERROR::{user_id}--{session_id}::LLM Model not successfully loaded during startup. Aborting query.\n")
        sys.stderr.flush()
        return
    
    try:
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
            search_kwargs={"score_threshold": 0.32, "k": 5},
        ).invoke(query)
        
        has_relevant_document = len(located_docs) > 0

        if not has_relevant_document:
            # Added session/user ID to output for logging clarity on the pipe
            sys.stdout.write(f"{user_id}--{session_id}::{STOP_PHRASE}\n") 
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
        
        streamer = TextIteratorStreamer(LLM_TOKENIZER, skip_prompt=True, skip_special_tokens=True)
        generation_started_event = threading.Event()

        thread = threading.Thread(
            target=generate_with_streamer, 
            args=(final_messages, LLM_DEVICE, LLM_TOKENIZER, LLM_MODEL, streamer, generation_started_event,)
        )
        thread.start()
        generation_started_event.wait() 
        
        for chunk in streamer: 
            # Write only the chunk data, as the prefix is already written
            sys.stdout.write(f"{user_id}--{session_id}::{clean_to_ascii(chunk)}")
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
    """
    Clears CUDA cache and collects Python garbage. 
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    """
    Initializes the model once and then enters a persistent loop
    to handle queries via standard input (stdin).
    """
    # Load the model and tokenizer ONCE before the loop starts
    load_llm_resources(model_name="microsoft/Phi-3-mini-128k-instruct") 
    
    # Start the continuous query processing loop
    while True:
        
        try:
            string_to_use = sys.stdin.readline()
            
            # Exit loop if standard input pipe is closed
            if not string_to_use:
                break
                
            query_string = string_to_use.strip()
            
            # Parsing logic for query, user_id, and session_id
            # Assuming format is: user_id--session_id::query
            
            match = re.search(r"(.*)\-\-(.*)::(.*)", query_string)
            if not match:
                sys.stderr.write(f"ERROR::Input format incorrect: {query_string}\n")
                continue
                
            user_id, session_id, query = match.groups()
            
            if len(query) > 0:
                run_rag_ai(query, session_id, user_id)
                
            cleanup_thread = threading.Thread(target=clean_memory_states)
            cleanup_thread.daemon = True 
            cleanup_thread.start()
            
        except RuntimeError as err:
            sys.stderr.write(f"{user_id}--{session_id}::{err}\n")
            sys.stderr.flush()
        except Exception as e:
            sys.stderr.write(f"{user_id}--{session_id}::{e}\n")
            sys.stderr.flush()


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True) 
    sys.stderr.reconfigure(line_buffering=True)
    main()
