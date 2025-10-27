import sys
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextIteratorStreamer, pipeline
from langchain_community.llms import HuggingFacePipeline # Kept for warning suppression, but not used in streaming
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter
import threading
import logging
import traceback
import gc

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.basicConfig(level=logging.WARNING)

MODEL_MAX_LENGTH = 2048 
DEFAULT_K_DOCUMENTS = 3

def use_gpu(gpu_index):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=MODEL_MAX_LENGTH)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        dtype=torch.bfloat16, 
        device_map=f"cuda:{gpu_index}",
        low_cpu_mem_usage=True,
    )
    return model, tokenizer

def use_cpu():
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=MODEL_MAX_LENGTH)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer

def run_rag_ai(query):
    try:
        model = None
        tokenizer = None
        file_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(file_dir, "SampleContract-Shuttle.pdf")
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024,     
            chunk_overlap=100 
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
        
        # --- LangChain RAG Setup ---
        context_retriever_chain = itemgetter("question") | retriever
        
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template = """
        ### Instruction:
        You are a helpful legal assistant. You have provided the user with a contract that they must review to do business with the company you represent. 
        It is your job to answer questions the user poses about their contract in a clear and concise way to limit any further questions
        about the user's proposed query. Use the context below to answer the question clearly and concisely.
        Only answer questions based on the provided context. If the context does not contain the answer,
        say "I'm sorry, I don't have an answer for that." Do not generate jokes, response examples, or question examples. 
        Stay strictly within legal and contractual language. DO NOT OUTPUT EXAMPLE QUESTIONS AND ANSWERS.

        ### Context:
        {context}

        ### Question:
        {question}

        ###Answer:"""

        )
        
        located_docs = vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": 0.2, "k": 5},
        ).invoke(query)
        
        has_relevant_document = len(located_docs) > 0

        if not has_relevant_document:
            sys.stdout.write("I'm sorry, I don't have an answer for that.\n")
            sys.stdout.write("<<END_OF_STREAM>>\n")
            sys.stdout.flush()
            return

        context_docs = context_retriever_chain.invoke({"question": query})

        final_prompt_dict = {
            "context": context_docs,
            "question": query.strip() # Use the stripped query here
        }

        final_prompt_text = prompt_template.format(**final_prompt_dict)

        generation_started_event = threading.Event()

        def generate_with_streamer(prompt, event):
            inputs = tokenizer(prompt, return_tensors="pt", truncation=False)
            input_ids = inputs.input_ids
            attention_mask = inputs.attention_mask

            input_ids = input_ids.to(device) 
            attention_mask = attention_mask.to(device)

            event.set()

            model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=1000,
                temperature=0.65,
                do_sample=True,
                streamer=streamer,
                pad_token_id=tokenizer.eos_token_id, 
            )

        thread = threading.Thread(target=generate_with_streamer, args=(final_prompt_text, generation_started_event,))
        thread.start()

        generation_started_event.wait() 
        
        for chunk in streamer: 
            sys.stdout.write(chunk)
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
            query = sys.stdin.readline()
            
            if not query:
                break
            
            query = query.strip()
            
            if len(query) > 0:
                print(f'Processing query: "{query}"', file=sys.stderr)
                run_rag_ai(query)
            
        except RuntimeError as err:
            sys.stderr.write(f"Runtime Error: {err}\n")
            sys.stderr.flush()
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True) 
    sys.stderr.reconfigure(line_buffering=True)
    main()
