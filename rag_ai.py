import sys
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoConfig, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_classic.chains import RetrievalQA
from langchain_community.vectorstores.faiss import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Set environment variables
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# TinyLlama has a max context of 2048 tokens. We use this to cap the input/output.
MODEL_MAX_LENGTH = 2048 
DEFAULT_K_DOCUMENTS = 3 # Number of document chunks to retrieve

def use_gpu(gpu_index):
    """Initializes the model and tokenizer for GPU using 4-bit quantization (bnb)."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    
    # Explicitly set model_max_length here to align with the model's capability
    tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=MODEL_MAX_LENGTH)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        dtype=torch.bfloat16, 
        device_map=f"cuda:{gpu_index}",
        low_cpu_mem_usage=True,
    )
    return model, tokenizer

def use_cpu():
    """Initializes the model and tokenizer for CPU without quantization."""
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=MODEL_MAX_LENGTH)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer

def main():
    try:
        model = None
        tokenizer = None

        file_dir = os.path.dirname(os.path.abspath(__file__))
        # NOTE: Ensure 'SampleContract-Shuttle.pdf' exists in the same directory
        file_path = os.path.join(file_dir, "SampleContract-Shuttle.pdf")
        
        loader = PyPDFLoader(file_path)
        documents = loader.load() 

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024,      # Target size for each chunk (in characters)
            chunk_overlap=100     # Overlap to maintain context between chunks
        )
        
        split_documents = text_splitter.split_documents(documents)
        
        cuda_is_available = torch.cuda.is_available()
        print(f"Cuda version needed : {cuda_is_available} {torch.version.cuda}")
        if cuda_is_available:
            print("CUDA available")
            model, tokenizer = use_gpu(gpu_index=0)
        else:
            model, tokenizer = use_cpu()

        # Define the generation pipeline
        generator = pipeline(
            "text-generation",
            model=model, 
            tokenizer=tokenizer, 
            max_new_tokens=300,# Max tokens the LLM will *generate* (the response length)
            temperature=0.4, 
            top_p=0.9,
            do_sample=True,
            # Ensure the pipeline respects the model's context limit during input handling
            truncation=False,
            max_length=MODEL_MAX_LENGTH
        )

        llm = HuggingFacePipeline(pipeline=generator)
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2", 
            model_kwargs={"device": "cuda" if cuda_is_available else "cpu"},
            encode_kwargs={
                "convert_to_tensor":cuda_is_available,
                "batch_size": 16
            },
        )
        vectorstore = FAISS.from_documents(split_documents, embeddings)

        # --- FIX APPLIED HERE ---
        # We explicitly configure the retriever to only fetch 3 document chunks (k=3).
        # This prevents stuffing too much context into the prompt, keeping the total
        # token count below the 2048 limit.
        retriever = vectorstore.as_retriever(search_kwargs={"k": DEFAULT_K_DOCUMENTS})

        print("Building your response...")
        
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template = """
        ### Instruction:
        You are a helpful legal assistant. Use the context below to answer the question clearly and concisely.
        Only answer questions based on the provided context. If the context does not contain the answer,
        say "I'm sorry, I don't have an answer for that." Do not generate jokes, response examples, or question examples. 
        Stay strictly within legal and contractual language. DO NOT OUTPUT EXAMPLE QUESTIONS AND ANSWERS.

        ### Context:
        {context}

        ### Question:
        {question}

        ###Answer:"""

        )

        rag_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt_template} 
        )

        # We use a separate retriever call here to check for relevance, 
        # but the main RAG chain above uses the limited 'retriever' object.
        query = "How should the consultant handle billing?"
        
        located_docs = vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": 0.2, "k": 5}, # k=5 is fine here since it's just for the check
        ).invoke(query)

        has_relevant_document = len(located_docs) > 0

        print("Now generating response!")

        if has_relevant_document:
            print(query)
            result = rag_chain.invoke(query) # Changed to .invoke for modern LangChain syntax
            valid_answer_string = f"""{{"response":"Answer: {result['result']}\"}}"""
            print(valid_answer_string)
        else:
            invalid_answer_string = f"""{{"response":"Answer: I'm sorry, I don't have an answer for that.\"}}"""
            print(invalid_answer_string)
    except Exception as err:
        print("Something got messed up!!! : ", err)

if __name__ == "__main__":
    main()
