import sys

# Import core components
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline      
from langchain.llms import HuggingFacePipeline                             
from langchain.prompts import PromptTemplate                                
from langchain.vectorstores import FAISS                                    
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader  
from langchain.chains import RetrievalQA            
import torch

model = None
tokenizer = None

loader = PyPDFLoader("SampleContract-Shuttle.pdf")
documents = loader.load() 

if torch.cuda.is_available():
  from unsloth.fasterlm import FastLanguageModel

  model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/mistral-7b",
        max_seq_length=10000,
        dtype=None,
        rope_scaling={"type":"linear","factor":4.88},
        load_in_4bit=True
    )
else:
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"            
    tokenizer = AutoTokenizer.from_pretrained(model_name)        
    model = AutoModelForCausalLM.from_pretrained(model_name)    

generator = pipeline(
    "text-generation",                                       
    model=model,                                             
    tokenizer=tokenizer,                                     
    max_new_tokens=3000,                                      
    temperature=0.4,                                        
    top_p=0.9,                                               
    do_sample=True                                           
)

llm = HuggingFacePipeline(pipeline=generator)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")     
vectorstore = FAISS.from_documents(documents, embeddings)              # Embed documents and store them in a FAISS index for efficient retrieval
retriever = vectorstore.as_retriever()   

prompt_template = PromptTemplate(
    input_variables=["context", "question", "question_is_good"],
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

# In the below, we use our llm, our retriever, and our prompt_template to create a RAG chain
rag_chain = RetrievalQA.from_chain_type(          # Builds a full RAG pipeline that connects your retriever and LLM using a selected chain type
    llm=llm,                                      # The language model used to generate answers
    retriever=retriever,                          # Retrieves relevant documents based on the user's question
    chain_type_kwargs={"prompt": prompt_template} # Use a custom prompt template to control formatting and tone
)

query = sys.argv[1]

located_doc = None
located_docs = vectorstore.as_retriever(    
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.2},
).invoke(query)

has_relevant_document = len(located_docs) > 0

if has_relevant_document:
  result = rag_chain(query)
  valid_answer_string = f"""{{"answer":"Answer: {result['result']}\"}}"""
  print(valid_answer_string)
else:
  invalid_answer_string = f"""{{"answer":"Answer: I'm sorry, I don't have an answer for that.\"}}"""
  print(invalid_answer_string)