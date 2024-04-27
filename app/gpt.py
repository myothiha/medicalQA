import torch
import os
import re
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from transformers import AutoTokenizer, AutoModelForCausalLM,pipeline
from langchain.chains import LLMChain, ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain import HuggingFacePipeline
from langchain import PromptTemplate
import cloudpickle
def load_model():
    file_path ="D:/AIT/Sem2/NLP/medicalQA/app/model/medical_chatbot_pickle_version3.pkl"
    with open(file_path, 'rb') as f:
        chain = cloudpickle.load(f)
    return chain

def answer_question(query):
    # # Initialize device
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # # Define model and tokenizer names
    # model_name = 'hkunlp/instructor-base'

    # embedding_model = HuggingFaceInstructEmbeddings(
    # model_name = model_name,
    # model_kwargs = {"device" : device}
    # )
    
    # model_id = 'anas-awadalla/gpt2-medium-span-head-few-shot-k-16-finetuned-squad-seed-0'

    # tokenizer = AutoTokenizer.from_pretrained(
    #     model_id)

    # tokenizer.pad_token_id = tokenizer.eos_token_id


    # # Meidcal GPT Template
    # prompt_template = """
    #         Hello! I'm your Medical Assistant, here to provide you with insights and information about health conditions, treatments, and general medical advice.
    #         Whether you need information on symptoms, advice on health issues, or details about medical procedures, feel free to ask, and I'll do my best to provide you with clear and informative answers.
    #         From managing common illnesses to understanding complex medical conditions, I'm here to assist you in navigating through your medical queries.
    #         {context}
    #         Query: {question}
    #         Answer:
    #         """.strip()

    # PROMPT = PromptTemplate.from_template(
    #     template = prompt_template
    # )

    # PROMPT



    # # Load vector database
    # vector_path = 'D:/AIT/Sem2/NLP/medicalQA/vector-store'
    # db_file_name = 'nlp_stanford'

    # vectordb = FAISS.load_local(
    #     folder_path=os.path.join(vector_path, db_file_name),
    #     embeddings=embedding_model,
    #     index_name='nlp'
    # )
    # retriever = vectordb.as_retriever()

    # # Load tokenizer and model for text generation
    # tokenizer = AutoTokenizer.from_pretrained(model_id)
    # model = AutoModelForCausalLM.from_pretrained(model_id)

    # # Create a pipeline for text generation
    # pipe = pipeline(
    #     task="text-generation",
    #     model=model,
    #     tokenizer=tokenizer,
    #     device=device
    # )

    # llm = HuggingFacePipeline(pipeline=pipe)
    # doc_chain = load_qa_chain(
    #     llm=llm,
    #     chain_type='stuff',
    #     prompt=PROMPT,
    #     verbose=True
    # )
    # question_generator = LLMChain(
    #     llm=llm,
    #     prompt=CONDENSE_QUESTION_PROMPT,
    #     verbose=True
    # )
    # memory = ConversationBufferWindowMemory(
    #     k=3,
    #     memory_key="chat_history",
    #     return_messages=True,
    #     output_key='answer'
    # )
    # chain = ConversationalRetrievalChain(
    #     retriever=retriever,
    #     question_generator=question_generator,
    #     combine_docs_chain=doc_chain,
    #     return_source_documents=True,
    #     memory=memory,
    #     verbose=True,
    #     get_chat_history=lambda h: h
    # )

    # # Get the answer
    # answer = chain({"question": query})
 

    model = load_model()
    answer = model({"question":query})
    answer_text = answer['answer']
    print (answer_text)
    return answer_text

