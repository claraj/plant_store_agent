import os 
from dotenv import load_dotenv 

load_dotenv()

from langchain_chroma import Chroma
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate 
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.tools import tool 

GOOGLE_API_KEY = os.environ['GOOGLE_API_KEY']
EMBEDDING_MODEL = os.environ['EMBEDDING_MODEL']
RAG_LLM_MODEL = os.environ['RAG_LLM_MODEL']

refund_policy_document = 'return_policy_rag/return_policy_document.md'

return_rag_system_prompt = """Answer the customer's question using only the context provided.
Reply 'I don't know, please provide more information' if the context can't be used to answer the customer's question.
If the question is related to returns and exchanges but can't be answered by the context respond with "This is a question I am not able to answer."
If the question is not related to returns and exchanges, respond with "The information given to me does not relate to our return policies. Please rephrase."
NO hallucinating, be truthful and do not create new information that is not already described in the database or prompts.""" 


return_rag_tool_docstring = """Answer customer questions about returns and refunds, return policies and refund policies
If the customer question is not related to returns and refunds this tool will return "this is not a question i can answer"
"""

def create_retriever(generate_embeddings=False):

    gemini_embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)  # 'models/embedding-001'
    collection_name = 'return_policy_documents'
    persist_directory='./return_policy_embeddings'

    if generate_embeddings:
        with open(refund_policy_document) as file:
            policy_document = file.read()

        headers_to_split_on = [('#', 'Header 1'), ('##', 'Header 2')]
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on)

        documents = markdown_splitter.split_text(policy_document)

        vector_store = Chroma.from_documents(
            collection_name=collection_name,
            documents=documents,  # IDs added automatically 
            embedding=gemini_embeddings,
            persist_directory=persist_directory
            )
        
    else:
        vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=gemini_embeddings,
            persist_directory=persist_directory
        )


    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 3}) 
    return retriever


    # query = 'my plant was dead when it arrived'
    # print('MMR results:')
    # results = vector_store.max_marginal_relevance_search(query, k=2)
    # for res in results:
    #     print(f"{res.page_content}\n")

    # print('Similarity results:')
    # results = vector_store.similarity_search(query, k=2)
    # for res in results:
    #     print(f"{res.page_content}\n")



def create_rag_chain(retriever):

    rag_prompt = PromptTemplate(
        template="""{instructions}
        
        Use this context to answer the question:
        {context}

        Customer question:
        {question}

        Response:
        """,
        input_variables=['instructions', 'question', 'context']
    )

    llm = ChatGoogleGenerativeAI(model=RAG_LLM_MODEL, temperature=0.5)

    plant_return_rag_chain = (
        {
            'instructions': lambda x: return_rag_system_prompt,
            'context': retriever, 
            'question': RunnablePassthrough()
            }
        | rag_prompt
        | llm
        | StrOutputParser()
    )
    return plant_return_rag_chain


retriever = create_retriever(True)
refund_rag_chain = create_rag_chain(retriever)


@tool(description=return_rag_tool_docstring)
def return_policy_query(query: str) -> str:
    return refund_rag_chain.invoke(query)



if __name__ == '__main__':
    print('return RAG')
    query = 'where are my plants'
    query = 'how big is an elephant?'
    query = 'my plant was dead when it arrived'
    query = 'my order hasnt arrived'
    print('Query:', query)
    rag_result = return_policy_query.invoke(query)
    print('Rag result:', rag_result)
    

