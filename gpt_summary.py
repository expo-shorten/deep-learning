import re
import config
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import ReduceDocumentsChain
from langchain.chains import MapReduceDocumentsChain
from langchain.schema.document import Document

def remove_brackets(text):
    text_without_brackets = re.sub(r'\[.*?\]', '', text)
    text_without_spaces = re.sub(r'\s{2,}', ' ', text_without_brackets)
    return text_without_spaces

def summary_text(text):
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator=".",  # 분할기준
        chunk_size=4000,   # 사이즈
        chunk_overlap=50, # 중첩 사이즈
    ) 
    text = remove_brackets(''.join(text))
    
    split_docs = text_splitter.split_text(text)
    split_docs = text_splitter.create_documents(split_docs)

    map_template = """Please divide the article below into main topics and summarize them naturally. The answers should be provided in Korean.
    {pages}
    answer:"""

    map_prompt = PromptTemplate.from_template(map_template)

    llm = ChatOpenAI(temperature=0, 
                    model_name='gpt-3.5-turbo', api_key=config.OPENAI_API_KEY)
    map_chain = LLMChain(llm=llm, prompt=map_prompt)

    reduce_template = """Please divide the article below into main topics and summarize them naturally. The answers should be provided in Korean.
    {doc_summaries}
    answer:"""

    reduce_prompt = PromptTemplate.from_template(reduce_template)

    reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

    combine_documents_chain = StuffDocumentsChain(
        llm_chain=reduce_chain,                
        document_variable_name="doc_summaries"
    )

    reduce_documents_chain = ReduceDocumentsChain(
        combine_documents_chain=combine_documents_chain,
        collapse_documents_chain=combine_documents_chain,
        token_max=4095,
    )

    map_reduce_chain = MapReduceDocumentsChain(
        llm_chain=map_chain,
        reduce_documents_chain=reduce_documents_chain,
        document_variable_name="pages",
        return_intermediate_steps=False,
    )
    
    result = map_reduce_chain.run(split_docs)
    return result


def request_message(text, question):
    print('추가적인 답변 생성 준비 중...')
    llm = ChatOpenAI(temperature=0, 
                    model_name='gpt-3.5-turbo', 
                    api_key=config.OPENAI_API_KEY)
    
    map_template = f"""Please answer the summary below according to the user's question. The answer should be provided in Korean.

    {text}

    {question}

    answer:"""

    ans = llm.predict(map_template)
    print('답변 생성 완료')
    
    return ans