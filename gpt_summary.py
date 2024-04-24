import re
import config
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
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

    # Map 프롬프트 완성
    map_prompt = PromptTemplate.from_template(map_template)

    # Map에서 수행할 LLMChain 정의
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

    # Map 문서를 통합하고 순차적으로 Reduce합니다.
    reduce_documents_chain = ReduceDocumentsChain(
        combine_documents_chain=combine_documents_chain,
        collapse_documents_chain=combine_documents_chain,
        token_max=4095,
    )

    # 문서들에 체인을 매핑하여 결합하고, 그 다음 결과들을 결합합니다.
    map_reduce_chain = MapReduceDocumentsChain(
        llm_chain=map_chain,
        reduce_documents_chain=reduce_documents_chain,
        document_variable_name="pages",
        return_intermediate_steps=False,
    )
    
    result = map_reduce_chain.run(split_docs)
    return result