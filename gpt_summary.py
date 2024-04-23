from langchain.chains.mapreduce import MapReduceChain
from langchain.chains import ReduceDocumentsChain, MapReduceDocumentsChain
from langchain_community.chat_models import ChatOpenAI
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.schema.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
import config

def remove_brackets(text):
    text_without_brackets = re.sub(r'\[.*?\]', '', text)
    text_without_spaces = re.sub(r'\s{2,}', '', text_without_brackets)
    return text_without_spaces

def gpt_process(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
        length_function=len,
    )

    docs = [Document(page_content=x) for x in text_splitter.split_text(remove_brackets(text))]
    split_docs = text_splitter.split_documents(docs)
    
    llm = ChatOpenAI(temperature=0, openai_api_key=config.OPENAI_API_KEY)

    # Map prompt
    map_template = """The following is a set of documents
    {docs}
    Based on this list of docs, please identify the main themes
    Helpful Answer:"""

    map_prompt = PromptTemplate.from_template(map_template)

    # Reduce prompt
    reduce_template = """The following is set of summaries:
    {doc_summaries}
    Take these and distill it into a final, consolidated summary of the main themes.
    The final answer is a single paragraph of about 150~200 words and must be in Korean.
    Helpful Answer:"""

    reduce_prompt = PromptTemplate.from_template(reduce_template)
    
    # 1. Reduce chain
    reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

    # Takes a list of documents, combines them into a single string, and passes this to an LLMChain
    combine_documents_chain = StuffDocumentsChain(
        llm_chain=reduce_chain, document_variable_name="doc_summaries"
    )

    # Combines and iteravely reduces the mapped documents
    reduce_documents_chain = ReduceDocumentsChain(
        combine_documents_chain=combine_documents_chain,
        collapse_documents_chain=combine_documents_chain,
        token_max=4000,
    )

    # 2. Map chain
    map_chain = LLMChain(llm=llm, prompt=map_prompt)

    # Combining documents by mapping a chain over them, then combining results
    map_reduce_chain = MapReduceDocumentsChain(
        llm_chain=map_chain,
        reduce_documents_chain=reduce_documents_chain,
        document_variable_name="docs",
        return_intermediate_steps=False,
    )
    sum_result = map_reduce_chain.run(split_docs)
    
    return sum_result


# test = """[0:00:00 ~ ]  뉴욕 경찰서에 근무하는 리차이드 프레저
# [0:00:03 ~ ]  그는 경찰관님과 통시에 풀어복소로 활약하고 있습니다.
# [0:00:07 ~ ]  그리고 다음 경기가 잡혔다는 소식을 듣습니다.
# [0:00:09 ~ ]  하지만 문제가 생겼습니다.
# [0:00:11 ~ ]  다음 상대가 바로 이 사람이었죠.
# [0:00:14 ~ ]  로이전스, 진여.
# [0:00:15 ~ ]  오늘은 뉴욕 경찰관 출신 복서, 리차이드 프레저
# [0:00:18 ~ ]  그리고 로이전 준여의 경기 이야기를 준비했습니다.
# [0:00:21 ~ ]  우선 오늘 먼저 소개해드릴 선수는 로이전 준여입니다.
# [0:00:24 ~ ]  두 마라 분 이 바쁜 당시 최고의 실력을 가진 선수였습니다.
# [0:00:28 ~ ]  그는 이미 6년 전 1993년 뻔나도피스을 판정으로 이기고
# [0:00:32 ~ ]  미디급 챔피언 밸드를 얻었고 1994년에는
# [0:00:35 ~ ]  당대 또 다른 복싱천재, james tony를 이기고 슈퍼미디급 챔피언에 등극합니다.
# [0:00:40 ~ ]  그는 이미 미디급 슈퍼미디급 챔피언제를 차지했었고
# [0:00:44 ~ ]  슈퍼미디급에서 5번의 타이틀 방어전 성공중이었습니다.
# [0:00:48 ~ ]  그리고 1997년에는 몬테이 그리핀은 1라운드 케어 시키고
# [0:00:52 ~ ]  라이트에 비급에서 맞아 챔피언에 등극합니다.
# [0:00:55 ~ ]  심지어 98년에는 루도 벨라를 이기고 라이트에 비급 두 개기고 통합 챔피언에 등극합니다.
# [0:01:03 ~ ]  그리고 1999년 그의 라이트에 비급 통합타이틀 두 번째 방어전 상대로
# [0:01:08 ~ ]  특이한 경력을 가진 상대 리채드 프레저를 만납니다.
# [0:01:11 ~ ]  당시 리채드 프레저�는 124승 오페라는 아마처 경력을 가지고 있었고
# [0:01:17 ~ ]  세 번의 골든 글로버스 상 그리고 92년 첫 데뷔 하여
# [0:01:21 ~ ]  99년 존수와의 경기 전까지 18승 상패라는 나쁘지 않은 선적을 가지고 있었습니다.
# [0:01:27 ~ ]  무엇보다도 그의 또다는 직업이 무료, 뉴욕 경찰이였습니다.
# [0:01:33 ~ ]  99년 1월 천재 복서 그리고 뉴욕 경찰 두 선수간 링 위에서 맞아 합니다.
# [0:01:40 ~ ]  멋들어진 제복을 입고 난 프레저를 과연 링 위에서 그의 뉴욕 경찰의 파우가 나올 수 있을지요.
# [0:01:46 ~ ]  경기 시작합니다. 검색 트럭카 프레저를 황금색이 로위조준이에요입니다.
# [0:01:52 ~ ]  따로 위조준 준위원 나이카럼 쯰부로 기선지 앞 들어갑니다.
# [0:01:55 ~ ]  와 창처럼 들어간 로위조인쇼쯰부
# [0:02:00 ~ ]  그리고 이 여진을 로위조인쇼 번개 같은 연타
# [0:02:03 ~ ]  정말 빠릅니다. 일라운드 끝나갑니다.
# [0:02:06 ~ ]  바아 존스의 라이트 후 위험했습니다 프레저를.
# [0:02:10 ~ ]  존스의 스윗레트 나이카로웠습니다.
# [0:02:14 ~ ]  와 존스의 원투 그리고 후 다운덩한 프레저를 이곳을 정탄은 없던 것 같은데요.
# [0:02:20 ~ ]  이런 아미다 프레저를 한 번 다운덩만 보겠습니다.
# [0:02:23 ~ ]  원 투
# [0:02:25 ~ ]  아 이거 로위조인쇼 기세에 눌려쓰러지는 건가에 프레저를
# [0:02:30 ~ ]  뉴욕 경찰도 링 위에서는 힘듭니다.
# [0:02:33 ~ ]  이런 나고 경기 다시 진행됩니다.
# [0:02:36 ~ ]  프레저를 견단하게 라운드 끝납니다.
# [0:02:38 ~ ]  첫번째 다운덩면 다시 보겠습니다.
# [0:02:40 ~ ]  여기서 원 투
# [0:02:42 ~ ]  다 바리 미끄러져서 넘어진 듯 보입니다.
# [0:02:47 ~ ]  이라운드입니다. 상대축역한 로위조수 와 유도탄처럼 주목이 들어갑니다.
# [0:02:53 ~ ]  와 후 날카라웠습니다.
# [0:02:56 ~ ]  얻은 이라운드 1분 정도 남았습니다.
# [0:02:59 ~ ]  로위조인쇼 쥐프 쥐고 라이트 바디 어퍼까지 정말 빠릅니다.
# [0:03:06 ~ ]  자 존스의 스윗레트와 카메라에 담기지 않는 스피드입니다.
# [0:03:10 ~ ]  쥐프 쥐고 아 양국 다 랩툭 두 번 들어가구요.
# [0:03:14 ~ ]  정신 없습니다 상대.
# [0:03:16 ~ ]  이라운드 끝나갑니다.
# [0:03:18 ~ ]  로위조준 이어 여유롭습니다.
# [0:03:21 ~ ]  우와 상대 쥐프 쥐프 쥐프 쥐프 두 번째 다운디입니다.
# [0:03:25 ~ ]  이거 일어날 수 있을까요?
# [0:03:28 ~ ]  네 일어나지만 주심 경기 끝납니다.
# [0:03:31 ~ ]  아 이거 다운덩면 보겠습니다.
# [0:03:33 ~ ]  여기서 쥐프 쥐프 쥐프 랩틀 그리고 따라 들어가서 원 투
# [0:03:37 ~ ]  여기서 스피하고 랩틀.
# [0:03:39 ~ ]  정말 좋았습니다.
# [0:03:42 ~ ]  프레저는 이경기 이후 은퇴를 발표합니다.
# [0:03:45 ~ ]  그리고 로위조 준현은 이경기 후 3년대 클렌트 킬리를 이기며 무려 라이트
# [0:03:50 ~ ]  비 끝 7개기고 통합 챔피언에 등극합니다.
# [0:03:54 ~ ]  오늘도 끝까지 봐셔서 정말 감사합니다.
# [0:03:56 ~ ]  쥐어고도군 더욱 감사합니다.
# """
# print(gpt_process(test))
