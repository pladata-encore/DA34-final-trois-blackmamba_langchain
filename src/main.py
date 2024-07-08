import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from fastapi import FastAPI, Query
from pydantic import BaseModel
import os
import csv
import io
import threading
from pytube import YouTube
from pydub import AudioSegment
import simpleaudio as sa
import googleapiclient.discovery
from dotenv import load_dotenv
from itertools import groupby
from operator import itemgetter
import pdfplumber
from langchain.docstore.document import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
import pandas as pd
import logging
from contextlib import asynccontextmanager

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# .env 파일에서 환경 변수 로드
load_dotenv()

# OpenAI API 키 설정
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError("OPENAI_API_KEY is not set in the environment variables")

os.environ["OPENAI_API_KEY"] = api_key

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.debug("Startup event triggered")
    yield
    logger.debug("Shutdown event triggered")

app = FastAPI(lifespan=lifespan)

@app.get("/")
async def root():
    logger.debug("Root endpoint called")
    return {"message": "Hello World"}

def extract_content_from_pdf(pdf_path):
    content = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()
                if text:
                    content.append((os.path.basename(pdf_path), f"Page {page_number}", text))
                tables = page.extract_tables()
                if tables:
                    for table in tables:
                        table_text = '\n'.join(['\t'.join([str(cell) if cell is not None else '' for cell in row]) for row in table])
                        content.append((os.path.basename(pdf_path), f"Page {page_number}", table_text))
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
    return content

def process_pdfs(directory):
    documents = []
    pdf_contents = []
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            file_path = os.path.join(directory, filename)
            pdf_content = extract_content_from_pdf(file_path)
            pdf_contents.extend(pdf_content)
            for content in pdf_content:
                documents.append(Document(page_content=content[2], metadata={"source": filename, "part": content[1]}))
    return documents, pdf_contents

def process_csvs(directory):
    documents = []
    csv_contents = []
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory, filename)
            try:
                df = pd.read_csv(file_path)
                csv_text = df.to_csv(index=False)
                csv_contents.append((filename, "Full", csv_text))
                documents.append(Document(page_content=csv_text, metadata={"source": filename}))
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    return documents, csv_contents

def process_files(directory):
    pdf_documents, pdf_contents = process_pdfs(directory)
    csv_documents, csv_contents = process_csvs(directory)
    
    return pdf_documents, csv_documents, pdf_contents, csv_contents

def extract_csv_content(csv_path, specific_columns=None):
    csv_content = []
    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if specific_columns:
                selected_columns = {col: row[col] for col in specific_columns if col in row}
                csv_content.append(selected_columns)
            else:
                csv_content.append(row)
    return csv_content

def get_restaurant_list():
    restaurant_csv_path = os.path.join('/app/csvs', 'seongsu_restaurant_final.csv')
    restaurant_list = extract_csv_content(restaurant_csv_path)
    restaurant_list.sort(key=itemgetter('food_category_1st'))
    grouped_restaurant_list = []
    for key, group in groupby(restaurant_list, key=itemgetter('food_category_1st')):
        sorted_group = sorted(group, key=lambda x: float(x['rank_score']), reverse=True)
        grouped_restaurant_list.extend(sorted_group)
    return grouped_restaurant_list

def format_restaurant_info(restaurant):
    return (
        f"Title: {restaurant['title']}, "
        f"{restaurant['title']} ({restaurant['local_category']}, {restaurant['food_category_1st']}, {restaurant['food_category_2nd']}, "
        f"Information: {restaurant['information']}, "
        f"Parking: {restaurant['closest_parking_name']} ({restaurant['closest_parking_address']}), "
        f"Mon Business: {restaurant['mon_business']}, "
        f"Rank Score: {restaurant['rank_score']})"
    )

def get_formatted_restaurant_list(restaurant_list):
    return [format_restaurant_info(restaurant) for restaurant in restaurant_list]

restaurant_list = get_restaurant_list()
formatted_restaurant_list = get_formatted_restaurant_list(restaurant_list)

# ChromaDB로 벡터스토어 생성 또는 호출
persist_directory = "/app/projectDB/"

if not os.path.exists(persist_directory):
    pdf_documents, csv_documents, pdf_contents, csv_contents = process_files('/app/pdfs')
    documents = pdf_documents + csv_documents
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    embedding = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(documents=texts, embedding=embedding, persist_directory=persist_directory)
    vectordb.persist()
else:
    print("Persist directory found. Loading existing vector store.")
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=OpenAIEmbeddings())
    print("Vector store loaded.")

# 4. 응답 및 쿼리 처리
def extract_music_keyword(query):
    music_synonyms = ["음악 재생", "틀어 줘", "틀어줘", "들려줘", "들려 줘", "플레이 해줘", "재생 해줘",
                      "음악", "재생", "틀어", "들려", "플레이"]
    for synonym in music_synonyms:
        if synonym in query:
            keyword = query.split(synonym)[0].strip()
            return keyword
    return query

def process_llm_response(query):
    keyword = extract_music_keyword(query)
    return play_music_request(keyword)

# 응답 유형을 결정하는 함수
def determine_response_type(query: str):
    music_synonyms = ["음악 재생", "틀어 줘", "틀어줘", "들려줘", "들려 줘", "플레이 해줘", "재생 해줘",
                      "음악", "재생", "틀어", "들려", "플레이"]
    restaurant_names = [restaurant['title'] for restaurant in restaurant_list]
    restaurant_synonyms = ["맛집 추천", "맛집 알려줘", "맛집 추천해 줘", "맛집 알려 줘", "추천", "맛집", "식당",
                           "양식", "한식", "카페", "중식", "일식", "아시안", "술집",
                           "햄버거", "백반", "파스타", "곰탕", "베이커리", "카페", "케이크", "우육면", "커피", 
                           "소바", "오므라이스", "베이글", "족발", "닭요리", "돈카츠", "일정식", "솥밥", "디저트", 
                           "우동", "와인바", "국수", "카레", "쌀국수", "고깃집", "보쌈", "샐러드", "초밥", 
                           "피자", "브런치", "중식당", "칵테일", "맥주", "딤섬", "막걸리", "삼겹살", 
                           "오코노미야끼", "치킨", "부대찌개", "태국", "짜장면", "양꼬치", "해물", "회", 
                           "순대", "냉면", "분식", "규카츠", "오마카세", "라멘", "마라탕", "짬뽕", 
                           "샌드위치", "김밥", "소시지", "국밥", "샤브샤브", "스테이크", "타코", "아시안요리", 
                           "에스프레소", "아이스크림", "갈비", "곱창", "비건"]

    if any(synonym in query for synonym in music_synonyms):
        return "music"
    elif any(synonym in query for synonym in restaurant_synonyms) or any(name in query for name in restaurant_names):
        return "restaurant"
    else:
        return "car"

# 음악 재생 기능 추가
os.environ["GOOGLE_API_KEY"] = os.getenv('GOOGLE_API_KEY')
youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=os.getenv("GOOGLE_API_KEY"))

music_thread = None
stop_event = threading.Event()

def search_music(keyword):
    search_response = youtube.search().list(
        q=keyword,
        part="id",
        type="video",
        maxResults=1
    ).execute()
    
    video_id = search_response["items"][0]["id"]["videoId"]
    return video_id

def download_audio(video_id):
    try:
        yt = YouTube(f"https://www.youtube.com/watch?v={video_id}")
        stream = yt.streams.filter(only_audio=True).first()
        if not stream:
            raise Exception("No audio stream available")
        audio_data = io.BytesIO()
        stream.stream_to_buffer(audio_data)
        audio_data.seek(0)
        return audio_data, None
    except Exception as e:
        print(f"Error downloading audio: {e}")
        return None, str(e)

# def play_music_thread(audio_data):
#     try:
#         if not audio_data:
#             print("No audio data to play")
#             return
#         audio = AudioSegment.from_file(audio_data)
#         stop_event.clear()
#         play_obj = sa.play_buffer(audio.raw_data, num_channels=audio.channels,
#                                   bytes_per_sample=audio.sample_width, sample_rate=audio.frame_rate)
#         while play_obj.is_playing():
#             if stop_event.is_set():
#                 play_obj.stop()
#                 break
#     except Exception as e:
#         print(f"Error playing audio: {e}")

def play_music_request(keyword):
    try:
        video_id = search_music(keyword)
        audio_data, error = download_audio(video_id)
        if audio_data:
            response_text = f"'{keyword}'을(를) 검색할게요."
            print(response_text)
            # global music_thread
            # music_thread = threading.Thread(target=play_music_thread, args=(audio_data,))
            # music_thread.start()
            return response_text
        else:
            response_text = f"'{keyword}'을(를) 재생할 수 없습니다. 오류: {error}"
            print(response_text)
            return response_text
    except Exception as e:
        response_text = f"음악을 재생하는 중에 문제가 발생했습니다. 다시 시도해주세요. 오류: {str(e)}"
        print(response_text)
        return response_text

def stop_music():
    global stop_event, music_thread
    if music_thread and music_thread.is_alive():
        stop_event.set()
        music_thread.join()
        print("음악 재생을 중지했습니다.")
    else:
        print("현재 재생 중인 음악이 없습니다.")

@app.get("/query")
async def query_get(query: str):
    response_type = determine_response_type(query)
    if response_type == "music":
        answer = process_llm_response(query)
    else:
        # Load the QA chain with GPT-4
        llm = ChatOpenAI(model="gpt-4o")

        prompt_template = """
        안녕하세요 저는 드라이브톡입니다. 차량에 대한 정보, 음악재생, 성수동 맛집 정보를 정확히 제공해 드리는 시스템입니다.
        Context:{context}
        Question: {question}
        """

        prompt = PromptTemplate(input_variables=["context", "question"],
                                template=prompt_template)

        # QA 체인 로드
        qa_chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
        docs = vectordb.similarity_search(query, k=5)
        answer = qa_chain.run(input_documents=docs, question=query)
    return {"type": response_type, "answer": answer}

@app.get("/stop_music")
async def stop_music_endpoint():
    stop_music()
    return {"response": "음악 재생을 중지했습니다."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
