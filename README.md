# DriveChat AI Assistant

DriveChat은 차량 내 AI 비서 시스템으로, 차량 정보 제공, 음악 재생, 그리고 성수동 맛집 추천 기능을 제공합니다. FastAPI를 기반으로 구축되었으며, 자연어 처리를 위해 LangChain과 OpenAI의 GPT-4o를 활용합니다.

## 주요 기능

1. **차량 정보 제공**: 사용자의 질문에 대해 차량 관련 정보를 제공합니다.
2. **음악 재생**: YouTube API를 통해 음악을 검색하고 재생합니다.
3. **성수동 맛집 추천**: 지역 맛집 정보를 CSV 파일에서 추출하여 추천합니다.

## 기술 스택

- FastAPI
- LangChain
- OpenAI GPT-4o
- YouTube API
- Docker
- Python 3.9

## 설치 및 실행

### 환경 설정

1. `.env` 파일을 생성하고 필요한 API 키를 설정합니다:

```
OPENAI_API_KEY=your_openai_api_key
GOOGLE_API_KEY=your_google_api_key
```

### Docker를 이용한 실행

1. Docker 이미지 빌드:

```bash
docker build -t drivechat-ai -f Dockerfile.cloud .
```

2. Docker 컨테이너 실행:

```bash
docker run -p 8000:8000 -v $(pwd)/.env:/app/.env drivechat-ai
```

### Docker Compose를 이용한 실행

1. `docker-compose.yaml` 파일이 있는 디렉토리에서 다음 명령어를 실행합니다:

```bash
docker-compose up --build
```

## API 엔드포인트

- `GET /`: 헬스 체크
- `GET /query`: 사용자 질의 처리
- `GET /stop_music`: 음악 재생 중지

## 프로젝트 구조

```
.
├── src/
│   ├── main.py
│   └── requirements.txt
├── csvs/
│   └── seongsu_restaurant_final.csv
├── pdfs/
│   └── (차량 관련 PDF 문서들)
├── projectDB/
│   └── (벡터 데이터베이스 파일들)
├── Dockerfile.cloud
├── docker-compose.yaml
└── README.md
```

## 개발 및 기여

이 프로젝트는 지속적으로 개발 중입니다. 버그 리포트, 기능 제안, 또는 풀 리퀘스트는 언제나 환영합니다.

## 라이센스

이 프로젝트는 MIT 라이센스 하에 배포됩니다. 자세한 내용은 `LICENSE` 파일을 참조하세요.
