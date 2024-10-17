import warnings
warnings.filterwarnings("ignore")
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator, ValidationInfo
from fastapi.concurrency import iterate_in_threadpool
from starlette.background import BackgroundTask
from collections.abc import AsyncIterable, Iterable

import os
import uuid
import time
import asyncio
import shutil
import yaml
import json
from pathlib import Path
from typing import Optional
import boto3
from botocore.exceptions import ClientError
import atexit
import watchtower
from tenacity import retry, stop_after_attempt, wait_exponential

from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
from generate import Generate

from secrets_manager import get_secret

import logging
logging.getLogger('asyncio').setLevel(logging.WARNING)

with open(Path('config/config.yaml').resolve()) as f:
    config = yaml.safe_load(f)

ec2_client = boto3.Session().resource('ec2', region_name=config['aws_region'])
instance_id = ec2_client.meta.client.describe_instances()['Reservations'][0]['Instances'][0]['InstanceId']

cloudwatch_client = boto3.client(
            'logs',
            aws_access_key_id=config['aws_access_key_id'],
            aws_secret_access_key=config['aws_secret_access_key'],
            region_name=config['aws_region']
        )
cloudwatch_handler = watchtower.CloudWatchLogHandler(
        log_group=config['aws_cloudwatch_log_group'],
        stream_name=instance_id,
        boto3_client=cloudwatch_client,
        use_queues=False
    )
logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[cloudwatch_handler]
    )

logger = logging.getLogger(__name__)

def flush_logs():
    for handler in logging.getLogger().handlers:
        if isinstance(handler, watchtower.CloudWatchLogHandler):
            handler.flush()

atexit.register(flush_logs)

# class JSONStreamingResponse(StreamingResponse, JSONResponse):
#     """StreamingResponse that also render with JSON."""

#     def __init__(
#         self,
#         content: Iterable | AsyncIterable,
#         status_code: int = 200,
#         headers: dict[str, str] | None = None,
#         media_type: str | None = None,
#         background: BackgroundTask | None = None,
#     ) -> None:
#         if isinstance(content, AsyncIterable):
#             self._content_iterable: AsyncIterable = content
#         else:
#             self._content_iterable = iterate_in_threadpool(content)

#         async def body_iterator() -> AsyncIterable[bytes]:
#             async for content_ in self._content_iterable:
#                 if isinstance(content_, BaseModel):
#                     content_ = content_.model_dump()
#                 yield self.render(content_)

#         self.body_iterator = body_iterator()
#         self.status_code = status_code
#         if media_type is not None:
#             self.media_type = media_type
#         self.background = background
#         self.init_headers(headers)

app = FastAPI(
    title="AltGAN API",
    version="1.0",
)

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response

secret = get_secret(config)

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def openai_with_retry(*args, **kwargs):
    return OpenAI(*args, **kwargs)

llm = openai_with_retry(model=config['openai_models']['small']['version'], api_key=secret['openai_api_key'], 
                            temperature=0.0, top_p=0.4, max_tokens=config['openai_models']['small']['max_tokens'])

async def cleanup_models():
    try:
        start_hist = time.perf_counter()
        chat_summ_file = f"{config['chat_summ_bucket']}/{Generate.chat_id}.md"

        Generate.chat_hist = Generate.openai_llm.complete(config['PROMPTS']['HISTORY_SUMMARIZER_PROMPT']
                                            .format(Generate.chat_hist)).text.strip()
        end_hist = time.perf_counter()
        logger.info(f"Time taken to generate Chat History: {end_hist-start_hist} secs.")

        start_put_obj = time.perf_counter()
        Generate.s3.put_object(
            Bucket=config["aws_s3_bucket"],
            Key=chat_summ_file,
            Body=Generate.chat_hist,
        )

        end_put_obj = time.perf_counter()
        logger.info(f"S3 Put Object Time Taken: {end_put_obj-start_put_obj} secs.")

    except ClientError as err:
        raise ValueError(
            f"Error in modification of {chat_summ_file} file in S3..."
        ) from err
    
    logger.info("Response Generation Completed")


# Settings.llm = llm
# def random_choice() -> str:
#     return "".join(random.choices(string.ascii_lowercase + string.digits + string.ascii_uppercase, k=8))

class RAG(BaseModel):
    chat_id: str = Field(default="zpf87cm9")
    query: str = Field(...)
    category: str = Field(...)
    file_name: Optional[str] = Field(default="")
    # keyword: Optional[str] = Field(default="")
    collection_name: Optional[str] = Field(default="rag_llm")
    persist_dir: Optional[str] = Field(default="persist")

    @field_validator('chat_id')
    @classmethod
    def validate_id(cls, v: str, info: ValidationInfo) -> str:
        if not v.isalnum():
            raise HTTPException(status_code=403, 
                                detail=f'Invalid {info.field_name}. Chat ID must be alphanumeric.')
        return v
    
    @field_validator('query')
    @classmethod
    def validate_id(cls, v: str, info: ValidationInfo) -> str:
        if llm.complete(config['PROMPTS']['PROFANITY_FILTER_PROMPT'].format(v)).text.strip() == "True":
            raise HTTPException(status_code=406, 
                                detail=f'Sorry, I won\'t be able to answer your query.')
        return v
    
    @field_validator('category')
    @classmethod
    def validate_category(cls, v: str, info: ValidationInfo) -> str:
        if v not in ["finance", "oil_gas"]:
            raise HTTPException(status_code=403, 
                    detail=f'Invalid {info.field_name}. Category must be either `Finance` or `Oil and Gas`.')
        return v


# metadata_dict = {"page_num": "metadata.page_num", "file_name": "metadata.file_name", 
#                  "category": "metadata.category", "keywords": "metadata.keywords"}

metadata = {}
category_mapping = {"oil_gas": "Oil and Gas", "finance": "Finance"}

@app.get("/health")
async def health_check():
    return {"status": "OK"}


@app.post("/v1/chat")
async def get_answer(rag: RAG, background_tasks: BackgroundTasks) -> StreamingResponse:
    # background_tasks.add_task(cleanup_models)

    try:
        start_greet = time.perf_counter()
        greeting_classifier = llm.complete(config['PROMPTS']['GREETING_CLASSIFIER_PROMPT'].format(rag.query)).text.strip()
        logger.info(f"GREETING_CLASSIFIER_PROMPT: {greeting_classifier}")
        
        if greeting_classifier == "True":
            generic_resp = llm.complete(config['PROMPTS']['GREETING_PROMPT'].format(rag.query)).text.strip()    
            return StreamingResponse(json.dumps({"response_id": str(uuid.uuid4()), "type": "greeting", "text": generic_resp}), media_type="text/event-stream")
            # return JSONStreamingResponse({"response_id": str(uuid.uuid4()), "type": "greeting", "text": generic_resp})
        

        # if rag.keyword:
        #     metadata["keyword"] = rag.keyword

        if rag.file_name:
            metadata["file_name"] = rag.file_name
            
        metadata["category"] = rag.category

        end_greet = time.perf_counter()
        logger.info(F"Time taken for greeting: {end_greet-start_greet} secs.")

        logger.info("Running generate.py script!")
        generate_obj = Generate(config=config, chat_id=rag.chat_id, query=rag.query, category_name=category_mapping[rag.category], metadata=metadata,
                                persist_dir=rag.persist_dir, collection_name=rag.collection_name)
        
        background_tasks.add_task(cleanup_models)
        
        
        response = generate_obj.generate_answer()

        if response is not None:
            return StreamingResponse(response, media_type="text/event-stream")
            # return JSONStreamingResponse(response)
        
    except Exception as err:
        logger.error(f"Error processing request: {err}")
        raise HTTPException(status_code=500, detail=f"Server Error: {err}")
            

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=False,
        workers=1,
    )