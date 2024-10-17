import warnings
warnings.filterwarnings("ignore")
import os
import time
import argparse
import string
import random
import tiktoken
import yaml
import re
import uuid
import nest_asyncio
import boto3
from botocore.exceptions import ClientError
from urllib.parse import quote
from pathlib import Path
from typing import Optional, Tuple, Generator
import json
import logging

import qdrant_client
from llama_index.core import StorageContext, Settings, load_index_from_storage
from llama_index.core.schema import QueryBundle
# from llama_index.core.memory import ChatMemoryBuffer
# from llama_index.core.storage.chat_store import SimpleChatStore
from llama_index.core.query_engine import CitationQueryEngine
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter
from llama_index.llms.openai import OpenAI
from llama_index.llms.together import TogetherLLM
from llama_index.core.prompts import PromptTemplate

from tenacity import retry, stop_after_attempt, wait_exponential
from secrets_manager import get_secret

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger(__name__)

nest_asyncio.apply()

# CITATION_TEMPLATE = """
#     Please provide an answer based solely on the provided sources. \
#     answer must be concise and have empathetic tone. answer must not contain words such as `Based on the provided sources` and `In this context`. \
#     generate markdown-rich answer tokens based on the provided sources (as sources are in markdown format). \
#     answer must not contain `<|ANSWER|>:`, `<|HISTORY|>:` or `<|QUERY|>:` tags.
#     When referencing information from a source, cite the appropriate source(s) using their corresponding numbers. \
#     Every answer should include at least one source citation. Only cite a source when you are explicitly referencing it. \
#     If none of the sources are helpful, you should indicate that. \
#     For example:
#     Source 1:
#     In a world where magic is real, a rare gemstone known as the **Dragon's Eye** is discovered to possess immense power.
#     Source 2:
#     The Dragon's Eye is sought after by various factions, including a secretive organization called the *Crimson Order*.
#     Source 3:
#     A young adventurer named Aria embarks on a quest to find the Dragon's Eye before it falls into the wrong hands.
#     <|QUERY|>: What is the significance of the Dragon's Eye, and why is Aria's quest to find it so important?
#     <|ANSWER|>: `Aria's quest` to find the immensely powerful **Dragon's Eye** [1] \
#     is crucial to prevent the secretive *Crimson Order* [2], \
#     and other factions from obtaining the rare gemstone and potentially misusing its magic [3].
#     """

# QA_TEMPLATE = """
#     Generate markdown-rich answer tokens based on the provided sources (as sources are in markdown format). \
#     apply markdown formatting in answer such as bold, italics, headings, ordered lists, images, links etc. \
#     do not convert citations or references such as [1] into footnotes in markdown. \
#     Now it's your turn. Below are several numbered sources of information:
#     ------
#     {context_str}
#     ------
#     {query_str}
#     <|ANSWER|>: 
#     """

# REFINE_TEMPLATE = """
#     Now it's your turn. We have provided an existing answer: {existing_answer}
#     Below are several numbered sources of information. Use them to refine the existing markdown-rich formatted answer. \
#     If the provided sources are not helpful, you will repeat the existing answer.
#     Begin refining!
#     ------
#     {context_msg}
#     ------
#     {query_str}
#     <|ANSWER|>: 
#     """

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def openai_with_retry(*args, **kwargs):
    return OpenAI(*args, **kwargs)

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def together_with_retry(*args, **kwargs):
    return TogetherLLM(*args, **kwargs)

class Generate:
    s3 = None
    embed = None
    openai_llm = None
    tg_llm = None

    citation_template = None
    qa_template = None
    refine_template = None

    chat_id = None
    chat_hist = None

    def __init__(self, config: dict, chat_id: str, query: str, category_name: str, persist_dir: str, collection_name: str, metadata: Optional[dict] = {}) -> None:
        start_init = time.perf_counter()

        self.config = config
        self.secret = get_secret(self.config)
        self.chat_id = chat_id
        self.query = query
        self.category_name = category_name
        # self.chat_summ_file = f"{self.config['chat_summ_bucket']}/{self.chat_id}.md"
        
        self._load_templates()

        self.persist = os.path.join(Path(persist_dir).resolve(), collection_name)

        # Path(chat_hist).mkdir(parents=True, exist_ok=True)
        # self.chat_logs = Path(f"{chat_hist}/chat_{chat_id}.md").resolve()
        # self.chat_logs = f"{chat_hist}/{chat_id}.md"

        self._load_models()

        # self.embed = HuggingFaceEmbedding(model_name=self.config['embed_model'])
        # self.openai_llm = OpenAI(model=self.config['openai_model'], api_key=self.config['openai_api_key'], 
        #                     temperature=0.7, max_tokens=2048)
        # self.tg_llm = TogetherLLM(model=self.config['tg_model'], api_key=self.config['tg_api_key'], 
        #                      temperature=0.8, max_tokens=1024)
        
        # Settings.llm = self.openai_llm
        # Settings.embed_model = self.embed

        # self.encoding = tiktoken.encoding_for_model(self.config['openai_models']['multimodal']['version'])

        self._chat_history()

        if Generate.chat_hist is not None:
            self.refined_query = "<|CHAT HISTORY|>: " + Generate.chat_hist + "\n<|QUERY|>: " + self.query
        
        else:
            self.refined_query = "<|QUERY|>: " + self.query

        client = qdrant_client.QdrantClient(url=self.config['qdrant_url'], api_key=self.secret['qdrant_api_key']) # location=":memory:"
        vector_store = QdrantVectorStore(client=client, collection_name=collection_name)
        
        storage_context = StorageContext.from_defaults(persist_dir=self.persist, vector_store=vector_store)
        index = load_index_from_storage(storage_context)

        # if not Path(f"chat_logs/chat_{chat_id}.json").exists():
        #     self.chat_store = SimpleChatStore()
        # else:
        #     self.chat_store = SimpleChatStore.from_persist_path(persist_path=Path(f"chat_logs/chat_{chat_id}.json").resolve())
        # memory = ChatMemoryBuffer.from_defaults(token_limit=8192, chat_store=self.chat_store, 
        #                                         chat_store_key=chat_id, llm=openai_llm)
        
        sim_processor = SimilarityPostprocessor(similarity_cutoff=0.70)
        rerank = CohereRerank(api_key=self.secret['cohere_api_key'], model=self.config['cohere_model'], top_n=6)
        
        filters = []
        if metadata:
            for key, value in metadata.items():
                filters.append(MetadataFilter(key=key, value=value))


        # self.chat_engine = index.as_chat_engine(
        #     chat_mode="context",
        #     memory=memory,
        #     similarity_top_k=6,
        #     node_postprocessors=[rerank],
        #     filters=MetadataFilters(filters=filters),
        #     llm=openai_llm
        # )
        
        # self.query_engine = CitationQueryEngine.from_args(
        #                 index,
        #                 embed_model=self.embed,
        #                 chat_mode="context",
        #                 citation_chunk_size=1024,
        #                 citation_chunk_overlap=32,
        #                 citation_qa_template=PromptTemplate(CITATION_TEMPLATE + QA_TEMPLATE),
        #                 citation_refine_template=PromptTemplate(CITATION_TEMPLATE + REFINE_TEMPLATE),
        #                 similarity_top_k=6,
        #                 node_postprocessors=[rerank],
        #                 filters=MetadataFilters(filters=filters),
        #                 llm=self.openai_llm,
        #                 streaming=True,
        # )

        # self.query_engine = CitationQueryEngine.from_args(
        #         index,
        #         embed_model=self.embed,
        #         chat_mode="context",
        #         citation_chunk_size=1024,
        #         citation_chunk_overlap=32,
        #         citation_qa_template=PromptTemplate(self.citation_template + self.qa_template),
        #         citation_refine_template=PromptTemplate(self.citation_template + self.refine_template),
        #         similarity_top_k=10,
        #         node_postprocessors=[rerank, sim_processor],
        #         filters=MetadataFilters(filters=filters),
        #         llm=self.openai_llm,
        #         streaming=True,
        # )

        self.query_engine = CitationQueryEngine.from_args(
                index,
                embed_model=self.embed,
                chat_mode="context",
                citation_chunk_size=1024,
                citation_chunk_overlap=32,
                citation_qa_template=PromptTemplate(self.citation_template + self.qa_template),
                citation_refine_template=PromptTemplate(self.citation_template + self.refine_template),
                similarity_top_k=20,
                node_postprocessors=[rerank, sim_processor],
                filters=MetadataFilters(filters=filters),
                llm=self.openai_llm,
                streaming=True,
        )

        end_init = time.perf_counter()
        logger.info(f"Time taken to initialize Generate class: {end_init-start_init} secs.")

    def _load_models(self):
        if Generate.embed is None:
            Generate.embed = HuggingFaceEmbedding(model_name=self.config['hf_embed'])

        if Generate.openai_llm is None:
            Generate.openai_llm = openai_with_retry(model=self.config['openai_models']['multimodal']['version'], api_key=self.secret['openai_api_key'], 
                            temperature=0.7, top_p=0.7, max_tokens=self.config['openai_models']['multimodal']['max_tokens'])
            
        if Generate.tg_llm is None:
            Generate.tg_llm = together_with_retry(model=self.config['tg_model'], api_key=self.secret['tg_api_key'], 
                             temperature=0.8, max_tokens=1024)
        
        if Generate.s3 is None:
            Generate.s3 = boto3.client(
                        "s3",
                        aws_access_key_id=self.config['aws_access_key_id'],
                        aws_secret_access_key=self.config['aws_secret_access_key'],
                        region_name=self.config['aws_region']
                    )
        
        self.embed = Generate.embed
        self.openai_llm = Generate.openai_llm
        self.tg_llm = Generate.tg_llm
        self.s3 = Generate.s3

        Settings.llm = self.openai_llm
        Settings.embed_model = self.embed
    
    def _load_templates(self):
        with open('config/prompts.json', 'r', encoding='utf-8') as f:
            prompts = json.load(f)

        if Generate.citation_template is None:
            with open(prompts["templates"]["CITATION_TEMPLATE"][1]["prompt"], 'r', encoding='utf-8') as f:
                Generate.citation_template = f.read()
        
        if Generate.qa_template is None:
            with open(prompts["templates"]["QA_TEMPLATE"][1]["prompt"], 'r', encoding='utf-8') as f:
                Generate.qa_template = f.read()
        
        if Generate.refine_template is None:
            with open(prompts["templates"]["REFINE_TEMPLATE"][1]["prompt"], 'r', encoding='utf-8') as f:
                Generate.refine_template = f.read()

        self.citation_template = Generate.citation_template
        self.qa_template = Generate.qa_template
        self.refine_template = Generate.refine_template

    def _chat_history(self):
        if Generate.chat_id != self.chat_id:
            Generate.chat_hist = None

        if Generate.chat_id is None or Generate.chat_id != self.chat_id or Generate.chat_hist is None:
            Generate.chat_id = self.chat_id
            chat_summ_file = f"{self.config['chat_summ_bucket']}/{Generate.chat_id}.md"

            s3_chat_summ_obj = self.s3.list_objects_v2(
                Bucket=self.config["aws_s3_bucket"], Delimiter="/", Prefix=f"{self.config['chat_summ_bucket']}/"
            )
            if "Contents" not in s3_chat_summ_obj:
                self.s3.put_object(Bucket=self.config["aws_s3_bucket"], Key=f"{self.config['chat_summ_bucket']}/")

            elif "Contents" in s3_chat_summ_obj and chat_summ_file in [
                    content["Key"] for content in s3_chat_summ_obj["Contents"]
                ]:
                try:
                    s3_chat_summ_resp = self.s3.get_object(
                        Bucket=self.config["aws_s3_bucket"], Key=chat_summ_file
                    )
                    Generate.chat_hist = s3_chat_summ_resp["Body"].read().decode("utf-8")

                except ClientError as err:
                    raise ValueError(
                        f"Error in retrieving/reading contents of {chat_summ_file} file from S3..."
                ) from err


    def generate_answer(self) -> Generator[str, None, None]:
        answer = ""
        retrieved_docs = self.query_engine.retrieve(QueryBundle(query_str=self.refined_query))
        logger.info(f"Number of retrieved_docs: {len(retrieved_docs)}")
        
        if not retrieved_docs:
            logger.info(f"No contexts have been retrieved")
            generic_resp = self.openai_llm.complete(self.config['PROMPTS']['GENERIC_PROMPT'].format(self.category_name, self.query)).text.strip()
            yield json.dumps({"response_id": str(uuid.uuid4()), "type": "error", "text": generic_resp})
            # yield {"response_id": str(uuid.uuid4()), "type": "error", "text": generic_resp}
            return

        retrieved_docs_score = [doc.score for doc in retrieved_docs]
        
        logger.info(F"Number of retrieved docs: {len(retrieved_docs_score)}.\nScores List: {retrieved_docs_score}")
        logger.info(f"Maximum Retrieved Score: {max(retrieved_docs_score)}")

        if max(retrieved_docs_score) <= 0.70:
            generic_resp = self.openai_llm.complete(self.config['PROMPTS']['GENERIC_PROMPT'].format(self.category_name, self.query)).text.strip()
            yield json.dumps({"response_id": str(uuid.uuid4()), "type": "error", "text": generic_resp})
            # yield {"response_id": str(uuid.uuid4()), "type": "error", "text": generic_resp}
            return
        
        start_resp = time.perf_counter()
        response = self.query_engine.query(self.refined_query)
        # response = self.chat_engine.stream_chat(prompt)

        for text in response.response_gen:
            if text != "Empty Response":
                answer += text
                yield json.dumps({"response_id": str(uuid.uuid4()), "type": "tokens", "text": text})
                # yield {"response_id": str(uuid.uuid4()), "type": "tokens", "text": text}
                # print(text, end="", flush=True)
        
        end_resp = time.perf_counter()
        logger.info(f"Successfully streamed output tokens! {end_resp-start_resp} secs.")

        start_md = time.perf_counter()

        # md_answer = self.openai_llm.complete(self.config['PROMPTS']['MARKDOWN_PROMPT'].format(answer)).text.strip()

        # end_md = time.perf_counter()

        # logger.info(f"Markdown-formatted answer has been generated! {end_md-start_md} secs.")
        
        extract_pattern = r'^Source \d+:\s*\n'
        cited_nums = re.findall(r'\[(\d+)\]', answer)
        source_lst = []

        logger.info(f"Cited Numbers: {cited_nums}")

        for idx, source in enumerate(response.source_nodes):
            source_text = re.sub(extract_pattern, '', source.node.get_text(), flags=re.MULTILINE).strip()
            source_lst.append(source_text)
        
        contexts = {}
        retrieved_counter = 0

        logger.info(f"Source List: {source_lst}")

        for idx, doc in enumerate(retrieved_docs):
            if str(idx+1) not in cited_nums:
                continue

            if doc.text.strip() in source_lst:
                retrieved_counter += 1
                chunk = doc.metadata['highlighted_chunk']
                file_name = doc.metadata['file_name']
                page_num = doc.metadata['page_num']

                contexts[str(retrieved_counter)] = {'file_name': file_name, 'page_num': page_num, 'chunk': chunk}
                answer = answer.replace(f'[{str(idx+1)}]', f'[[{retrieved_counter}]]({self.config["aws_s3_pdf_base_url"]}{quote(file_name)}.pdf)')
        
        yield json.dumps({"response_id": str(uuid.uuid4()), "type": "answer", "text": answer})
        # yield {"response_id": str(uuid.uuid4()), "type": "answer", "text": answer}

        end_md = time.perf_counter()
        logger.info(f"Markdown-formatted answer has been generated! {end_md-start_md} secs.")

        contexts_dict = json.dumps(contexts)
        logger.info(f"Context Dictionary: {contexts_dict}")
        yield json.dumps({"response_id": str(uuid.uuid4()), "type": "context", "text": contexts_dict})
        # yield {"response_id": str(uuid.uuid4()), "type": "context", "text": contexts_dict}
        # print(f"\n\n{'#'*50}\nContexts: {contexts_dict}")
        
        start_related = time.perf_counter()
        related_queries = self.openai_llm.complete(self.config['PROMPTS']['RELATED_QUERIES_PROMPT'].format(self.query, '\n\n'.join(source_lst), answer)).text.strip()
        logger.info(f"Related Queries: {related_queries}")
        yield json.dumps({"response_id": str(uuid.uuid4()), "type": "related", "text": related_queries})
        # yield {"response_id": str(uuid.uuid4()), "type": "related", "text": related_queries}
        end_related = time.perf_counter()

        logger.info(f"Time taken to generate related queries: {end_related-start_related} secs.")
        
        if Generate.chat_hist is None:
            start_title = time.perf_counter()

            conversation_title = self.openai_llm.complete(self.config['PROMPTS']['CONVERSATION_TITLE_PROMPT'].format(self.query, self.category_name)).text.strip()
            logger.info(f"Conversation Title: {conversation_title}")
            yield json.dumps({"response_id": str(uuid.uuid4()), "type": "title", "text": conversation_title})
            # yield {"response_id": str(uuid.uuid4()), "type": "title", "text": conversation_title}
            
            end_title = time.perf_counter()
            logger.info(f"Time taken to generate conversation title: {end_title-start_title} secs.")
        
        Generate.chat_hist = self.refined_query + '\n' + answer + '\n\n'

        # try:
        #     history = self.refined_query + '\n' + answer + '\n\n'

        #     start_hist = time.perf_counter()
        #     history_summarized = self.openai_llm.complete(self.config['PROMPTS']['HISTORY_SUMMARIZER_PROMPT']
        #                                         .format(history)).text.strip()
        #     end_hist = time.perf_counter()
        #     logger.info(f"Time taken to generate Chat History: {end_hist-start_hist} secs.")

        #     Generate.chat_hist = history_summarized

        #     start_put_obj = time.perf_counter()
        #     self.s3.put_object(
        #         Bucket=self.config["aws_s3_bucket"],
        #         Key=self.chat_summ_file,
        #         Body=history_summarized,
        #     )

        #     end_put_obj = time.perf_counter()
        #     logger.info(f"S3 Put Object Time Taken: {end_put_obj-start_put_obj} secs.")

        # except ClientError as err:
        #     raise ValueError(
        #         f"Error in modification of {self.chat_summ_file} file in S3..."
        #     ) from err
        
        
        # with open(self.chat_logs, mode, encoding='utf-8') as file:
        #     file.write(query + '\n' + answer + '\n\n')

        # self.chat_store.persist(persist_path=Path(f"chat_logs/chat_{self.chat_id}.json"))
    

    # def chat_history(self, chat_hist: str, query: str) -> Tuple[str, str]:
    #     s3_chat_logs_obj = self.s3.list_objects_v2(
    #         Bucket=self.config["aws_s3_bucket"], Delimiter="/", Prefix=f"{chat_hist}/"
    #     )
    #     if "Contents" not in s3_chat_logs_obj:
    #         self.s3.put_object(Bucket=self.config["aws_s3_bucket"], Key=f"{chat_hist}/")

    #     elif "Contents" in s3_chat_logs_obj and self.chat_logs in [
    #             content["Key"] for content in s3_chat_logs_obj["Contents"]
    #         ]:
    #         try:
    #             s3_chat_logs_resp = self.s3.get_object(
    #                 Bucket=self.config["aws_s3_bucket"], Key=self.chat_logs
    #             )
    #             history = s3_chat_logs_resp["Body"].read().decode("utf-8")

    #             tokens = self.encoding.encode(history)[-4096:]
    #             history_summarized = self.openai_llm.complete(self.config['PROMPTS']['HISTORY_SUMMARIZER_PROMPT']
    #                                             .format(self.encoding.decode(tokens))).text.strip()
                
    #             refined_query = "<|HISTORY|>: " + history_summarized + "\n<|QUERY|>: " + query
    #             return refined_query, "a"

    #         except ClientError as err:
    #             raise ValueError(
    #                 f"Error in retrieving/reading contents of {self.chat_logs} file from S3..."
    #             ) from err
            
        # if os.path.exists(self.chat_logs):
        #     with open(self.chat_logs, 'r', encoding='utf-8') as file:
        #         history = file.read()

        #     tokens = self.encoding.encode(history)[-2048:]
        #     history_summarized = self.tg_llm.complete(self.config['PROMPTS']['HISTORY_SUMMARIZER_PROMPT']
        #                                     .format(self.encoding.decode(tokens))).text.strip()
            
        #     refined_query = "<|HISTORY|>: " + history_summarized + "\n<|QUERY|>: " + query
        #     return refined_query, "a"
        
        # refined_query = "<|QUERY|>: " + query
        # return refined_query, "w"
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--query", help="User Query", required=True)
    parser.add_argument("--category", help="Category", required=True)
    parser.add_argument("--chat_id", default="new_chat", help="Chat ID")
    parser.add_argument("--persist_dir", default="persist", help="Persistent Storage")
    parser.add_argument("--collection_name", default="rag_llm", help="Collection Name")
    parser.add_argument("--chat_summ_bucket", default="chat_hist", help="Chat History")
    
    args = parser.parse_args()

    assert args.category in ["finance", "oil_gas"]
    category_mapping = {"oil_gas": "Oil and Gas", "finance": "Finance"}

    with open(Path('self.config/self.config.yaml').resolve()) as f:
        config = yaml.safe_load(f)
    
    if args.chat_id == "new_chat":
        args.chat_id = "".join(random.choices(string.ascii_lowercase + string.digits, k=8))

    generate_obj = Generate(config, args.chat_id, args.query, category_mapping[args.category], args.persist_dir, args.collection_name, args.chat_summ_bucket)

    generate_obj.generate_answer()