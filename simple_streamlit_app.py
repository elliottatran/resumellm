import streamlit as st
import boto3
from botocore.exceptions import ClientError
import json
import time
import os
from pathlib import Path
from docx import Document

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma

with open('openai.txt') as f:
    lines = f.readlines()
    OPENAI_API_KEY = lines[0].strip()

with open('aws_account.txt') as f:
    lines = f.readlines()
    key_id = lines[0].strip()
    access_key = lines[1].strip()

textract_client = boto3.client(
    service_name = 'textract',
    region_name='us-east-2',
    aws_access_key_id=key_id,
    aws_secret_access_key=access_key,
    verify=False)


client = boto3.client(
    service_name="bedrock-runtime", 
    region_name='us-east-2', 
    aws_access_key_id=key_id,
    aws_secret_access_key=access_key,
    verify=False)
    
s3_client = boto3.client('s3',
    region_name='us-east-2',
    aws_access_key_id=key_id,
    aws_secret_access_key=access_key,
    verify=False)
    
model_id = 'us.meta.llama3-1-8b-instruct-v1:0'

def upload_file(file_name, bucket, object_name=None):
    """Upload a file to an S3 bucket

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """

    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = os.path.basename(file_name)

    # Upload the file
    try:
        response = s3_client.upload_fileobj(file_name, bucket, object_name)
    except ClientError as e:
        print('error')
        return False
    return True
    
def start_job(client, s3_bucket_name, object_name):
    response = None
    response = client.start_document_text_detection(
        DocumentLocation={
            'S3Object': {
                'Bucket': s3_bucket_name,
                'Name': object_name
            }})

    return response["JobId"]


def is_job_complete(client, job_id):
    time.sleep(1)
    response = client.get_document_text_detection(JobId=job_id)
    status = response["JobStatus"]
    print("Job status: {}".format(status))

    while(status == "IN_PROGRESS"):
        time.sleep(1)
        response = client.get_document_text_detection(JobId=job_id)
        status = response["JobStatus"]
        print("Job status: {}".format(status))

    return status


def get_job_results(client, job_id):
    pages = []
    time.sleep(1)
    response = client.get_document_text_detection(JobId=job_id)
    pages.append(response)
    print("Resultset page received: {}".format(len(pages)))
    next_token = None
    if 'NextToken' in response:
        next_token = response['NextToken']

    while next_token:
        time.sleep(1)
        response = client.\
            get_document_text_detection(JobId=job_id, NextToken=next_token)
        pages.append(response)
        print("Resultset page received: {}".format(len(pages)))
        next_token = None
        if 'NextToken' in response:
            next_token = response['NextToken']

    return pages
    
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large", api_key=OPENAI_API_KEY)

# initiate the model
llm = ChatOpenAI(temperature=0.5, model='gpt-4o-mini', api_key=OPENAI_API_KEY)

CHROMA_PATH = r"chroma_db"

# connect to the chromadb
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH, 
)

# Set up the vectorstore to be the retriever
num_results = 1
retriever = vector_store.as_retriever(search_kwargs={'k': num_results})
    
uploaded_file = st.file_uploader("Choose a PDF, DOC, DOCX, PNG, JPEG OR TIFF file")


if uploaded_file is not None:
    file_type = Path(uploaded_file.name ).suffix
    if file_type == ".pdf":
    # To read file as bytes:
        s3_upload = upload_file(uploaded_file, 'resumeappllm', "st_upload.pdf")
        job_id = start_job(textract_client, 'resumeappllm', 'st_upload.pdf')
        job_status = is_job_complete(textract_client, job_id)
        if job_status == "SUCCEEDED":
            response = get_job_results(textract_client, job_id)
            documentText = ""
            for result_page in response:
                for item in result_page["Blocks"]:
                    if item["BlockType"] == "LINE":
                        documentText = documentText + ' ' + item["Text"]
        
        s3_client.delete_object(
            Bucket='resumeappllm',
            Key='st_upload.pdf')
    elif file_type == ".doc" or  file_type == ".docx":
        document = Document(uploaded_file)
        documentText = ""
        for i in document.paragraphs:
            documentText = documentText + ' ' + i.text 
    elif file_type == ".png" or file_type == ".jpeg" or file_type == ".jpg"or file_type == ".tiff":
        bytes_data = uploaded_file.getvalue()
        ddt = textract_client.detect_document_text(Document={'Bytes':bytes_data})
        documentText = ""
        for item in ddt["Blocks"]:
            if item["BlockType"] == "LINE":
                documentText = documentText + ' ' + item["Text"]
                    
    try:            
        docs = retriever.invoke(documentText)

        # add all the chunks to 'knowledge'
        knowledge = ""

        for doc in docs:
            knowledge += doc.page_content+"\n\n"


        # make the call to the LLM (including prompt)
        if documentText is not None:


            rag_prompt = f"""
            You are a human resource professional who is hiring a data scientist. 
            Review the resume in the "user input" section and give a summary.
            Then provide a rating from 0 to 100 on how likely they are to get the job in comparison to other candidates in "The knowledge" section.
            Do not infer any data based on previous training, strictly use information from "The knowledge" section and the user input.

            User input: {documentText}

            The knowledge: {knowledge}

            """

            #print(rag_prompt)
            response = llm.invoke(rag_prompt)
            response.content
    except:
        st.write("Please try again with a file type that is supported by this application.")
        
