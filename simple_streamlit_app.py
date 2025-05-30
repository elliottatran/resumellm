import streamlit as st
import boto3
from botocore.exceptions import ClientError
import json
import time
import os

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
    
uploaded_file = st.file_uploader("Choose a PDF file")


if uploaded_file is not None:
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
                    documentText = documentText = documentText + ' ' + item["Text"]
                    
        
        # Define the prompt for the model.
        sys_prompt = f'''You are a server API that receives text and returns a JSON object with the content of the resume supplied. 
        
        Extract the candidate's name, job titles, company names, schools and skills. Provide a confidence score between 0 to 100 on how confident you are in each extraction. If you cannot find the field, leave it blank. An extraction schema for resume parsing can be define as follows:
        name: object
            - candidate_name: string
            - email: string
            - location: string
            - score: integer
        job_experience: object
            - job_title: string
            - company_name: string
            - start_date: Date
            - end_date: Date
            - score: integer
        education: object
            - school_name: string
            - date: Date
            - score: integer
        skills: object
            - skill: Array
            - score: integer
            
        Do not infer any data based on previous training, strictly use only the user text given as input.
        ''' 
        
        # Embed the prompt in Llama 3's instruction format.
        formatted_prompt = f"""
        <|begin_of_text|><|start_header_id|>sytem<|end_header_id|>
        {sys_prompt}
        <|eot_id|><|start_header_id|>user<|end_header_id|>
        {documentText}
       <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """
        
        # Format the request payload using the model's native structure.
        native_request = {
            "prompt": formatted_prompt,
            "temperature": 0.2,
        }
        
        # Convert the native request to JSON.
        request = json.dumps(native_request)
        
        try:
            # Invoke the model with the request.
            response = client.invoke_model(modelId=model_id, body=request)
        
        except (ClientError, Exception) as e:
            print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
            exit(1)
        
        # Decode the response body.
        model_response = json.loads(response["body"].read())
        
        # Extract and print the response text.
        response_text = model_response["generation"]
        response_text
        
    s3_client.delete_object(
        Bucket='resumeappllm',
        Key='st_upload.pdf',
    )