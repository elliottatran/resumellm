import streamlit as st
import boto3
from botocore.exceptions import ClientError
import json

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
    
model_id = 'us.meta.llama3-1-8b-instruct-v1:0'
    
uploaded_file = st.file_uploader("Choose a PDF file")


if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    ddt = textract_client.detect_document_text(Document={'Bytes':bytes_data})
    documentText = ""
    
    for item in ddt["Blocks"]:
        if item["BlockType"] == "LINE":
            documentText = documentText + ' ' + item["Text"]
    
            # print('\033[94m' +  item["Text"] + '\033[0m')
            # # print(item["Text"])
    
        

    
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