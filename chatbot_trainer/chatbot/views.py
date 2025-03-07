import os
import json
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .models import UploadedFile
from .serializers import UploadedFileSerializer
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import tiktoken

# Helper function to process the file and create embeddings
def process_file(file_path, course_id):
    # Create OpenAI Embeddings and FAISS index
    embeddings = OpenAIEmbeddings(openai_api_key="your-openai-api-key")
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    
    faiss_db = FAISS.from_documents(texts, embeddings)
    
    # Save the FAISS index (assuming we save the index locally)
    faiss_db.save_local(f"faiss_index/{course_id}_index")
    
    # Calculate tokens
    enc = tiktoken.encoding_for_model("gpt-4o-mini")
    request_tokens = len(enc.encode(str(documents)))
    response_tokens = len(enc.encode(str(faiss_db)))  # Just an example; you can improve token calculation as needed

    return request_tokens, response_tokens

@api_view(['POST'])
def upload_file(request):
    api_key = request.headers.get('Authorization')  # Get API key from the header
    if not api_key:
        return Response({'detail': 'API key is missing'}, status=status.HTTP_400_BAD_REQUEST)

    course_id = request.data.get('course_id')
    file = request.FILES['file']
    
    # Save file locally
    file_path = f'uploads/{file.name}'
    with open(file_path, 'wb') as f:
        for chunk in file.chunks():
            f.write(chunk)

    # Process the file and create embeddings
    request_tokens, response_tokens = process_file(file_path, course_id)

    # Save file details in the database
    uploaded_file = UploadedFile.objects.create(
        course_id=course_id,
        filename=file.name,
        filetype=file.content_type,
        filepath=file_path,
        request_token=request_tokens,
        response_token=response_tokens
    )

    return Response({'detail': 'File uploaded successfully', 'file': UploadedFileSerializer(uploaded_file).data}, status=status.HTTP_201_CREATED)

@api_view(['GET'])
def list_files(request):
    api_key = request.headers.get('Authorization')  # Get API key from the header
    if not api_key:
        return Response({'detail': 'API key is missing'}, status=status.HTTP_400_BAD_REQUEST)

    files = UploadedFile.objects.all()
    return Response(UploadedFileSerializer(files, many=True).data)

@api_view(['DELETE'])
def delete_file(request, pk):
    api_key = request.headers.get('Authorization')  # Get API key from the header
    if not api_key:
        return Response({'detail': 'API key is missing'}, status=status.HTTP_400_BAD_REQUEST)

    try:
        uploaded_file = UploadedFile.objects.get(pk=pk)
    except UploadedFile.DoesNotExist:
        return Response({'detail': 'File not found'}, status=status.HTTP_404_NOT_FOUND)

    # Delete the file from the filesystem
    if os.path.exists(uploaded_file.filepath):
        os.remove(uploaded_file.filepath)

    # Delete the file record from the database
    uploaded_file.delete()

    return Response({'detail': 'File deleted successfully'}, status=status.HTTP_204_NO_CONTENT)
