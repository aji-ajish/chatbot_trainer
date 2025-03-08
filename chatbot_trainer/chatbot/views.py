import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader, UnstructuredExcelLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import tiktoken  
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from .models import UploadedFile
from .serializers import UploadedFileSerializer
import logging

# Helper function to get the appropriate loader based on file type
def get_loader(file_path, file_type):
    if file_type == 'pdf':
        return PyPDFLoader(file_path)
    elif file_type == 'txt':
        return TextLoader(file_path)
    elif file_type == 'docx':
        return UnstructuredWordDocumentLoader(file_path)  # Updated to use UnstructuredWordDocumentLoader
    elif file_type == 'xlsx':
        return UnstructuredExcelLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

# Helper function to process the file and create embeddings
def process_file(file_path, file_type, course_id,api_key):
    # Create OpenAI Embeddings and FAISS index
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    
    # Load the file using the appropriate loader
    loader = get_loader(file_path, file_type)
    documents = loader.load()
    
    # Split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    
    # Create a FAISS index from the texts
    faiss_db = FAISS.from_documents(texts, embeddings)
    
    # Save the FAISS index locally
    faiss_index_path = f"faiss_index/{course_id}_index"
    faiss_db.save_local(faiss_index_path)
    print(f"✅ FAISS index saved at {faiss_index_path}")
    
    # Calculate tokens
    enc = tiktoken.encoding_for_model("gpt-4o-mini")
    request_tokens = len(enc.encode(str(documents)))
    response_tokens = len(enc.encode(str(faiss_db)))  # Just an example; you can improve token calculation as needed

    return request_tokens, response_tokens

@api_view(['POST'])
def upload_file(request):
    # Log incoming request data
    print(f"Request data: {request.data}")
    print(f"Request files: {request.FILES}")


    # Get API Key from request header
    api_key = request.headers.get('Authorization')  
    if not api_key:
        return Response({'detail': 'API key is missing'}, status=status.HTTP_400_BAD_REQUEST)
    
    # Strip 'Bearer ' from the key
    api_key = api_key.replace('Bearer ', '').strip()
    if not api_key:
        return Response({'detail': 'API key is missing'}, status=status.HTTP_400_BAD_REQUEST)

    # Log API key to verify (for debugging purposes)
    print(f"API Key: {api_key}")

    # Get course ID from request data
    course_id = request.data.get('course_id')
    if not course_id:
        return Response({'detail': 'Course ID is required'}, status=status.HTTP_400_BAD_REQUEST)

    # Get file from request
    file = request.FILES.get('file')
    if not file:
        return Response({'detail': 'File is required'}, status=status.HTTP_400_BAD_REQUEST)

    # Save the file locally
    upload_dir = 'uploads'
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)
    
    file_path = os.path.join(upload_dir, file.name)
    with open(file_path, 'wb') as f:
        for chunk in file.chunks():
            f.write(chunk)

    # Get file type (use mimetypes or file extension to determine type)
    file_type = file.name.split('.')[-1].lower()
    if file_type not in ['pdf', 'txt', 'docx', 'xlsx']:
        return Response({'detail': f'Unsupported file type: {file_type}'}, status=status.HTTP_400_BAD_REQUEST)

    # Process the file and create embeddings
    try:
        request_tokens, response_tokens = process_file(file_path, file_type, course_id,api_key)
    except Exception as e:
        return Response({'detail': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

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
    # Fetch all uploaded files from the database
    uploaded_files = UploadedFile.objects.all()

    file_data = []
    for uploaded_file in uploaded_files:

        file_data.append({
            'id': uploaded_file.id,
            'filename': uploaded_file.filename,
            'filetype': uploaded_file.filetype,
            'request_token': uploaded_file.request_token,
            'response_token': uploaded_file.response_token,
            'course_name': uploaded_file.course_id,  # Add course_name to the response
            'upload_date': uploaded_file.upload_date,
        })

    return Response(file_data)


# Set up logging
logger = logging.getLogger(__name__)

@api_view(['DELETE'])
def delete_file(request, file_id):
    try:
        # Fetch the uploaded file record from the database
        uploaded_file = UploadedFile.objects.get(id=file_id)
        
        # Get the file paths
        file_path = uploaded_file.filepath  # Path of the uploaded file
        faiss_index_path = f"faiss_index/{uploaded_file.course_id}_index"  # Path of the FAISS index file

        # Log file paths for debugging
        logger.info(f"Attempting to delete uploaded file: {file_path}")
        logger.info(f"Attempting to delete FAISS index file: {faiss_index_path}")

        # Delete the uploaded file from the filesystem if it exists
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Deleted uploaded file: {file_path}")
        else:
            logger.warning(f"Uploaded file does not exist: {file_path}")
        
        # Delete the FAISS index directory or file from the filesystem
        if os.path.exists(faiss_index_path):
            try:
                # Check if FAISS index is a directory and delete recursively
                if os.path.isdir(faiss_index_path):
                    shutil.rmtree(faiss_index_path)  # Recursively remove directory
                    logger.info(f"Deleted FAISS index directory: {faiss_index_path}")
                else:
                    os.remove(faiss_index_path)  # Remove as a file
                    logger.info(f"Deleted FAISS index file: {faiss_index_path}")
            except Exception as e:
                logger.error(f"Error while deleting FAISS index directory/file: {e}")
                return Response({'detail': f"Error while deleting FAISS index: {e}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        else:
            logger.warning(f"FAISS index does not exist: {faiss_index_path}")

        # Delete the record from the database
        uploaded_file.delete()
        logger.info(f"Deleted record from database for file ID: {file_id}")

        return Response({'detail': 'File and related data deleted successfully'}, status=status.HTTP_204_NO_CONTENT)

    except UploadedFile.DoesNotExist:
        logger.error(f"File with ID {file_id} not found.")
        return Response({'detail': 'File not found'}, status=status.HTTP_404_NOT_FOUND)

    except Exception as e:
        logger.error(f"Unexpected error while deleting file {file_id}: {str(e)}")
        return Response({'detail': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)