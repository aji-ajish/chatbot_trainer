import os
import shutil
import time
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
from .models import ChatHistory
from .serializers import ChatHistorySerializer
from django.db.models import Sum, Value
from django.db import models
from django.db.models.functions import Coalesce

# Set up logging
logger = logging.getLogger(__name__)

# Helper function to get the appropriate loader based on file type
def get_loader(file_path, file_type):
    if file_type == 'pdf':
        return PyPDFLoader(file_path)
    elif file_type == 'txt':
        return TextLoader(file_path)
    elif file_type == 'docx':
        return UnstructuredWordDocumentLoader(file_path)
    elif file_type == 'xlsx':
        return UnstructuredExcelLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

# Helper function to process the file and create embeddings
def process_file(file_path, file_type, faiss_index_path, api_key):
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    
    # Load the file
    loader = get_loader(file_path, file_type)
    documents = loader.load()
    
    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    
    # Create FAISS index
    faiss_db = FAISS.from_documents(texts, embeddings)
    
    # Save the FAISS index in a unique folder
    faiss_db.save_local(faiss_index_path)
    print(f"✅ FAISS index saved at {faiss_index_path}")
    
    # Calculate tokens
    enc = tiktoken.encoding_for_model("gpt-4o-mini")
    request_tokens = len(enc.encode(str(documents)))
    response_tokens = len(enc.encode(str(faiss_db)))

    return request_tokens, response_tokens

@api_view(['POST'])
def upload_file(request):
    # Log incoming request data
    print(f"Request data: {request.data}")
    print(f"Request files: {request.FILES}")

    # Get API Key from request header
    api_key = request.headers.get('Authorization', '').replace('Bearer ', '').strip()
    if not api_key:
        return Response({'detail': 'API key is missing'}, status=status.HTTP_400_BAD_REQUEST)

    # Get course ID
    course_id = request.data.get('course_id')
    if not course_id:
        return Response({'detail': 'Course ID is required'}, status=status.HTTP_400_BAD_REQUEST)

    # Get file
    file = request.FILES.get('file')
    if not file:
        return Response({'detail': 'File is required'}, status=status.HTTP_400_BAD_REQUEST)

    # Generate a unique filename
    timestamp = int(time.time())  # Unique timestamp
    filename = f"{course_id}_{timestamp}_{file.name}"
    
    # Ensure upload directory exists
    upload_dir = 'uploads'
    os.makedirs(upload_dir, exist_ok=True)

    # Define file path
    file_path = os.path.join(upload_dir, filename)

    # Save file
    with open(file_path, 'wb') as f:
        for chunk in file.chunks():
            f.write(chunk)

    # Get file type
    file_type = file.name.split('.')[-1].lower()
    if file_type not in ['pdf', 'txt', 'docx', 'xlsx']:
        return Response({'detail': f'Unsupported file type: {file_type}'}, status=status.HTTP_400_BAD_REQUEST)

    # Generate a unique FAISS subfolder path
    faiss_index_path = f"faiss_index/{course_id}_{timestamp}"
    os.makedirs(faiss_index_path, exist_ok=True)

    # Process the file and create embeddings
    try:
        request_tokens, response_tokens = process_file(file_path, file_type, faiss_index_path, api_key)
    except Exception as e:
        return Response({'detail': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    # Save file details in the database
    uploaded_file = UploadedFile.objects.create(
        course_id=course_id,
        filename=filename,
        filetype=file.content_type,
        filepath=file_path,
        request_token=request_tokens,
        response_token=response_tokens
    )

    return Response({'detail': 'File uploaded successfully', 'file': UploadedFileSerializer(uploaded_file).data}, status=status.HTTP_201_CREATED)

@api_view(['GET'])
def list_files(request):
    uploaded_files = UploadedFile.objects.all()

    file_data = [
        {
            'id': uploaded_file.id,
            'filename': uploaded_file.filename,
            'filetype': uploaded_file.filetype,
            'request_token': uploaded_file.request_token,
            'response_token': uploaded_file.response_token,
            'course_name': uploaded_file.course_id,
            'upload_date': uploaded_file.upload_date,
        }
        for uploaded_file in uploaded_files
    ]

    return Response(file_data)

@api_view(['DELETE'])
def delete_file(request, file_id):
    try:
        uploaded_file = UploadedFile.objects.get(id=file_id)
        
        # Get file paths
        file_path = uploaded_file.filepath
        filename_parts = uploaded_file.filename.split("_")  # Extract timestamp
        if len(filename_parts) > 2:
            timestamp = filename_parts[1]  # Get timestamp from filename
            faiss_index_path = f"faiss_index/{uploaded_file.course_id}_{timestamp}"
        else:
            faiss_index_path = f"faiss_index/{uploaded_file.course_id}"

        # Delete uploaded file
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Deleted uploaded file: {file_path}")
        else:
            logger.warning(f"Uploaded file does not exist: {file_path}")

        # Delete FAISS index subfolder
        if os.path.exists(faiss_index_path):
            try:
                shutil.rmtree(faiss_index_path)  # Remove entire FAISS folder
                logger.info(f"Deleted FAISS index directory: {faiss_index_path}")
            except Exception as e:
                logger.error(f"Error deleting FAISS directory: {e}")
                return Response({'detail': f"Error deleting FAISS index: {e}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        else:
            logger.warning(f"FAISS index does not exist: {faiss_index_path}")

        # Delete record from database
        uploaded_file.delete()
        logger.info(f"Deleted record from database for file ID: {file_id}")

        return Response({'detail': 'File and related data deleted successfully'}, status=status.HTTP_204_NO_CONTENT)

    except UploadedFile.DoesNotExist:
        logger.error(f"File with ID {file_id} not found.")
        return Response({'detail': 'File not found'}, status=status.HTTP_404_NOT_FOUND)

    except Exception as e:
        logger.error(f"Unexpected error while deleting file {file_id}: {str(e)}")
        return Response({'detail': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# Process Query API
# Process Query API
@api_view(['POST'])
def process_query(request):
    api_key = request.headers.get('Authorization', '').replace('Bearer ', '').strip()
    if not api_key:
        return Response({'detail': 'API key is missing'}, status=status.HTTP_400_BAD_REQUEST)

    user_id = request.data.get('user_id')
    course_id = request.data.get('course_id')
    user_query = request.data.get('user_query')
    response_text = request.data.get('response')  # Static response from Moodle
    is_static = request.data.get('is_static', False)  # Check if it's a static response

    if not user_id or not course_id or not user_query:
        return Response({'detail': 'user_id, course_id, and user_query are required'}, status=status.HTTP_400_BAD_REQUEST)

    # Token count for request
    enc = tiktoken.encoding_for_model("gpt-4o-mini")
    request_tokens = len(enc.encode(user_query))

    # If it's a static response, store it directly
    if is_static:
        chat_entry = ChatHistory.objects.create(
            user_id=user_id,
            course_id=course_id,
            user_query=user_query,
            response=response_text,
            req_token=request_tokens,
            res_token=0  # No AI tokens used
        )

        return Response({
            'user_query': user_query,
            'response': response_text,
            'req_token': request_tokens,
            'res_token': 0,
            'chat_id': chat_entry.id
        }, status=status.HTTP_200_OK)

    # Process with FAISS and LLM (same as before)
    faiss_dir = "faiss_index"
    # Only retrieve indexes for the current course_id
    course_faiss_indexes = [d for d in os.listdir(faiss_dir) if d.startswith(f"{course_id}_")]
    print(f"Indexing for course {course_id}: {course_faiss_indexes}")  # Debugging

    if not course_faiss_indexes:
        default_response = "I'm sorry, but I couldn't find relevant training data for this course. Please contact your instructor for assistance."

        chat_entry = ChatHistory.objects.create(
            user_id=user_id,
            course_id=course_id,
            user_query=user_query,
            response=default_response,
            req_token=request_tokens,
            res_token=0  # No AI tokens used
        )

        return Response({
            'user_query': user_query,
            'response': default_response,
            'req_token': request_tokens,
            'res_token': 0,
            'chat_id': chat_entry.id
        }, status=status.HTTP_200_OK)

    # Process the most recent index for the course
    latest_index = sorted(course_faiss_indexes, reverse=True)[0]
    faiss_index_path = os.path.join(faiss_dir, latest_index)
    print(f"Using FAISS index: {faiss_index_path}")  # Debugging

    try:
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        faiss_db = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)

        # Retrieve top 3 matches
        retriever = faiss_db.as_retriever(search_kwargs={"k": 3})
        docs = retriever.get_relevant_documents(user_query)

        # Debugging the documents retrieved
        print(f"Documents retrieved: {docs}")  # Debugging

        # Filter the documents based on the course_id dynamically (no static "react" term)
        docs = [doc for doc in docs if doc.metadata.get("course_id") == str(course_id)]

        # If no relevant match, return "I don't know"
        if not docs:
            response_text = "I'm sorry, but I don't have enough information in the training data to answer that."

            # Store in chat history
            chat_entry = ChatHistory.objects.create(
                user_id=user_id,
                course_id=course_id,
                user_query=user_query,
                response=response_text,
                req_token=request_tokens,
                res_token=0  # No AI tokens used
            )

            return Response({
                'user_query': user_query,
                'response': response_text,
                'req_token': request_tokens,
                'res_token': 0,
                'chat_id': chat_entry.id
            }, status=status.HTTP_200_OK)

        # Force LLM to only answer based on retrieved FAISS data
        llm = ChatOpenAI(openai_api_key=api_key)
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=faiss_db.as_retriever())

        response = qa_chain.invoke({"query": user_query})  # Correct way to call the chain
        response_text = response["result"]  # Extract answer from the output dictionary

        # Token count for response
        response_tokens = len(enc.encode(response_text))

        # Save chat history (only for AI responses)
        chat_entry = ChatHistory.objects.create(
            user_id=user_id,
            course_id=course_id,
            user_query=user_query,
            response=response_text,
            req_token=request_tokens,
            res_token=response_tokens
        )

        return Response({
            'user_query': user_query,
            'response': response_text,
            'req_token': request_tokens,
            'res_token': response_tokens,
            'chat_id': chat_entry.id
        }, status=status.HTTP_200_OK)

    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return Response({'detail': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)



# Chat History API
@api_view(['GET'])
def chat_history(request):
    user_id = request.query_params.get('user_id')
    course_id = request.query_params.get('course_id')

    if not user_id:
        return Response({'detail': 'user_id is required'}, status=status.HTTP_400_BAD_REQUEST)

    if user_id == "admin":
        # Admin: Summarize all courses by summing tokens
        chat_summary = (
            ChatHistory.objects.values('course_id')
            .annotate(
                total_req_token=Coalesce(Sum('req_token'), Value(0)),
                total_res_token=Coalesce(Sum('res_token'), Value(0))
            )
        )

        # Ensure JSON format consistency
        return Response({"summary": list(chat_summary)}, status=status.HTTP_200_OK)
    elif course_id:
        # Regular user: Fetch only their own chat history for a specific course
        chat_entries = ChatHistory.objects.filter(user_id=user_id, course_id=course_id)
    else:
        return Response({'detail': 'course_id is required for non-admin users'}, status=status.HTTP_400_BAD_REQUEST)

    return Response(ChatHistorySerializer(chat_entries, many=True).data, status=status.HTTP_200_OK)

