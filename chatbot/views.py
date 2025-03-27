import os
import shutil
import time
import logging
import tiktoken
import faiss
from django.db.models import Sum, Value
from django.db.models.functions import Coalesce
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader, UnstructuredExcelLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from .models import UploadedFile, ChatHistory
from .serializers import UploadedFileSerializer, ChatHistorySerializer
from langchain.schema import AIMessage
import unicodedata
from django.db.utils import IntegrityError


logger = logging.getLogger(__name__)



# Helper function to get the appropriate loader based on file type
def get_loader(file_path, file_type):
    loaders = {
        'pdf': PyPDFLoader,
        'txt': TextLoader,
        'docx': UnstructuredWordDocumentLoader,
        'xlsx': UnstructuredExcelLoader
    }
    
    if file_type in loaders:
        return loaders[file_type](file_path)
    return None

# Helper function to process and append data to FAISS index
def process_file(file_path, file_type, faiss_index_path, api_key):
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    loader = get_loader(file_path, file_type)

    if not loader:
        raise ValueError(f"Unsupported file type: {file_type}")

    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    # Ensure FAISS index path exists
    os.makedirs(faiss_index_path, exist_ok=True)
    index_file = os.path.join(faiss_index_path, "index.faiss")

    # Check if FAISS index exists, load and append or create new
    if os.path.exists(index_file):
        try:
            faiss_db = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)
            faiss_db.add_documents(texts)
        except Exception as e:
            raise RuntimeError(f"Error loading FAISS index: {str(e)}")
    else:
        faiss_db = FAISS.from_documents(texts, embeddings)
        print(f"🆕 New FAISS Index Created: Contains {faiss_db.index.ntotal} vectors")

    # Save the updated FAISS index
    faiss_db.save_local(faiss_index_path)
    print(f"📌 FAISS Index Updated: Contains {count_faiss_vectors(faiss_index_path, api_key)} vectors")

    # Token calculation
    enc = tiktoken.encoding_for_model("gpt-4o-mini")
    request_tokens = sum(len(enc.encode(doc.page_content)) for doc in texts)
    response_tokens = len(enc.encode(str(faiss_db)))

    return request_tokens, response_tokens



@api_view(['POST'])
def upload_file(request):
    api_key = request.headers.get('Authorization', '').replace('Bearer ', '').strip()
    if not api_key:
        return Response({'detail': 'API key is missing'}, status=status.HTTP_400_BAD_REQUEST)

    course_id = request.data.get('course_id')
    type = request.data.get('type')
    if not course_id:
        return Response({'detail': 'Course ID is required'}, status=status.HTTP_400_BAD_REQUEST)

    file = request.FILES.get('file')
    if not file:
        return Response({'detail': 'File is required'}, status=status.HTTP_400_BAD_REQUEST)

    file_type = file.name.split('.')[-1].lower()
    if file_type not in ['pdf', 'txt', 'docx', 'xlsx']:
        return Response({'detail': f'Unsupported file type: {file_type}'}, status=status.HTTP_400_BAD_REQUEST)

    # Define paths
    upload_dir = 'uploads'
    os.makedirs(upload_dir, exist_ok=True)

    timestamp = int(time.time())
    filename = f"{course_id}_{timestamp}_{file.name}"
    file_path = os.path.join(upload_dir, filename)

    # Save the file
    with open(file_path, 'wb') as f:
        for chunk in file.chunks():
            f.write(chunk)

    faiss_index_path = f"faiss_index/{course_id}"

    try:
        request_tokens, response_tokens = process_file(file_path, file_type, faiss_index_path, api_key)
    except Exception as e:
        return Response({'detail': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    uploaded_file = UploadedFile.objects.create(
        course_id=course_id,
        filename=filename,
        filetype=file.content_type,
        filepath=file_path,
        type=type,
        request_token=request_tokens,
        response_token=response_tokens
    )

    return Response({'detail': 'File uploaded successfully', 'file': UploadedFileSerializer(uploaded_file).data}, status=status.HTTP_201_CREATED)

@api_view(['GET'])
def list_files(request):
    uploaded_files = UploadedFile.objects.all()
    return Response(UploadedFileSerializer(uploaded_files, many=True).data)

@api_view(['DELETE'])
def delete_file(request, file_id):
    api_key = request.headers.get('Authorization', '').replace('Bearer ', '').strip()
    if not api_key:
        return Response({'detail': 'API key is missing'}, status=status.HTTP_400_BAD_REQUEST)

    try:
        file_record = UploadedFile.objects.get(id=file_id)
    except UploadedFile.DoesNotExist:
        return Response({'detail': 'File not found in database'}, status=status.HTTP_404_NOT_FOUND)

    course_id = file_record.course_id
    file_path = file_record.filepath

    # Attempt to delete the file
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
        else:
            print(f"⚠️ File not found: {file_path}, skipping deletion")
    except Exception as e:
        print(f"❌ Error deleting file: {str(e)}")
        return Response({'detail': f'Error deleting file: {str(e)}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    # Delete database record
    try:
        file_record.delete()
    except Exception as e:
        print(f"❌ Error deleting file record: {str(e)}")
        return Response({'detail': f'Error deleting file record: {str(e)}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    # Rebuild FAISS index
    try:
        rebuild_faiss_index(course_id, api_key)
        print(f"📌 FAISS Index Rebuilt: Now Contains {count_faiss_vectors(f'faiss_index/{course_id}', api_key)} vectors")
    except Exception as e:
        print(f"❌ Error rebuilding FAISS index: {str(e)}")
        return Response({'detail': f'Error rebuilding FAISS index: {str(e)}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    return Response({'detail': 'File deleted successfully'}, status=status.HTTP_200_OK)


def rebuild_faiss_index(course_id, api_key):
    """
    Rebuilds FAISS index for a course after a file is deleted.
    """
    faiss_index_path = f"faiss_index/{course_id}"
    index_file = os.path.join(faiss_index_path, "index.faiss")

    # Ensure FAISS folder exists
    if not os.path.exists(faiss_index_path):
        print("No FAISS index found for this course.")
        return

    # Get remaining files for this course
    remaining_files = UploadedFile.objects.filter(course_id=course_id)
    
    if not remaining_files.exists():
        print("No remaining files. Deleting FAISS index.")
        shutil.rmtree(faiss_index_path)  # Delete the entire folder
        return

    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    all_texts = []

    for file_record in remaining_files:
        file_path = file_record.filepath
        file_type = file_record.filename.split('.')[-1].lower()
        
        loader = get_loader(file_path, file_type)
        if not loader:
            print(f"Skipping unsupported file: {file_record.filename}")
            continue

        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = text_splitter.split_documents(documents)
        all_texts.extend(texts)

    if all_texts:
        faiss_db = FAISS.from_documents(all_texts, embeddings)
        faiss_db.save_local(faiss_index_path)
        print(f"✅ FAISS Index Rebuilt: Contains {faiss_db.index.ntotal} vectors")
    else:
        shutil.rmtree(faiss_index_path)
        print("❌ No valid documents found. FAISS index deleted.")


def count_faiss_vectors(faiss_index_path, api_key):
    """ Returns the number of vectors in the FAISS index. """
    if os.path.exists(f"{faiss_index_path}/index.faiss"):
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        faiss_db = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)
        return faiss_db.index.ntotal
    return 0



def classify_query(query, course_name, api_key):
    """Classifies the query while ensuring relevance to the course."""
    try:
        llm = ChatOpenAI(openai_api_key=api_key)
        response = llm.invoke(
            f"Classify the user query into one of the following categories for the course '{course_name}':\n\n"
            f"Query: '{query}'\n\n"
            f"Categories:\n"
            f"- 'Concept Explanation' (If asking about a topic conceptually)\n"
            f"- 'Question Paper Request' (If requesting a new question paper)\n"
            f"- 'Question Paper Answer Request' (If requesting answers for an existing question paper OR seeking solutions to specific questions)\n"
            f"- 'General Query' (For other course-related questions)\n"
            f"- 'Irrelevant Query' (If NOT related to the course '{course_name}')\n\n"
            f"Respond **only** with the category name."
        )

        category = response.content.strip() if isinstance(response, AIMessage) else str(response).strip()

        print(f"Query Type: {category}")  # Debugging

        return category

    except Exception as e:
        print(f"OpenAI Classification Error: {str(e)}")
        return "General Query"


def clean_text(text):
    """Normalize Unicode text to avoid database encoding issues."""
    return unicodedata.normalize('NFKC', text)

@api_view(['POST'])
def process_query(request):
    api_key = request.headers.get('Authorization', '').replace('Bearer ', '').strip()
    if not api_key:
        return Response({'detail': 'API key is missing'}, status=status.HTTP_400_BAD_REQUEST)

    user_id = request.data.get('user_id')
    course_id = request.data.get('course_id')
    course_name = request.data.get('course_name')
    user_query = request.data.get('user_query')
    response_text = request.data.get('response', "")
    is_static = request.data.get('is_static', False)

    if not user_id or not course_id or not user_query or not course_name:
        return Response({'detail': 'user_id, course_id, course_name, and user_query are required'}, status=status.HTTP_400_BAD_REQUEST)

    # Normalize Unicode characters
    user_query = clean_text(user_query)

    enc = tiktoken.encoding_for_model("gpt-4o-mini")
    request_tokens = len(enc.encode(user_query))
    print(f"🔹 Request Tokens (User Query): {request_tokens}")

    if is_static:
        try:
            chat_entry = ChatHistory.objects.create(
                user_id=user_id, course_id=course_id, user_query=user_query,
                response=response_text, req_token=request_tokens, res_token=0
            )
            return Response({'user_query': user_query, 'response': response_text, 
                            'req_token': request_tokens, 'res_token': 0, 
                            'chat_id': chat_entry.id}, status=status.HTTP_200_OK)
        except IntegrityError as e:
            print(f"Database Error: {str(e)}")
            return Response({'detail': 'Database error while saving chat history.'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    # Classify the query
    query_type = classify_query(user_query, course_name, api_key)

    if query_type == "Irrelevant Query":
        response_text = (
            f"Your query does not seem relevant to the course '{course_name}'.\n\n"
            "Please make sure your question is related to the course content. If you need help, try rephrasing your query "
            "or providing more context about what you're looking for."
        )
        
        try:
            chat_entry = ChatHistory.objects.create(
                user_id=user_id, course_id=course_id, user_query=user_query,
                response=response_text, req_token=request_tokens, res_token=0
            )
            return Response({
                'user_query': user_query,
                'response': response_text,
                'req_token': request_tokens,
                'res_token': 0,
                'chat_id': chat_entry.id
            }, status=status.HTTP_200_OK)
        
        except IntegrityError as e:
            print(f"Database Error: {str(e)}")
            return Response({'detail': 'An error occurred while saving the chat history. Please try again later.'}, 
                            status=status.HTTP_500_INTERNAL_SERVER_ERROR)


    # Check for advanced processing
    has_advanced_processing = UploadedFile.objects.filter(course_id=course_id, type=1).exists()
    faiss_index_path = f"faiss_index/{course_id}"
    
    if not os.path.exists(faiss_index_path):
        response_text = "No training data available for this course."
        try:
            chat_entry = ChatHistory.objects.create(
                user_id=user_id, course_id=course_id, user_query=user_query,
                response=response_text, req_token=request_tokens, res_token=0
            )
            return Response({'user_query': user_query, 'response': response_text, 
                            'req_token': request_tokens, 'res_token': 0, 
                            'chat_id': chat_entry.id}, status=status.HTTP_200_OK)
        except IntegrityError as e:
            print(f"Database Error: {str(e)}")
            return Response({'detail': 'Database error while saving chat history.'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    try:
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        faiss_db = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)
        retriever = faiss_db.as_retriever(search_kwargs={"k": 5})
        retrieved_docs = retriever.invoke(user_query)

        filtered_docs = [doc for doc in retrieved_docs if str(course_id) in doc.metadata.get("source", "")]

        if not filtered_docs:
            response_text = "No relevant training data found for this course."
            try:
                chat_entry = ChatHistory.objects.create(
                    user_id=user_id, course_id=course_id, user_query=user_query,
                    response=response_text, req_token=request_tokens, res_token=0
                )
                return Response({'user_query': user_query, 'response': response_text, 
                                'req_token': request_tokens, 'res_token': 0, 
                                'chat_id': chat_entry.id}, status=status.HTTP_200_OK)
            except IntegrityError as e:
                print(f"Database Error: {str(e)}")
                return Response({'detail': 'Database error while saving chat history.'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        context_text = "\n".join([doc.page_content for doc in filtered_docs])

        if has_advanced_processing:
            try:
                llm = ChatOpenAI(openai_api_key=api_key)
                optimized_context = llm.invoke(f"Optimize this training data for clarity and conciseness:\n{context_text}")
                context_text = optimized_context.content.strip() if isinstance(optimized_context, AIMessage) else str(optimized_context).strip()
            except Exception as e:
                print(f"Error optimizing training data: {str(e)}")

        try:
            llm = ChatOpenAI(openai_api_key=api_key)

            # Define the prompt based on query type
            if query_type == "Question Paper Request":
                prompt = (
                    f"Generate a structured question paper for the course '{course_name}' using this course material:\n{context_text}\n\n"
                    "The question paper should include:\n"
                    "- **5 Multiple Choice Questions** with 4 answer choices\n"
                    "- **5 Short Answer Questions**\n"
                    "- **3 Problem-Solving Exercises**\n\n"
                    "Format the question paper clearly."
                )
            elif query_type == "Question Paper Answer Request":
                prompt = (
                    f"Identify the questions from the given text and provide step-by-step solutions.\n\n"
                    f"Course: {course_name}\n"
                    f"Question Paper:\n{context_text}\n\n"
                    f"User Query: {user_query}\n\n"
                    "Format answers clearly with explanations where necessary."
                )

            elif query_type == "Concept Explanation":
                prompt = (
                    f"Explain the following concept in detail using only the given course material:\n\n"
                    f"Course: {course_name}\n"
                    f"Concept: {user_query}\n\n"
                    "Provide a clear and structured explanation, including examples where possible."
                )
            elif query_type == "General Query":
                prompt = (
                    f"Use only the following course-related information to answer the question:\n{context_text}\n\n"
                    f"Question: {user_query}\n\n"
                    "Provide a precise and informative response."
                )
            else:  # Default case (fallback)
                prompt = (
                    f"Use only the following course-related information to respond appropriately:\n{context_text}\n\n"
                    f"User Query: {user_query}"
                )

            # Debugging log
            print(f"Final Prompt:\n{prompt[:500]}...")  # Print the first 500 characters for debugging



            input_tokens = len(enc.encode(prompt))
            print(f"🔹 Input Tokens (Prompt Size): {input_tokens}")

            if input_tokens > 4096:
                return Response({'detail': "Input query is too long. Try a shorter query."}, status=status.HTTP_400_BAD_REQUEST)

            refined_response = llm.invoke(prompt)
            response_text = refined_response.content.strip() if isinstance(refined_response, AIMessage) else str(refined_response).strip()

            if not response_text:
                response_text = "I'm unable to generate a response for this query."

        except Exception as e:
            error_message = f"Error processing OpenAI request: {str(e)}"
            print(error_message)
            return Response({'detail': error_message}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # Normalize response text before storing
        response_text = clean_text(response_text)
        response_tokens = len(enc.encode(response_text))
        print(f"🔹 Response Tokens (Generated Answer): {response_tokens}")

        # **🔹 Print Total Token Usage**
        total_tokens = request_tokens + input_tokens + response_tokens
        print(f"🔥 Total Tokens Used in Query Processing: {total_tokens}")

        try:
            chat_entry = ChatHistory.objects.create(
                user_id=user_id, course_id=course_id, user_query=user_query,
                response=response_text, req_token=request_tokens, res_token=response_tokens,input_prompt_token=input_tokens,total_token=total_tokens
            )
            return Response({'user_query': user_query, 'response': response_text, 
                            'req_token': request_tokens, 'res_token': response_tokens,
                            'input_prompt_token':input_tokens,'total_token':total_tokens,
                            'chat_id': chat_entry.id}, status=status.HTTP_200_OK)
        except IntegrityError as e:
            print(f"Database Error: {str(e)}")
            return Response({'detail': 'Database error while saving chat history.'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    except Exception as e:
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
                total_res_token=Coalesce(Sum('res_token'), Value(0)),
                total_input_prompt_token=Coalesce(Sum('input_prompt_token'), Value(0)),
                total_token=Coalesce(Sum('total_token'), Value(0))
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

@api_view(['PATCH'])
def update_file_type(request, file_id):
    """API to update the 'type' field in the UploadedFile model."""
    try:
        uploaded_file = UploadedFile.objects.get(id=file_id)
    except UploadedFile.DoesNotExist:
        return Response({'detail': 'File not found'}, status=status.HTTP_404_NOT_FOUND)

    new_type = request.data.get("type")
    if new_type not in [0, 1]:
        return Response({'detail': 'Invalid type. Must be 0 or 1.'}, status=status.HTTP_400_BAD_REQUEST)

    uploaded_file.type = new_type
    uploaded_file.save()

    return Response({'detail': f'File type updated to {new_type} successfully.'}, status=status.HTTP_200_OK)
