import os
import google.generativeai as genai
from flask import Blueprint, request, jsonify
from flask_cors import cross_origin
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.schema import Document
import fitz  # PyMuPDF

# Create blueprint
chatbot_bp = Blueprint('chatbot', __name__)

# Configure API key
os.environ["GEMINI_API_KEY"] = "AIzaSyAU5xz5YO4_d-GXTFagwhn2TS1WNgtB950"
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# Initialize embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=os.getenv("GEMINI_API_KEY"))

# Create prompt template
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="You are an expert in solar system, the plants, the moon and all that. Your name is StarChaser. Your goal is to help people with their questions about the solar system and all what relates to it. use this :\n\n{context}\n\nQuestion: {question}\n\nAnswer 👉"
)

# Global variable for vectorstore (will be initialized when PDF is uploaded)
vectorstore = None

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF file"""
    doc = fitz.open(pdf_path)
    text = "\n".join([page.get_text("text") for page in doc])
    return text

def initialize_vectorstore_from_text(text):
    """Initialize vectorstore from text content"""
    global vectorstore
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text(text)
    
    # Convert chunks to documents
    documents = [Document(page_content=chunk) for chunk in chunks]
    
    # Create vectorstore
    vectorstore = Chroma.from_documents(documents, embedding_model, persist_directory="./chroma_db")
    
    return True

def rag_chatbot(question):
    """Main chatbot function using RAG"""
    global vectorstore
    
    if vectorstore is None:
        return "Please upload a PDF document first to initialize the knowledge base."
    
    # Search for relevant documents
    retrieved_docs = vectorstore.similarity_search(question, k=3)
    context = "\n".join([doc.page_content for doc in retrieved_docs])
    
    # Create conversation context
    conversation_context = f"As a StarChaser, here's the context from where you will gather your information :\n{context}\n\nQuestion : {question}\n\nIf you found any answer that doesn't use any information from what I provided you, you have to answer professionally, accurately and short \n\nAnswer 👉 :"
    
    # Generate response
    response = prompt | llm
    result = response.invoke({"context": conversation_context, "question": question})
    
    return result.content

@chatbot_bp.route('/chat', methods=['POST'])
@cross_origin()
def chat():
    """Handle chat requests"""
    try:
        data = request.get_json()
        if not data or 'question' not in data:
            return jsonify({'error': 'Question is required'}), 400
        
        question = data['question']
        response = rag_chatbot(question)
        
        return jsonify({
            'question': question,
            'answer': response,
            'status': 'success'
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@chatbot_bp.route('/upload-pdf', methods=['POST'])
@cross_origin()
def upload_pdf():
    """Handle PDF upload and initialize knowledge base"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not file.filename.lower().endswith('.pdf'):
            return jsonify({'error': 'Only PDF files are allowed'}), 400
        
        # Save uploaded file
        upload_path = os.path.join(os.path.dirname(__file__), '..', 'uploads')
        os.makedirs(upload_path, exist_ok=True)
        file_path = os.path.join(upload_path, file.filename)
        file.save(file_path)
        
        # Extract text and initialize vectorstore
        pdf_text = extract_text_from_pdf(file_path)
        initialize_vectorstore_from_text(pdf_text)
        
        return jsonify({
            'message': 'PDF uploaded and processed successfully',
            'filename': file.filename,
            'status': 'success'
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@chatbot_bp.route('/initialize-default', methods=['POST'])
@cross_origin()
def initialize_default():
    """Initialize with default solar system knowledge"""
    try:
        # Default solar system knowledge
        default_text = """
        The Solar System is the gravitationally bound system of the Sun and the objects that orbit it. 
        It consists of the Sun, eight planets, their moons, and various small bodies including asteroids, comets, and meteoroids.

        The eight planets are:
        1. Mercury - The smallest planet and closest to the Sun
        2. Venus - The hottest planet with a thick atmosphere
        3. Earth - The only known planet with life
        4. Mars - The red planet with polar ice caps
        5. Jupiter - The largest planet, a gas giant
        6. Saturn - Known for its prominent ring system
        7. Uranus - An ice giant tilted on its side
        8. Neptune - The windiest planet in the solar system

        Pluto was reclassified as a dwarf planet in 2006 by the International Astronomical Union.

        Black holes are regions of spacetime where gravity is so strong that nothing, not even light, can escape.
        They form when massive stars collapse at the end of their lives.

        The Moon is Earth's only natural satellite, formed about 4.5 billion years ago.
        It influences Earth's tides and stabilizes our planet's axial tilt.
        """
        
        initialize_vectorstore_from_text(default_text)
        
        return jsonify({
            'message': 'Default solar system knowledge base initialized successfully',
            'status': 'success'
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

