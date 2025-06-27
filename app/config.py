import os
from dotenv import load_dotenv

# Try to load .env file with error handling
try:
    load_dotenv('.env', override=True, encoding='utf-8')
    print("✅ Loaded .env file")
except Exception as e:
    print(f"⚠️ Could not load .env file: {e}")
    print("Using environment variables directly")

class Settings:
    def __init__(self):
        # API Keys
        self.openai_api_key = os.getenv('OPENAI_API_KEY', '').strip()
        self.llamaparse_api_key = os.getenv('LLAMAPARSE_API_KEY', '').strip()
        
        # Database Configuration
        self.database_url = os.getenv('DATABASE_URL', 'mysql+pymysql://test_user:test123@localhost:3242/policy_test')
        self.db_host = os.getenv('DB_HOST', 'localhost')
        self.db_port = int(os.getenv('DB_PORT', '3242'))
        self.db_name = os.getenv('DB_NAME', 'policy_test')
        self.db_user = os.getenv('DB_USER', 'test_user')
        self.db_password = os.getenv('DB_PASSWORD', 'test123')
        
        # Qdrant Configuration
        self.qdrant_url = os.getenv('QDRANT_URL', 'https://ca579e00-fc37-4a9a-927f-118560d6b4de.eu-west-2-0.aws.cloud.qdrant.io:6333')
        self.qdrant_api_key = os.getenv('QDRANT_API_KEY', 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.qyts-gqayBpiQZdMckSnzcZOqoyE-eB6wmyd_alR_6w')
        
        # File Storage
        self.upload_dir = os.getenv('UPLOAD_DIR', './data/documents')
        self.vector_dir = os.getenv('VECTOR_DIR', './data/vectors')
        self.cache_dir = os.getenv('CACHE_DIR', './data/cache')
        self.log_dir = os.getenv('LOG_DIR', './data/logs')
        
        # Application Settings
        self.app_name = os.getenv('APP_NAME', 'Policy Document AI')
        self.app_version = os.getenv('APP_VERSION', '1.0.0')
        self.debug = os.getenv('DEBUG', 'True').lower() == 'true'
        self.log_level = os.getenv('LOG_LEVEL', 'INFO')
        
        # OpenAI Configuration
        self.openai_model = os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')
        self.embedding_model = os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small')  # Changed default
        self.max_tokens = int(os.getenv('MAX_TOKENS', '4000'))
        self.temperature = float(os.getenv('TEMPERATURE', '0.1'))
        
        # Vector Search Configuration
        self.vector_dimension = int(os.getenv('VECTOR_DIMENSION', '1536'))  # text-embedding-3-small is 1536 dimensions
        self.similarity_threshold = float(os.getenv('SIMILARITY_THRESHOLD', '0.7'))
        self.max_chunks_per_query = int(os.getenv('MAX_CHUNKS_PER_QUERY', '10'))
        
        # Server Configuration
        self.api_host = os.getenv('API_HOST', '0.0.0.0')
        self.api_port = int(os.getenv('API_PORT', '8000'))
        self.streamlit_port = int(os.getenv('STREAMLIT_PORT', '8501'))
        
        # Security
        self.secret_key = os.getenv('SECRET_KEY', 'change-this-secret-key')
        self.allowed_hosts = os.getenv('ALLOWED_HOSTS', 'localhost,127.0.0.1').split(',')
        
        # Debug prints
        print(f"OpenAI key loaded: {len(self.openai_api_key)} chars")
        print(f"LlamaParse key loaded: {len(self.llamaparse_api_key)} chars")
        print(f"Database: {self.db_host}:{self.db_port}/{self.db_name}")
        print(f"Qdrant URL: {self.qdrant_url}")
        
        # Create directories if they don't exist
        os.makedirs(self.upload_dir, exist_ok=True)
        os.makedirs(self.vector_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

settings = Settings()