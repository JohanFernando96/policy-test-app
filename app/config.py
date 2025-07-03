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
        self.qdrant_url = os.getenv('QDRANT_URL',
                                    'https://ca579e00-fc37-4a9a-927f-118560d6b4de.eu-west-2-0.aws.cloud.qdrant.io:6333')
        self.qdrant_api_key = os.getenv('QDRANT_API_KEY',
                                        'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.qyts-gqayBpiQZdMckSnzcZOqoyE-eB6wmyd_alR_6w')

        # File Storage
        self.upload_dir = os.getenv('UPLOAD_DIR', './data/documents')
        self.vector_dir = os.getenv('VECTOR_DIR', './data/vectors')
        self.cache_dir = os.getenv('CACHE_DIR', './data/cache')
        self.log_dir = os.getenv('LOG_DIR', './data/logs')

        # Application Settings
        self.app_name = os.getenv('APP_NAME', 'Policy Document AI')
        self.app_version = os.getenv('APP_VERSION', '2.0.0')  # Updated version
        self.debug = os.getenv('DEBUG', 'True').lower() == 'true'
        self.log_level = os.getenv('LOG_LEVEL', 'INFO')

        # OpenAI Configuration
        self.openai_model = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')
        self.embedding_model = os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small')
        self.max_tokens = int(os.getenv('MAX_TOKENS', '4000'))
        self.temperature = float(os.getenv('TEMPERATURE', '0.1'))

        # Vector Search Configuration
        self.vector_dimension = int(os.getenv('VECTOR_DIMENSION', '1536'))  # text-embedding-3-small is 1536 dimensions
        self.similarity_threshold = float(os.getenv('SIMILARITY_THRESHOLD', '0.7'))
        self.max_chunks_per_query = int(os.getenv('MAX_CHUNKS_PER_QUERY', '10'))

        # Intelligent Chunking Configuration
        self.enable_intelligent_chunking = os.getenv('ENABLE_INTELLIGENT_CHUNKING', 'True').lower() == 'true'
        self.max_chunk_size = int(os.getenv('MAX_CHUNK_SIZE', '1200'))  # characters
        self.min_chunk_size = int(os.getenv('MIN_CHUNK_SIZE', '300'))  # characters
        self.chunk_overlap = int(os.getenv('CHUNK_OVERLAP', '200'))  # characters
        self.semantic_analysis_enabled = os.getenv('SEMANTIC_ANALYSIS_ENABLED', 'True').lower() == 'true'
        self.hierarchical_chunking_enabled = os.getenv('HIERARCHICAL_CHUNKING_ENABLED', 'True').lower() == 'true'

        # LLM Analysis Configuration
        self.llm_analysis_model = os.getenv('LLM_ANALYSIS_MODEL', 'gpt-4o-mini')
        self.max_analysis_tokens = int(os.getenv('MAX_ANALYSIS_TOKENS', '1000'))
        self.analysis_temperature = float(os.getenv('ANALYSIS_TEMPERATURE', '0.1'))

        # Context Enhancement Configuration
        self.enable_context_enhancement = os.getenv('ENABLE_CONTEXT_ENHANCEMENT', 'True').lower() == 'true'
        self.context_window_size = int(os.getenv('CONTEXT_WINDOW_SIZE', '2'))  # chunks before/after
        self.enable_relationship_mapping = os.getenv('ENABLE_RELATIONSHIP_MAPPING', 'True').lower() == 'true'

        # Search Enhancement Configuration
        self.enable_query_enhancement = os.getenv('ENABLE_QUERY_ENHANCEMENT', 'True').lower() == 'true'
        self.enable_intent_analysis = os.getenv('ENABLE_INTENT_ANALYSIS', 'True').lower() == 'true'
        self.enable_context_expansion = os.getenv('ENABLE_CONTEXT_EXPANSION', 'True').lower() == 'true'

        # Performance Configuration
        self.embedding_batch_size = int(os.getenv('EMBEDDING_BATCH_SIZE', '50'))  # Reduced for stability
        self.llm_processing_timeout = int(os.getenv('LLM_PROCESSING_TIMEOUT', '120'))  # seconds
        self.max_concurrent_requests = int(os.getenv('MAX_CONCURRENT_REQUESTS', '5'))

        # Fallback Configuration
        self.enable_fallback_chunking = os.getenv('ENABLE_FALLBACK_CHUNKING', 'True').lower() == 'true'
        self.fallback_chunk_method = os.getenv('FALLBACK_CHUNK_METHOD', 'paragraph')  # paragraph, sentence, token

        # Monitoring and Logging
        self.enable_chunking_metrics = os.getenv('ENABLE_CHUNKING_METRICS', 'True').lower() == 'true'
        self.log_chunk_statistics = os.getenv('LOG_CHUNK_STATISTICS', 'True').lower() == 'true'
        self.save_chunking_analysis = os.getenv('SAVE_CHUNKING_ANALYSIS', 'True').lower() == 'true'

        # Server Configuration
        self.api_host = os.getenv('API_HOST', '0.0.0.0')
        self.api_port = int(os.getenv('API_PORT', '8000'))
        self.streamlit_port = int(os.getenv('STREAMLIT_PORT', '8501'))

        # Security
        self.secret_key = os.getenv('SECRET_KEY', 'change-this-secret-key')
        self.allowed_hosts = os.getenv('ALLOWED_HOSTS', 'localhost,127.0.0.1').split(',')

        # Advanced Features
        self.enable_multilingual_support = os.getenv('ENABLE_MULTILINGUAL_SUPPORT', 'False').lower() == 'true'
        self.enable_document_versioning = os.getenv('ENABLE_DOCUMENT_VERSIONING', 'False').lower() == 'true'
        self.enable_audit_trail = os.getenv('ENABLE_AUDIT_TRAIL', 'True').lower() == 'true'

        # Additional Features for Compatibility (added these)
        self.enable_structured_extraction = os.getenv('ENABLE_STRUCTURED_EXTRACTION', 'True').lower() == 'true'
        self.enable_section_detection = os.getenv('ENABLE_SECTION_DETECTION', 'True').lower() == 'true'
        self.enable_table_extraction = os.getenv('ENABLE_TABLE_EXTRACTION', 'True').lower() == 'true'

        # Processing Configuration
        self.max_file_size_mb = int(os.getenv('MAX_FILE_SIZE_MB', '50'))  # Maximum file size in MB
        self.processing_batch_size = int(os.getenv('PROCESSING_BATCH_SIZE', '10'))  # Chunks per batch

        # Error Handling Configuration
        self.retry_attempts = int(os.getenv('RETRY_ATTEMPTS', '3'))
        self.retry_delay = float(os.getenv('RETRY_DELAY', '1.0'))  # seconds
        self.enable_graceful_degradation = os.getenv('ENABLE_GRACEFUL_DEGRADATION', 'True').lower() == 'true'

        # Feature flags for safe deployment
        self.enable_experimental_features = os.getenv('ENABLE_EXPERIMENTAL_FEATURES', 'False').lower() == 'true'
        self.enable_beta_chunking = os.getenv('ENABLE_BETA_CHUNKING', 'False').lower() == 'true'

        # Caching Configuration
        self.enable_embedding_cache = os.getenv('ENABLE_EMBEDDING_CACHE', 'True').lower() == 'true'
        self.cache_ttl_hours = int(os.getenv('CACHE_TTL_HOURS', '24'))  # Cache time-to-live

        # Rate Limiting
        self.api_rate_limit_per_minute = int(os.getenv('API_RATE_LIMIT_PER_MINUTE', '60'))
        self.embedding_rate_limit_per_minute = int(os.getenv('EMBEDDING_RATE_LIMIT_PER_MINUTE', '1000'))

        # Debug and status output
        self._print_startup_info()

        # Create directories
        self._create_directories()

    def _print_startup_info(self):
        """Print startup configuration information"""
        print(f"🔑 OpenAI key loaded: {len(self.openai_api_key)} chars")
        print(f"🔑 LlamaParse key loaded: {len(self.llamaparse_api_key)} chars")
        print(f"🗄️ Database: {self.db_host}:{self.db_port}/{self.db_name}")
        print(f"🔍 Qdrant URL: {self.qdrant_url}")
        print(f"🧠 Intelligent chunking: {'✅ Enabled' if self.enable_intelligent_chunking else '❌ Disabled'}")
        print(f"📊 Semantic analysis: {'✅ Enabled' if self.semantic_analysis_enabled else '❌ Disabled'}")
        print(f"🏗️ Hierarchical chunking: {'✅ Enabled' if self.hierarchical_chunking_enabled else '❌ Disabled'}")
        print(f"💡 Query enhancement: {'✅ Enabled' if self.enable_query_enhancement else '❌ Disabled'}")

        # Additional feature status
        if self.debug:
            print(f"🔧 Debug mode: {'✅ Enabled' if self.debug else '❌ Disabled'}")
            print(f"📈 Chunking metrics: {'✅ Enabled' if self.enable_chunking_metrics else '❌ Disabled'}")
            print(f"💾 Analysis saving: {'✅ Enabled' if self.save_chunking_analysis else '❌ Disabled'}")

    def _create_directories(self):
        """Create necessary directories"""
        try:
            # Basic directories
            os.makedirs(self.upload_dir, exist_ok=True)
            os.makedirs(self.vector_dir, exist_ok=True)
            os.makedirs(self.cache_dir, exist_ok=True)
            os.makedirs(self.log_dir, exist_ok=True)

            # Analysis directories
            if self.save_chunking_analysis:
                os.makedirs(os.path.join(self.cache_dir, 'analysis'), exist_ok=True)
                os.makedirs(os.path.join(self.cache_dir, 'chunks'), exist_ok=True)
                os.makedirs(os.path.join(self.cache_dir, 'embeddings'), exist_ok=True)

            # Processing directories
            os.makedirs(os.path.join(self.cache_dir, 'processing'), exist_ok=True)
            os.makedirs(os.path.join(self.log_dir, 'errors'), exist_ok=True)

            print("✅ All directories created successfully")

        except Exception as e:
            print(f"⚠️ Warning: Could not create some directories: {e}")

    def get_max_file_size_bytes(self):
        """Get maximum file size in bytes"""
        return self.max_file_size_mb * 1024 * 1024

    def is_feature_enabled(self, feature_name: str) -> bool:
        """Check if a feature is enabled with safe fallback"""
        try:
            return getattr(self, feature_name, False)
        except AttributeError:
            return False

    def get_chunking_config(self) -> dict:
        """Get chunking configuration as dictionary"""
        return {
            "enabled": self.enable_intelligent_chunking,
            "max_chunk_size": self.max_chunk_size,
            "min_chunk_size": self.min_chunk_size,
            "overlap": self.chunk_overlap,
            "semantic_analysis": self.semantic_analysis_enabled,
            "hierarchical": self.hierarchical_chunking_enabled,
            "fallback_enabled": self.enable_fallback_chunking,
            "fallback_method": self.fallback_chunk_method
        }

    def get_openai_config(self) -> dict:
        """Get OpenAI configuration as dictionary"""
        return {
            "model": self.openai_model,
            "embedding_model": self.embedding_model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "analysis_model": self.llm_analysis_model,
            "analysis_tokens": self.max_analysis_tokens,
            "analysis_temperature": self.analysis_temperature
        }

    def get_vector_config(self) -> dict:
        """Get vector database configuration as dictionary"""
        return {
            "url": self.qdrant_url,
            "dimension": self.vector_dimension,
            "similarity_threshold": self.similarity_threshold,
            "max_chunks_per_query": self.max_chunks_per_query
        }

    def validate_configuration(self) -> dict:
        """Validate configuration and return status"""
        status = {
            "valid": True,
            "errors": [],
            "warnings": []
        }

        # Check API keys
        if not self.openai_api_key or len(self.openai_api_key) < 20:
            status["errors"].append("OpenAI API key missing or invalid")
            status["valid"] = False

        if not self.llamaparse_api_key or len(self.llamaparse_api_key) < 20:
            status["errors"].append("LlamaParse API key missing or invalid")
            status["valid"] = False

        # Check database configuration
        if not all([self.db_host, self.db_user, self.db_password, self.db_name]):
            status["errors"].append("Database configuration incomplete")
            status["valid"] = False

        # Check Qdrant configuration
        if not self.qdrant_url or not self.qdrant_api_key:
            status["errors"].append("Qdrant configuration missing")
            status["valid"] = False

        # Warnings for performance
        if self.embedding_batch_size > 100:
            status["warnings"].append("Large embedding batch size may cause rate limiting")

        if self.max_chunk_size > 2000:
            status["warnings"].append("Large chunk size may affect embedding quality")

        return status


# Create global settings instance
settings = Settings()

# Validate configuration on import
config_status = settings.validate_configuration()
if not config_status["valid"]:
    print("❌ Configuration validation failed:")
    for error in config_status["errors"]:
        print(f"  - {error}")

if config_status["warnings"]:
    print("⚠️ Configuration warnings:")
    for warning in config_status["warnings"]:
        print(f"  - {warning}")

if config_status["valid"]:
    print("✅ Configuration validation passed")