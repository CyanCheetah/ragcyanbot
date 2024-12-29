import os
import logging
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_environment():
    try:
        # Get current directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Create necessary directories
        directories = [
            os.path.join(current_dir, "pyllama", "models"),
            os.path.join(current_dir, "documents"),
            os.path.join(current_dir, "chroma_db"),
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Created directory: {directory}")
        
        # Create sample document
        sample_doc = os.path.join(current_dir, "documents", "sample.txt")
        with open(sample_doc, "w") as f:
            f.write("""
I am an AI assistant powered by the LLaMA language model. I can help with various tasks including:
- Answering questions about technology and AI
- Providing explanations and clarifications
- Assisting with general knowledge queries
- Engaging in helpful conversations

I aim to be helpful while being clear and concise in my responses. I base my knowledge on the information provided in my training data and the documents I have access to.
""".strip())
        logger.info(f"Created sample document: {sample_doc}")
        
        # Clean up existing vector store
        chroma_db = os.path.join(current_dir, "chroma_db")
        if os.path.exists(chroma_db):
            shutil.rmtree(chroma_db)
            logger.info("Cleaned up existing vector store")
        
        return True
    except Exception as e:
        logger.error(f"Setup failed: {str(e)}", exc_info=True)
        return False

if __name__ == "__main__":
    logger.info("Starting setup...")
    success = setup_environment()
    logger.info(f"Setup {'completed successfully' if success else 'failed'}") 