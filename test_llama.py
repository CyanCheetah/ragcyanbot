from langchain_community.llms import LlamaCpp
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_llama():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, "pyllama", "models", "llama-2-7b-chat.Q4_K_M.gguf")
        
        # Check if model exists
        if not os.path.exists(model_path):
            logger.error(f"Model file not found at: {model_path}")
            return False
            
        logger.info(f"Found model at: {model_path}")
        
        llm = LlamaCpp(
            model_path=model_path,
            temperature=0.7,
            max_tokens=256,
            top_p=1,
            n_ctx=2048,
            n_gpu_layers=1,
            n_batch=512,
            verbose=True,
            f16_kv=True,
            use_metal=True,
            seed=42
        )
        
        # Test the model
        prompt = "Say 'hello' in one word."
        logger.info(f"Testing model with prompt: {prompt}")
        response = llm.invoke(prompt)
        logger.info(f"Response: {response}")
        return True
        
    except Exception as e:
        logger.error(f"Error testing LLaMA: {str(e)}", exc_info=True)
        return False

if __name__ == "__main__":
    logger.info("Starting LLaMA test...")
    success = test_llama()
    logger.info(f"Test {'succeeded' if success else 'failed'}") 