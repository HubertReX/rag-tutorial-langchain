from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.embeddings.bedrock import BedrockEmbeddings

OLLAMA_EMBEDDINGS_MODEL = "snowflake-arctic-embed" # nomic-embed-text, snowflake-arctic-embed 

def get_embedding_function(embeddings_provider="Ollama"):
    if embeddings_provider == "Ollama":
        embeddings = OllamaEmbeddings(model=OLLAMA_EMBEDDINGS_MODEL)
    elif embeddings_provider == "Bedrock":
        embeddings = BedrockEmbeddings(
            credentials_profile_name="default", region_name="us-east-1"
        )
    else:
        raise NotImplementedError(f"Provider '{embeddings_provider}' is not implemented. Only 'Ollama' and 'Bedrock' are currently supported.")
    return embeddings
