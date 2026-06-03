from openai import OpenAI
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
import os

load_dotenv()

openai_client = OpenAI(
    api_key = os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key= os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1"

)
#store vectores
vector_db = QdrantVectorStore.from_existing_collection(
    url = "http://localhost:6333",
    collection_name="learning-rag",
    embedding = embedding_model,
)


def process_queue(query: str):
    search_results= vector_db.similarity_search(query= query)

    context = "\n\n\n".join([f"""
        page_content: {result.page_content}
        page_number: {result.metadata.get('page_label')}
        file_location: {result.metadata.get('source')}
        """
        for result in search_results
    ])

    system_prompt = f""" You are a helpful assistant.

    Answer the user's question ONLY from the provided context.

    Also tell the user which page number to open for more details.

    Context:
    {context}
    """

    response = openai_client.chat.completions.create(
        model = "openai/gpt-5-nano",
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
    )

    return response.choices[0].message.content