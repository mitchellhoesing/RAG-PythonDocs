import argparse
import os
from ChromaDB import ChromaDB
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv

DATA_PATH = r"data/"
CHROMA_PATH = r"chroma"
load_dotenv('.env')

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--query_text", type=str, help="The query text.", required=True)
    parser.add_argument("--create_database", type=str, help="Option to create Chroma database.")
    args = parser.parse_args()
    query_text = args.query_text
    create_database = args.create_database

    db = ChromaDB(DATA_PATH, CHROMA_PATH)

    if create_database == "True":
        print("Creating database...")
        db.create_database()

    embedding_function = OpenAIEmbeddings(openai_api_key=os.getenv('OPENAI_API_KEY'))
    db = db.load_database(embedding_function)

    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    if len(results) == 0 or results[0][1] < 0.7:
        print(f"Unable to find matching results.")
        return

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(f"Prompt:\n\n {prompt}")

    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0.3,
        openai_api_key=os.getenv('OPENAI_API_KEY'))
    response_text = llm.invoke(prompt)

    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)


if __name__ == '__main__':
    main()


