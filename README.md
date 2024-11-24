# **RAG-PythonDocs**

**RAG-PythonDocs** is a Retrieval-Augmented Generation (RAG) application designed to assist junior developers in querying and understanding Python documentation in context. By leveraging state-of-the-art language models and embeddings, the tool provides clear and accurate answers to Python-related questions.

---

## **Features**
- Query Python documentation directly to get contextual answers.
- Retrieves and displays the source(s) used to generate each answer, providing full context for the response.
- Simplifies complex Python concepts for junior developers.
- Lightweight and efficient architecture using LangChain and ChromaDB.
- Integration with gpt-3.5-turbo for enhanced natural language understanding.

---

## **Installation**

1. Clone the repository:
   ```bash
   git clone https://github.com/mitchellhoesing/RAG-PythonDocs.git
   cd RAG-PythonDocs
   ```

2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your OpenAI API key as an environment variable:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

---

## **Example Usage**

To query Python documentation, use the following command:

```bash
python main.py --query_text="How do I return a range from a list using indexing?" --create_database="True"
```

This example demonstrates how to query the application and build the vector embedding database.

**Note:** You only need to build the vector database once. If the vector embedding database has been built, set 
--create_database to "False"

**Using the OpenAI API for vector embeddings and gpt-3.5-turbo will incur charges.**

---

## **Architecture**

**RAG-PythonDocs** combines the following key components:
- **ChromaDB**: A high-performance vector database for efficient document retrieval.
- **OpenAI API & OpenAI Embeddings**: To generate semantic embeddings for the Python documentation.
- **gpt-3.5-turbo**: OpenAI's advanced language model for natural language processing.
- **LangChain**: For building RAG workflows and managing the interaction between components.


---

## **Examples/Demos**
You can view the index slice and stdout query examples here:
- **[Index slice query](index_slice_results.txt)**
- **[Stdout query](stdout_results.txt)**

---

## **Credits**

This project was inspired by the tutorial: [LangChain RAG Tutorial](https://github.com/pixegami/langchain-rag-tutorial). Special thanks to the open-source community for providing the tools and guidance for this project.

---

