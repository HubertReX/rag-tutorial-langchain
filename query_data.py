import argparse
from rich import print
from rich.pretty import Pretty
from rich.console import Console
from rich.status import Status


DEBUG = False
NUMBER_OF_CONTEXT_BINDINGS = 3
CHROMA_PATH = "chroma"
OLLAMA_MODEL = "mistral" # llama3, llama2 (=7b), llama2:13b, mistral, wizardlm2

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based only on the above context: {question}
"""


def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    console = Console()
    with console.status("[bold blue]Processing query...[/]", spinner="dots12") as status:
        query_rag(query_text, status)


def query_rag(query_text: str, status: Status | None = None):
    # Prepare the DB.
    from langchain.vectorstores.chroma import Chroma
    from langchain.prompts import ChatPromptTemplate
    from langchain_community.llms.ollama import Ollama
        
    from get_embedding_function import get_embedding_function

    if status:
        status.update(status="[bold blue]Initializing ChromaDB...")
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    if status:
        status.update(status="[bold blue]Getting context...")
    results = db.similarity_search_with_score(query_text, k=NUMBER_OF_CONTEXT_BINDINGS)
    if DEBUG:
        print(f"[yellow]{ '=' * 80}[/]")
        for i, result in enumerate(results):
            doc, _score = result
            print(f"[yellow]Context {i+1}:\nContent:[/]\n{doc.page_content}\n[yellow]Meta:[/]\n", Pretty(doc.metadata, expand_all=True), "")
    # exit(1)
    
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    if DEBUG:
        print(f"[yellow]{ '=' * 80}\nPrompt:\n")
        print(prompt)
        print(f"[yellow]{ '=' * 80}[/]")
    if status:
        status.update(status="[bold blue]Sending query...[/]")
    model = Ollama(model=OLLAMA_MODEL)
    response_text = model.invoke(prompt)

    if status:
        status.update(status="[bold blue]Done![/]")
        status.stop()
    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = [f"\n\n[green]Response:[/]\n{response_text}\n[green]Sources:[/]", Pretty(sources, expand_all=True)]
    print(*formatted_response)
    return response_text


if __name__ == "__main__":
    main()
