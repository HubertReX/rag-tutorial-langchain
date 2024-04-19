from query_data import query_rag
from langchain_community.llms.ollama import Ollama
from rich import print

from query_data import OLLAMA_MODEL

EVAL_PROMPT = """
Expected Response: {expected_response}
Actual Response: {actual_response}
---
(Answer with 'true' or 'false') Does the actual response match the expected response? 
"""


def test_monopoly_start_money():
    assert query_and_validate(
        question="How much total money does a player start with in Monopoly? (Answer with the number only)",
        expected_response="$1500",
    )
    
def test_monopoly_po_box():
    assert query_and_validate(
        question="What is the p.o. box of Hasbro Games? (Answer with the address only)",
        expected_response="P.O. Box 200, Pawtucket, RI 02862",
    )


def test_ticket_to_ride_rules():
    assert query_and_validate(
        question="How many points does the longest continuous train get in Ticket to Ride? (Answer with the number only)",
        expected_response="10 points",
    )


def query_and_validate(question: str, expected_response: str):
    response_text = query_rag(question)
    prompt = EVAL_PROMPT.format(
        expected_response=expected_response, actual_response=response_text
    )

    model = Ollama(model=OLLAMA_MODEL)
    evaluation_results_str = model.invoke(prompt)
    evaluation_results_str_cleaned = evaluation_results_str.strip().lower()

    print(prompt)

    if "true" in evaluation_results_str_cleaned:
        # Print response in Green if it is correct.
        print(f"[green]Response: {evaluation_results_str_cleaned}[/]")
        return True
    elif "false" in evaluation_results_str_cleaned:
        # Print response in Red if it is incorrect.
        print(f"[red]Response: {evaluation_results_str_cleaned}[/]")
        return False
    else:
        raise ValueError(
            f"[red]Invalid evaluation result. Cannot determine if 'true' or 'false'.[/]"
        )
