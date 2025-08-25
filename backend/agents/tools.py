import math
import re
from typing import Annotated

import numexpr
from langchain.tools.retriever import create_retriever_tool
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL

from backend.agents.rag import FAISSManager


@tool
def calculator(expression: str) -> str:
    """Calculates a math expression using numexpr.

    Useful for when you need to answer questions about math using numexpr.
    This tool is only for math questions and nothing else. Only input
    math expressions.

    Args:
        expression (str): A valid numexpr formatted math expression.

    Returns:
        str: The result of the math expression.
    """

    try:
        local_dict = {"pi": math.pi, "e": math.e}
        output = str(
            numexpr.evaluate(
                expression.strip(),
                global_dict={},  # restrict access to globals
                local_dict=local_dict,  # add common mathematical functions
            )
        )
        return re.sub(r"^\[|\]$", "", output)
    except Exception as e:
        raise ValueError(
            f'calculator("{expression}") raised error: {e}.'
            " Please try again with a valid numerical expression"
        )


# TODO: Convert function using @tool decorator and use search()
# method of FAISSManager instead of create_retriever_tool function
# to handle metadata returned by search method
def get_retriever_tool():
    vector_db = FAISSManager(index_name="faiss_index")
    retriever = vector_db.get_retriever()
    retriever_tool = create_retriever_tool(
        retriever,
        "retrieve_docs",
        "Search and return information from the documents.",
        response_format="content_and_artifact",
    )
    return retriever_tool


@tool
def python_repl(code: Annotated[str, "Python code or filename to read the code from"]):
    """Use this tool to execute python code. Make sure that you input the code correctly.
    Either input actual code or filename of the code. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user.
    """

    try:
        result = PythonREPL().run(code)
        print("RESULT CODE EXECUTION:", result)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    return f"Executed:\n```python\n{code}\n```\nStdout: {result}"
