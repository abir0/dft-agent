import asyncio
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Annotated, List, Optional

import numexpr
from genson import SchemaBuilder
from langchain.tools.retriever import create_retriever_tool
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL

from agents.rag import FAISSManager
from database import AsyncPostgresManager
from schema import PostgresDBSearchInput, PostgresDBSearchOutput
from settings import settings


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


def json_snippet(
    data: List[dict], max_length: int = 2000, format_type: str = "json"
) -> str:
    """
    Convert a JSON data object into a partial snippet.

    Args:
        data (List[dict]): Array of JSON objects in Python data type
        max_length (int): Maximum length of the output snippet
        format_type (str): 'json' for JSON format

    Returns:
        str: A truncated JSON snippet or text view
    """
    if format_type == "json":
        if len(data) <= 5:
            return json.dumps(data, indent=2)
        json_str = json.dumps(data[:5], indent=2)
        return json_str[:max_length] + ("..." if len(json_str) > max_length else "")


def generate_schema(data: List[dict]) -> dict:
    builder = SchemaBuilder()
    builder.add_object(data)
    return builder.to_schema()


@tool(args_schema=PostgresDBSearchInput)
def postgres_db_search(
    query: str,
    parameters: Optional[List[dict]] = None,
) -> PostgresDBSearchOutput:
    """
    Searches and queries data from PostgreSQL using SQL syntax.
    This tool helps retrieve data from PostgreSQL tables based on user queries.
    It returns a dict containing the filepath of the full data and a textual snippet
    of the data.

    Useful for when you need to:
    - Search for specific items in the database
    - Query data using SQL syntax
    - Filter and retrieve records based on conditions

    The query should be in PostgreSQL SQL syntax. Example queries:
    - "SELECT * FROM table_name"
    - "SELECT data->>'field' FROM table_name WHERE data->>'category' = 'books'"

    Args:
        query (str): SQL query to execute
        parameters (Optional[List[dict]]): Optional query parameters in sqlialchemy format

    Returns:
        PostgresDBSearchOutput: A structured response containing a snippet of query results,
        the filename where the full results are saved, the count of items, and the status.
    """
    try:
        db_url = settings.DATABASE_URL

        if not db_url:
            raise ValueError(
                "PostgreSQL database URL not found in environment variables"
            )

        # Ensure parameters is either None or a valid list of parameter dictionaries
        if parameters is not None and not isinstance(parameters, list):
            parameters = None

        async def execute_search():
            async with AsyncPostgresManager(db_url) as manager:
                results = await manager.query_data(
                    query=query,
                    parameters=parameters or [],  # Pass empty list if parameters is None
                )
                return results

        results = asyncio.run(execute_search())

        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        filename = f"{settings.ROOT_PATH}/data/postgres/results_{timestamp}.json"
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)

        snippet = json_snippet(results) if results else ""
        # schema = generate_schema(results)
        return PostgresDBSearchOutput(
            snippet=snippet,
            # db_schema=schema,
            file=filename,
            count=len(results),
            status="success",
        )

    except Exception as e:
        return PostgresDBSearchOutput(
            snippet=None, db_schema=None, file=None, count=0, status=f"error: {str(e)}"
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
