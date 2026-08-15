import pandas as pd


# Words/patterns that must never appear
# in LLM-generated pandas code.
BLOCKED_PATTERNS = [
    "import ",
    "__import__",
    "open(",
    "exec(",
    "eval(",
    "compile(",
    "os.",
    "sys.",
    "subprocess",
    "socket",
    "requests",
    "urllib",
    "http",
    "pathlib",
    "shutil",
    "__",
]


def validate_code(code: str):
    """
    Validate LLM-generated pandas code before execution.

    Returns:
        (True, "") if safe.
        (False, reason) if blocked.
    """

    lowered_code = code.lower()

    for pattern in BLOCKED_PATTERNS:
        if pattern.lower() in lowered_code:
            return (
                False,
                f"Blocked operation detected: {pattern}"
            )

    # Generated code must produce a variable named result.
    if "result" not in code:
        return (
            False,
            "Generated code must assign the final output "
            "to a variable named result."
        )

    return True, ""


def execute_pandas_code(code: str, df: pd.DataFrame):
    """
    Execute validated pandas code in a restricted namespace.

    The LLM can access only:
        df
        pd

    Returns:
        result, error
    """

    is_safe, validation_error = validate_code(code)

    if not is_safe:
        return None, validation_error

    # Restricted namespace
    safe_globals = {
        "__builtins__": {},
        "pd": pd,
    }

    safe_locals = {
        "df": df.copy()
    }

    try:
        exec(
            code,
            safe_globals,
            safe_locals
        )

        result = safe_locals.get("result")

        if result is None:
            return (
                None,
                "The generated code did not create "
                "a result variable."
            )

        # Convert Series to DataFrame for consistent UI output
        if isinstance(result, pd.Series):
            result = result.reset_index()

        # Scalar result
        if not isinstance(
            result,
            (pd.DataFrame, pd.Series)
        ):
            result = pd.DataFrame(
                {
                    "Result": [result]
                }
            )

        return result, None

    except Exception as error:

        return None, str(error)