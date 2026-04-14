def calculator_tool(expression: str) -> str:
    try:
        result = eval(expression)
        return f"Résultat : {result}"
    except Exception:
        return "Erreur dans le calcul"