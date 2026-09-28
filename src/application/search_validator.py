def verify_search_query(query):
    """Valida a query de busca"""
    if not query or not query.strip():
        return "Your search is empty, try again!"
    if len(query) < 3:
        return "Your search is too short, try again!"
    if not any(c.isalnum() for c in query):
        return "Your search has only special characters, try again!"
    return "OK!"