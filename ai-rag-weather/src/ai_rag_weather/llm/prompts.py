SYSTEM_PROMPT = """
You are an expert assistant. Always answer concisely, cite sources (page numbers), and use a friendly, professional tone.
"""

USER_DOC_QA_PROMPT = """
Given the following context chunks from a document, answer the user's question. Cite page numbers in your answer.

Context:
{context}

Question: {question}
"""

USER_WEATHER_PROMPT = """
Given the following weather data, answer the user's question in a clear, concise way.

Weather Data:
{weather}

Question: {question}
"""
