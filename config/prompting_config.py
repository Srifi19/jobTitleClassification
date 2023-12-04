# examples for few shot learning
EXAMPLES = [
    {
        "query": "Python engineer, [Teacher, Software Developer, Scientist, Worker]",
        "answer": "Software Developer",
    },
    {
        "query": "Science Instructor, [Teacher, Software Developer, Scientist, Worker]",
        "answer": "Teacher",
    },
]

# template for example prompt
EXAMPLE_TEMPLATE = """
User: {query}
AI: {answer}
"""

# few shot learning prompt prefix
PREFIX = """
The following are excerpts from conversations with an AI assistant.
The assistant is helpful, and clever. From a given job title, and a list of some job titles, 
he can know which job title is the most similar to the given job title. Here are some examples:
"""

# few shot learning prompt suffix
SUFFIX = """
User: {query}
AI:
"""
