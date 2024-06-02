from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.example_selectors import SemanticSimilarityExampleSelector

from langchain.chains.sql_database.prompt import PROMPT_SUFFIX, _mysql_prompt
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate

few_shots = [
    {
        'Question': "How many t-shirts do we have left for Nike in XL size and blue color?",
        'SQLQuery': "SELECT sum(stock_quantity) FROM t_shirts WHERE brand = 'Nike' AND color = 'White' AND size = 'XL'",
        'SQLResult': "Result of the SQL query",
        'Answer': '36'
    },
    {
        'Question': "How much is the total price of the inventory for all S-size t-shirts?",
        'SQLQuery': "SELECT SUM(price*stock_quantity) FROM t_shirts WHERE size = 'S'",
        'SQLResult': "Result of the SQL query",
        'Answer': '20421'},
    {
        'Question': "If we have to sell all the Levi’s T-shirts today with discounts applied. How much revenue  our store will generate (post discounts)?",
        'SQLQuery': """
            SELECT sum(a.total_amount * ((100-COALESCE(discounts.pct_discount,0))/100)) as total_revenue from
            (select sum(price*stock_quantity) as total_amount, t_shirt_id from t_shirts where brand = 'Levi'
            group by t_shirt_id) a left join discounts on a.t_shirt_id = discounts.t_shirt_id
        """,
        'SQLResult': "Result of the SQL query",
        'Answer': '32917.800000'},
    {
        'Question': "If we have to sell all the Levi’s T-shirts today. How much revenue our store will generate without discount?",
        'SQLQuery': "SELECT SUM(price * stock_quantity) FROM t_shirts WHERE brand = 'Levi'",
        'SQLResult': "Result of the SQL query",
        'Answer': '33375'
    },
    {
        'Question': "How many white color Levi's shirt I have?",
        'SQLQuery': "SELECT sum(stock_quantity) FROM t_shirts WHERE brand = 'Levi' AND color = 'White'",
        'SQLResult': "Result of the SQL query",
        'Answer': '411'
    }
]


def apply_few_shots_learning():
    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2'
    )

    # create a big and only one string with
    # the examples value for each obj in few_shots[]
    to_vectorize = [" ".join(example.values()) for example in few_shots]

    # saving to a vector database
    vectorstore = Chroma.from_texts(
        to_vectorize,
        embedding=embeddings,
        metadatas=few_shots
    )

    example_selector = SemanticSimilarityExampleSelector(
        vectorstore = vectorstore,
        k=2 # examples qnt
    )

    # search for similar examples
    example_selector.select_examples({'Question': "How many Adidas T shirts I have left in my store?"})
    return example_selector


def create_few_shots_prompt_template():
    """
    create a few shots prompt template ->
    say to LLM "if you are confuse, look into this vector db"
    :param example_selector: semantic similarity examples created
    :return: few shot prompt template
    """
    example_selector = apply_few_shots_learning()
    example_prompt = create_prompt_template()
    return FewShotPromptTemplate(
        example_selector = example_selector,
        example_prompt = example_prompt,
        prefix=_mysql_prompt,
        suffix=PROMPT_SUFFIX,
        input_variables=['input', 'table_info', 'top_k'] # these variables are used in the prefix and suffix
    )


def create_prompt_template():
    return PromptTemplate(
        input_variables = ['Question', 'SQLQuery', 'SQLResult', 'Answer'],
        template= "\nQuestion: {Question}\nSQLQuery: {SQLQuery}\nSQLResult: {SQLResult}\nAnswer: {Answer}"
    )