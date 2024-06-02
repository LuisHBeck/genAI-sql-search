from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
import few_shots_learning as fsl


def create_sql_db_obj(db_url:str):
    """
    This function is responsible to return a
    database object from a db url connection
    :param db_url: database url connection string
    :return: database object
    """
    return SQLDatabase.from_uri(
        db_url,
        sample_rows_in_table_info=3
    )


def create_sql_db_chain(llm, db_object, query:str):
    # create a prompt with some query examples
    # to grant more assertiveness in responses
    few_shots_prompt = fsl.create_few_shots_prompt_template()

    db_chain = SQLDatabaseChain.from_llm(
        llm,
        db_object,
        verbose=True, prompt=few_shots_prompt
    )
    return db_chain.run(query)