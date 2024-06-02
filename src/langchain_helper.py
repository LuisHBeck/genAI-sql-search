import os
from dotenv import load_dotenv
import db_helper
from langchain_community.llms import GooglePalm

load_dotenv()


def sample():
    """
    note: after apply few shots learning, now llm can perform correctly some query that before it cant

    examples of questions:
    "How many white color Levi's t-shirt i have?"
    "How much is the price of the inventory for all small size t-shirts?"
    "If we have to sell all the Nike's t-shirts today with discounts applied. How much revenue our store will generate (post discounts)?"
    """

    llm = GooglePalm(temperature=0.2)
    db_url = os.getenv("DB_URL")
    db_object = db_helper.create_sql_db_obj(db_url)
    print(db_helper.create_sql_db_chain(
        llm,
        db_object,
        "How many white color Levi's t-shirt i have?"
    ))