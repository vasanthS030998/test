import streamlit as st
import pandas as pd
import sqlalchemy
from langchain_ollama import OllamaLLM
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import os

# --- DB Configs ---
DB_CONFIGS = {
    "postgresql": {
        "type": "postgresql",
        "conn_string": "postgresql://postgres:admin@localhost:1234/ETL"
    },
    "mssql": {
        "type": "mssql",
        "conn_string": "mssql+pyodbc://admin:admin@127.0.0.1\\SQLEXPRESS/master?driver=ODBC+Driver+17+for+SQL+Server"
    }
}

# --- LLM Config ---
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", 'gemma3:12b')
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://100.106.200.114:11434")
llm = OllamaLLM(model=OLLAMA_MODEL, base_url=OLLAMA_HOST)

# --- Utility Functions (reusing your code) ---
def get_db_schema(conn_string):
    engine = sqlalchemy.create_engine(conn_string)
    inspector = sqlalchemy.inspect(engine)
    schema_info = {}
    with engine.connect() as conn:
        for table in inspector.get_table_names():
            schema_info[table] = inspector.get_columns(table)
    return schema_info, ""

def generate_sql_agentic(question, schema, db_name, metadata):
    prompt_template = """
    You are an AI assistant specialized in converting natural language to SQL queries.
    Given the database schema for the {db_name} database, your task is to generate a SQL query to answer the user's question.

    Database Name: {db_name}
    {schema_str}

    User Question: {question}

    Provide only the SQL query as your response, without any additional text, explanations, or formatting.
    """
    schema_str = "\n".join([
        f"Table: {table}\nColumns: {', '.join([col['name'] for col in columns])}"
        for table, columns in schema.items()
    ])
    prompt = PromptTemplate(
        input_variables=["question", "schema_str", "db_name"],
        template=prompt_template
    )
    raw_sql = LLMChain(llm=llm, prompt=prompt).run(question=question, schema_str=schema_str, db_name=db_name)
    return raw_sql.strip().replace("```sql", "").replace("```", "")

def execute_sql(conn_string, sql_query):
    engine = sqlalchemy.create_engine(conn_string)
    with engine.connect() as connection:
        result = connection.execute(sqlalchemy.text(sql_query))
        if result.returns_rows:
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
            return df
        else:
            return pd.DataFrame()

# --- Streamlit App ---
st.set_page_config(page_title="str_app_m2", layout="wide")

st.title("🧠 Agentic SQL Generator & Executor")

st.sidebar.header("Database Selection")
db_name = st.sidebar.selectbox("Choose a database", list(DB_CONFIGS.keys()))

queries = st.text_area("Enter your queries (one per line)", 
    "I want a query that shows the number of currently active sessions for each user.\nColumn Counts\nColumn Level Privileges\nConstraint Details\nCPU Usage\nCPU Usage By Query Type\nDatabase Objects\nObject Level Privileges\nReferentialConstraints\nRole To User Mapping\nRows Returned\nSQL Executions")

if st.button("Run Queries"):
    db_config = DB_CONFIGS[db_name]
    schema, metadata = get_db_schema(db_config["conn_string"])
    query_list = [q.strip() for q in queries.split("\n") if q.strip()]
    
    for i, question in enumerate(query_list, start=1):
        st.subheader(f"Query {i}: {question}")
        sql = generate_sql_agentic(question, schema, db_name, metadata)
        st.code(sql, language="sql")
        try:
            df = execute_sql(db_config["conn_string"], sql)
            if not df.empty:
                st.dataframe(df)
                file_name = f"{db_name}_query_{i}_results.xlsx"
                df.to_excel(file_name, index=False)
                st.success(f"✅ Exported results to {file_name}")
                with open(file_name, "rb") as f:
                    st.download_button("⬇️ Download Excel", f, file_name)
            else:
                st.warning("Query executed successfully, but returned no results.")
        except Exception as e:
            st.error(f"❌ Failed to execute query: {e}")
