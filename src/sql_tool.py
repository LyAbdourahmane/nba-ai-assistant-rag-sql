import os
import logging
from dotenv import load_dotenv

from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool

from langchain_mistralai import ChatMistralAI
from langchain.chains.sql_database.query import create_sql_query_chain

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from src.validation_pydantic import SQLResponse
import logfire 


# ================================================================
# Configuration
logfire.configure()

load_dotenv()
DB_PATH = os.getenv("DB_PATH")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

if not DB_PATH:
    raise ValueError("DB_PATH manquant dans .env")

if not MISTRAL_API_KEY:
    raise ValueError("MISTRAL_API_KEY manquante dans .env")

DB = SQLDatabase.from_uri(DB_PATH)

llm = ChatMistralAI(
    mistral_api_key=MISTRAL_API_KEY,
    model="mistral-small-latest",
    temperature=0.1
)


# ================================================================
# Few-shots SQL

few_shoots = """
Question : Combien de joueurs sont présents dans la base ?
SQL : SELECT COUNT(*) FROM players;

Question : Quels sont les joueurs des Denver Nuggets ?
SQL : SELECT Player FROM players WHERE Team = 'DEN';

Question : Quelle est la moyenne de points de Nikola Jokić ?
SQL : SELECT AVG(PTS) FROM players WHERE Player = 'Nikola Jokić';

Question : Quels sont les 10 meilleurs scoreurs ?
SQL : SELECT Player, PTS FROM players ORDER BY PTS DESC LIMIT 10;

Question : Quels joueurs ont plus de 30 ans ?
SQL : SELECT Player, Age FROM players WHERE Age > 30;

Question : Quel est le nom complet de l’équipe de Shai Gilgeous-Alexander ?
SQL : SELECT t.TeamName FROM players p JOIN teams t ON p.Team = t.Code WHERE p.Player = 'Shai Gilgeous-Alexander';

Question : Trouve les joueurs dont le nom contient "Williams".
SQL : SELECT Player FROM players WHERE Player LIKE '%Williams%';

Question : Quels joueurs ont un TSpct supérieur à 60 ?
SQL : SELECT Player, TSpct FROM players WHERE TSpct > 60;
"""


# ================================================================
# Prompt SQL

sql_only_prompt = PromptTemplate.from_template(
f"""
Tu es un expert SQL spécialisé en SQLite et en analyse de données NBA.

Ta mission :
- Lire la question utilisateur.
- Générer une requête SQL SQLite valide.
- Utiliser uniquement les tables et colonnes présentes dans le schéma.
- Ne renvoyer QUE la requête SQL, sans texte autour.

Voici les tables disponibles :
{{table_info}}

Nombre maximum de lignes à renvoyer : {{top_k}}

Exemples :
{few_shoots}

Règles :
- Pas de texte hors SQL.
- Pas de commentaires.
- Pas de LIMIT 1 inutile.
- Utilise ORDER BY + LIMIT pour les tops.
- Utilise Player (nom complet) et Team (code à 3 lettres).

Question :
{{input}}

SQLQuery:
"""
)


# ================================================================
# Prompt final

answer_prompt = PromptTemplate.from_template(
"""
Tu es un assistant expert en analyse NBA.

Règles :
- Réponds uniquement à partir du résultat SQL.
- Si le résultat est vide, dis-le simplement.
- Si erreur SQL, explique-la simplement.
- Réponds en français, de manière concise et naturelle.
- Ne montre jamais la requête SQL.

Question :
{question}

Requête SQL générée :
{query}

Résultat SQL :
{result}

Réponse finale :
"""
)

rephrase_answer = answer_prompt | llm | StrOutputParser()


# ================================================================
# Chaînes SQL

generate_sql = create_sql_query_chain(
    llm=llm,
    db=DB,
    prompt=sql_only_prompt
)

execute_query = QuerySQLDatabaseTool(db=DB)


# ================================================================
# Fonctions utilitaires

def validate_sql(query: str) -> str:
    q = query.lower().strip()

    if not q.startswith("select"):
        raise ValueError("Seules les requêtes SELECT sont autorisées.")

    forbidden = ["drop", "delete", "update", "insert", "alter"]
    if any(word in q for word in forbidden):
        raise ValueError("Requête SQL potentiellement dangereuse.")

    return query


def safe_execute(query: str):
    try:
        return execute_query.run(query)
    except Exception as e:
        return f"ERREUR_SQL: {str(e)}"


# ================================================================
# Pipeline NL → SQL → Réponse
@logfire.instrument()
def nl_2_sql(question: str) -> dict:
    logfire.info("SQL Tool start", question=question)
    try:
        agent = (
            RunnablePassthrough.assign(query=generate_sql)
            .assign(query=lambda x: validate_sql(x["query"]))
            .assign(result=lambda x: safe_execute(x["query"]))
            | rephrase_answer
        )

        result = agent.invoke({"question": question})
        logfire.info("SQL result", result=result)

        return SQLResponse(
            answer=result,
            contexts=[]
        ).model_dump()

    except Exception as e:
        logfire.error("SQL Tool error", error=str(e))
        return SQLResponse(
            answer=f"Erreur lors du traitement SQL : {e}",
            contexts=[]
        ).model_dump()
