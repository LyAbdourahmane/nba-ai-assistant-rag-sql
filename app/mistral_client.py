import logging
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from utils.config import MISTRAL_API_KEY, MODEL_NAME


# ================================================================
# Initialisation du client Mistral

if not MISTRAL_API_KEY:
    raise ValueError("MISTRAL_API_KEY manquante dans le fichier .env")

if not MODEL_NAME:
    raise ValueError("MODEL_NAME manquant dans utils/config.py")

client = MistralClient(api_key=MISTRAL_API_KEY)


# ================================================================
# Fonction utilitaire : appel au modèle Mistral

def mistral_chat(system_prompt: str, user_message: str, temperature: float = 0.1) -> str:
    """
    Envoie un message au modèle Mistral avec un prompt système + message utilisateur.
    Retourne uniquement le texte de la réponse.
    """

    try:
        logging.info(f"[MISTRAL] Appel modèle={MODEL_NAME} | temp={temperature}")

        response = client.chat(
            model=MODEL_NAME,
            messages=[
                ChatMessage(role="system", content=system_prompt),
                ChatMessage(role="user", content=user_message)
            ],
            temperature=temperature
        )

        answer = response.choices[0].message.content.strip()
        logging.info("[MISTRAL] Réponse reçue.")

        return answer

    except Exception as e:
        logging.error(f"[MISTRAL] Erreur lors de l'appel : {e}")
        logging.error(f"[MISTRAL] Prompt système envoyé : {system_prompt[:500]}...")
        logging.error(f"[MISTRAL] Message utilisateur : {user_message}")

        return "Une erreur est survenue lors de l'appel au modèle Mistral."
