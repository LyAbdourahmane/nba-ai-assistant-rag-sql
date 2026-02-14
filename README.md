# 📘 Assistant IA Basketball

### _RAG + SQL Tool + FAISS + Mistral + Logfire + Streamlit_

---

## **1. Présentation du projet**

Ce projet a été réalisé dans le cadre du d'un apprentissage.  
Il s’agit d’un **assistant IA complet**, capable de :

- répondre à des questions **basées sur des documents PDF** (RAG)
- répondre à des questions **basées sur une base SQLite** (SQL Tool)
- router automatiquement les questions vers le bon moteur (classification LLM)
- afficher une interface utilisateur moderne (Streamlit)
- tracer toutes les étapes du pipeline (Logfire)
- gérer automatiquement la base vectorielle FAISS

L’assistant est spécialisé dans l’analyse **basketball**, mais l’architecture est générique.

---

## **3. Fonctionnalités principales**

### **RAG (Retrieval-Augmented Generation)**

- Extraction texte PDF
- OCR automatique (EasyOCR) si texte absent
- Chunking intelligent (RecursiveCharacterTextSplitter)
- Embeddings Mistral
- Index FAISS (similarité cosinus)
- Prompt RAG optimisé
- Réponse contextualisée

### **SQL Tool**

- Génération SQL via Mistral
- Validation SQL
- Exécution sécurisée
- Reformulation de la réponse
- Compatible SQLite

### **Router intelligent**

- Classification LLM : _SQL_ ou _RAG_
- Routage automatique
- Gestion des erreurs

### **Interface Streamlit**

- Chat UI
- Historique des messages
- Reconstruction automatique FAISS
- Messages d’état
- Intégration propre du routeur

### **Observabilité Logfire**

- Traces détaillées
- Instrumentation des fonctions critiques
- Visualisation en temps réel
- Débogage facilité

---

## **4. Installation**

### 1️⃣ Cloner le projet

```bash
git clone https://github.com/LyAbdourahmane/nba-ai-assistant-rag-sql.git
cd nba-ai-assistant-rag-sql
```

### 2️⃣ Créer un environnement virtuel

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows
```

### 3️⃣ Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4️⃣ Ajouter les clés API dans `.env`

```
MISTRAL_API_KEY=xxxx...
```

---

## **6. Construction automatique de la base vectorielle**

Lors du premier lancement :

- FAISS n’existe pas -> reconstruction automatique
- OCR + extraction texte
- Chunking
- Embeddings
- Construction FAISS
- Sauvegarde dans `vector_db/`

Les fichiers générés :

```
vector_db/faiss_index.idx
vector_db/document_chunks.pkl
```

---

## **7. Lancer l’application**

```bash
streamlit run app/ui_streamlit.py
```

L’interface s’ouvre sur :

```
http://localhost:8501
```

---

## **9. Exemple de fonctionnement**

### 🔸 Question RAG

> _"Quelle équipe a le meilleur bilan cette saison ?"_

Pipeline :

1. Classification → RAG
2. Embedding de la requête
3. Recherche FAISS
4. Sélection des chunks
5. Prompt RAG
6. Réponse contextualisée

### 🔸 Question SQL

> _"Donne-moi les 5 meilleurs joueurs par points moyens."_

Pipeline :

1. Classification → SQL
2. Génération SQL
3. Validation
4. Exécution SQLite
5. Reformulation

---

## **10. Observabilité Logfire**

Chaque étape critique est tracée :

- classification
- routage
- recherche FAISS
- génération SQL
- exécution SQL
- génération RAG
- erreurs éventuelles

Dashboard :

👉 https://logfire.pydantic.dev/

---

## **11. Limites actuelles**

- OCR EasyOCR très lent sur CPU
- Coût API Mistral
- Pas de cache embeddings
- Pas de gestion multi-utilisateurs

---

## **12. Améliorations possibles**

- Remplacer EasyOCR par Tesseract
- Ajouter un cache embeddings local
- Ajouter un bouton Streamlit “Reconstruire FAISS”
- Ajouter un mode debug (afficher les chunks utilisés)
- Ajouter un toggle SQL/RAG manuel
- Ajouter un système de feedback utilisateur

---

# 👤 **Auteur**

**Abdourahamane LY**  
Data Scientist — MSc AI for Business  
Spécialiste RAG, MLOps, NLP, Computer Vision
