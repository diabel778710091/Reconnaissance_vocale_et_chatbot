import os
import nltk
import spacy
from nltk.tokenize import sent_tokenize
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import speech_recognition as sr
import streamlit as st

# Téléchargement des ressources NLTK
nltk.download('punkt')

# Chargement du modèle spaCy pour le français
nlp = spacy.load('fr_core_news_sm')

# Chargement du modèle SentenceTransformer
model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

# Charger le modèle de génération de réponses
generator = pipeline('text-generation', model='distilgpt2')


def load_and_preprocess_text(file_path):
    """Charge et prétraite le texte à partir d'un fichier."""
    try:
        with open(file_path, 'r', encoding='latin-1') as file:
            content = file.read()
        sentences = sent_tokenize(content)
        return sentences
    except Exception as e:
        print(f"Erreur lors du chargement du fichier : {e}")
        return []


def preprocess(text):
    """Prétraite le texte en le lemmatisant et en supprimant les mots vides."""
    doc = nlp(text)
    return ' '.join([token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct])


def preprocess_corpus(phrases):
    """Prétraite l'ensemble des phrases."""
    return [preprocess(phrase) for phrase in phrases]


def initialize_vectorizer(corpus):
    """Initialise le vectoriseur en encodant le corpus."""
    try:
        return model.encode(corpus)
    except Exception as e:
        print(f"Erreur lors de la vectorisation : {e}")
        return None


def get_most_relevant_sentence(question, vectors, phrases):
    """Trouve la phrase la plus pertinente pour une question donnée."""
    try:
        processed_question = preprocess(question)
        query_vec = model.encode([processed_question])
        similarities = util.pytorch_cos_sim(query_vec, vectors).flatten()
        index = similarities.argmax()
        return phrases[index]
    except Exception as e:
        print(f"Erreur lors du calcul de la similarité : {e}")
        return "Désolé, je n'ai pas pu trouver une réponse pertinente."


def speech_to_text():
    """Convertit la parole en texte à l'aide de la reconnaissance vocale."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Parlez maintenant...")
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio, language='fr-FR')
            st.write("Vous avez dit : " + text)
            return text
        except sr.UnknownValueError:
            st.write("Désolé, je n'ai pas compris.")
            return None
        except sr.RequestError:
            st.write("Erreur de connexion au service de reconnaissance vocale.")
            return None


def app():
    """Application Streamlit pour le chatbot."""
    st.title("ChatBot à Commande Vocale")

    file_path = 'Amélioration_des_services_bancaires (1).txt'
    phrases = load_and_preprocess_text(file_path)
    corpus = preprocess_corpus(phrases)
    vectors = initialize_vectorizer(corpus)

    if vectors is not None:
        option = st.selectbox("Choisissez la méthode d'entrée", ["Texte", "Voix"])

        if option == "Texte":
            entree_utilisateur = st.text_input("Entrez votre question ici : ")
            if entree_utilisateur:
                reponse = get_most_relevant_sentence(entree_utilisateur, vectors, phrases)
                st.write("ChatBot :", reponse)

        elif option == "Voix":
            if st.button("Enregistrer la voix"):
                # Traitement de la reconnaissance vocale
                entree_utilisateur = speech_to_text()

                # Vérification de la conversion avant de générer une réponse
                if entree_utilisateur:
                    reponse = get_most_relevant_sentence(entree_utilisateur, vectors, phrases)
                    st.write("ChatBot :", reponse)
    else:
        st.write("Erreur dans la vectorisation des phrases.")


if __name__ == "__main__":
    app()  # Pour exécuter avec Streamlit
