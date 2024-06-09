import configparser
import logging
import os

from azure.ai.textanalytics import TextAnalyticsClient
# import namespaces
from azure.core.credentials import AzureKeyCredential


class AzureTextAnalyzer:
    def __init__(self, ai_endpoint, credential):
        # creates a client for the Text Analytics API
        self.ai_client = TextAnalyticsClient(
            endpoint=ai_endpoint, credential=credential
        )

    @classmethod
    def from_config(cls, config_path: str):
        config = configparser.ConfigParser(interpolation=None)
        config.read(config_path)
        ai_endpoint = config["azure-ai-language"]["AI_SERVICE_ENDPOINT"]
        ai_key = config["azure-ai-language"]["AI_SERVICE_KEY"]
        credential = AzureKeyCredential(ai_key)
        return cls(ai_endpoint=ai_endpoint, credential=credential)

    def _handle_error(self, method_name, ex):
        logging.error(f"An error occurred in '{method_name}': {ex}")
        return []

    def detect_language(self, text: str) -> str:
        try:
            response = self.ai_client.detect_language(documents=[text])[0]
            return response.primary_language.name
        except Exception as ex:
            return self._handle_error("detect_language", ex)

    def detect_sentiment(self, text: str):
        try:
            response = self.ai_client.analyze_sentiment(documents=[text])[0]
            return response.sentiment
        except Exception as ex:
            return self._handle_error("detect_sentiment", ex)

    def detect_key_phrases(self, text: str):
        try:
            # Get key phrases
            phrases = self.ai_client.extract_key_phrases(documents=[text])[
                0
            ].key_phrases
            if len(phrases) > 0:
                l_key_phrases = []
                for phrase in phrases:
                    l_key_phrases.append(phrase)
            return l_key_phrases
        except Exception as ex:
            print(ex)
            return self._handle_error("detect_key_phrases", ex)

    def detect_entities(self, text: str):
        try:
            # Get entities
            entities = self.ai_client.recognize_entities(documents=[text])[0].entities
            if len(entities) > 0:
                dict_entities = {}
                for entity in entities:
                    dict_entities[entity.text] = entity.category
                return dict_entities
        except Exception as ex:
            return self._handle_error("detect_entities", ex)

    def detect_linked_entities(self, text: str):
        # Get linked entities
        try:
            entities = self.ai_client.recognize_linked_entities(documents=[text])[
                0
            ].entities
            if len(entities) > 0:
                dict_entities = {}
                for linked_entity in entities:
                    dict_entities[linked_entity.name] = linked_entity.url
                return dict_entities
        except Exception as ex:
            return self._handle_error("detect_linked_entities", ex)


if __name__ == "__main__":
    analyzer = AzureTextAnalyzer.from_config(config_path="config.ini")
    # Detect language
    detected_language = analyzer.detect_language(text="Bonjour, tout le monde")
    print(detected_language)
    # Detect sentiment (positive, neutral or negative)
    detected_sentiment = analyzer.detect_sentiment(
        text="There's a beautiful rainbow outside!"
    )
    print(detected_sentiment)
    # Detect key phrases
    detected_key_phrases = analyzer.detect_key_phrases(
        text="The journey of a thousand miles begins with a single step."
    )
    print(detected_key_phrases)
    # Detect key entities
    detected_entities = analyzer.detect_entities(text="Joe went to London on Saturday")
    print(detected_entities)
    # Venus the planet
    detected_linked_entities = analyzer.detect_linked_entities(
        text="I saw Venus shining in the sky"
    )
    print(detected_linked_entities)
    # Venus Methodology
    detected_linked_entities = analyzer.detect_linked_entities(
        text="Venus, the goddess of beauty"
    )
    print(detected_linked_entities)
