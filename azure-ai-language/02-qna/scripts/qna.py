import configparser
import logging
import os

from azure.ai.textanalytics import TextAnalyticsClient

# import namespaces
from azure.core.credentials import AzureKeyCredential
from azure.ai.language.questionanswering import QuestionAnsweringClient


class AzureQuestionAnswer:
    def __init__(self, ai_endpoint, credential, ai_deployment_name, ai_project_name):
        # creates a client for the Text Analytics API
        self.ai_client = QuestionAnsweringClient(endpoint=ai_endpoint, credential=credential)
        self.ai_deployment_name = ai_deployment_name
        self.ai_project_name = ai_project_name

    @classmethod
    def from_config(cls, config_path: str):
        config = configparser.ConfigParser(interpolation=None)
        config.read(config_path)
        ai_endpoint = config["azure-ai-language"]["AI_SERVICE_ENDPOINT"]
        ai_key = config["azure-ai-language"]["AI_SERVICE_KEY"]
        ai_deployment_name = config["azure-ai-language"]["QA_DEPLOYMENT_NAME"]
        ai_project_name = config["azure-ai-language"]["QA_PROJECT_NAME"]
        credential = AzureKeyCredential(ai_key)
        return cls(
            ai_endpoint=ai_endpoint,
            credential=credential,
            ai_project_name=ai_project_name,
            ai_deployment_name=ai_deployment_name,
        )

    def _handle_error(self, method_name, ex):
        logging.error(f"An error occurred in '{method_name}': {ex}")
        return []

    def answer_question(self):
         # Submit a question and display the answer
        user_question=""
        while user_question.lower() != 'quit':
            user_question = input('\nQuestion:\n')
            response = self.ai_client.get_answers(question=user_question,
                                            project_name=self.ai_project_name,
                                            deployment_name=self.ai_deployment_name)
            for candidate in response.answers:
                print(candidate.answer)
                print("Confidence: {}".format(candidate.confidence))
                print("Source: {}".format(candidate.source))


if __name__ == "__main__":
    analyzer = AzureQuestionAnswer.from_config(config_path="config.ini")
    analyzer.answer_question()
