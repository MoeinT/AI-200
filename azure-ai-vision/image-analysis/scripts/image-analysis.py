import configparser
import os

from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential


# Class definition
class AzureImageAnalyzer:
    def __init__(self, ai_endpoint, ai_key):
        self.client = ImageAnalysisClient(
            endpoint=ai_endpoint, credential=AzureKeyCredential(ai_key)
        )

    @classmethod
    def from_config(cls, config_path: str = "config.ini"):
        config = configparser.ConfigParser(interpolation=None)
        config.read(config_path)
        ai_endpoint = config["azure-ai-services"]["AI_SERVICE_ENDPOINT"]
        ai_key = config["azure-ai-services"]["AI_SERVICE_KEY"]
        return cls(ai_endpoint=ai_endpoint, ai_key=ai_key)

    def analyze_image(self, image_path: str):
        with open(image_path, "rb") as f:
            image_data = f.read()
        try:
            # Get result with specified features to be retrieved
            result = self.client.analyze(
                image_data=image_data,
                visual_features=[
                    VisualFeatures.CAPTION,
                    VisualFeatures.DENSE_CAPTIONS,
                    VisualFeatures.TAGS,
                    VisualFeatures.OBJECTS,
                    VisualFeatures.PEOPLE,
                ],
            )
            return result
        except Exception as e:
            print(f"Error analyzing image: {e}")
            return None

    def print_results(self, result):
        if result is None:
            print("No result to display")
            return

        if result.caption is not None:
            print("\nCaption:")
            print(
                f"'{result.caption.text}' (confidence: {result.caption.confidence * 100:.2f}%)"
            )

        if len(result.people["values"]) > 1:
            print("\nPeople detected:")
            for person in result.people["values"]:
                print(f" - {person}")


if __name__ == "__main__":
    analyzer = AzureImageAnalyzer.from_config()
    image_path = "images/person.jpg"
    result = analyzer.analyze_image(image_path)
    analyzer.print_results(result)
