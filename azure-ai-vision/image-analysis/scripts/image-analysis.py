import configparser
import os

from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw


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
                print(f"{person}")

        for detected_object in result.objects.list:
            print("\nObjects:")
            print(
                "{} (confidence: {:.2f}%)".format(
                    detected_object.tags[0].name, detected_object.tags[0].confidence
                )
            )

    def annotate_entities_and_save_image(
        self,
        result,
        image_path: str,
        entity_type: str,
        outputfile: str = "results/objects.jpg",
    ):
        # Determine the type of entity (people or objects)
        entities = result.people if entity_type == "people" else result.objects
        if entities is not None:
            # Prepare image for drawing
            image = Image.open(image_path)
            fig = plt.figure(figsize=(image.width / 100, image.height / 100))
            plt.axis("off")
            draw = ImageDraw.Draw(image)
            color = "cyan"

            for entity in entities.list:
                # Draw object bounding box
                r = entity.bounding_box
                bounding_box = ((r.x, r.y), (r.x + r.width, r.y + r.height))
                draw.rectangle(bounding_box, outline=color, width=3)
                if entity_type == "objects":
                    plt.annotate(entity.tags[0].name, (r.x, r.y), backgroundcolor=color)

            # Save annotated image
            plt.imshow(image)
            plt.tight_layout(pad=0)
            fig.savefig(outputfile)
            print("\nResults saved in", outputfile)


if __name__ == "__main__":
    analyzer = AzureImageAnalyzer.from_config()
    image_path = "images/flower.jpg"
    result = analyzer.analyze_image(image_path)
    analyzer.print_results(result)
    analyzer.annotate_entities_and_save_image(
        result,
        image_path=image_path,
        entity_type="people",
        outputfile="results/people.jpg",
    )
    analyzer.annotate_entities_and_save_image(
        result,
        image_path=image_path,
        entity_type="objects",
        outputfile="results/objects.jpg",
    )
