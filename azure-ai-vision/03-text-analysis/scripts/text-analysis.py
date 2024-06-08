import configparser
import os

from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw


# Class definition
class AzureImageAnalyzer:
    def __init__(self, ai_endpoint, ai_key, image_path: str):
        self.client = ImageAnalysisClient(
            endpoint=ai_endpoint, credential=AzureKeyCredential(ai_key)
        )
        self.image_path = image_path

    @classmethod
    def from_config(cls, config_path: str, image_path: str):
        config = configparser.ConfigParser(interpolation=None)
        config.read(config_path)
        ai_endpoint = config["azure-ai-services"]["AI_SERVICE_ENDPOINT"]
        ai_key = config["azure-ai-services"]["AI_SERVICE_KEY"]
        return cls(ai_endpoint=ai_endpoint, ai_key=ai_key, image_path=image_path)

    def analyze_image(self):
        with open(self.image_path, "rb") as f:
            image_data = f.read()
        try:
            # Get result with specified features to be retrieved
            result = self.client.analyze(
                image_data=image_data,
                visual_features=[
                    VisualFeatures.CAPTION,
                    VisualFeatures.PEOPLE,
                    VisualFeatures.READ,
                    VisualFeatures.TAGS,
                    VisualFeatures.OBJECTS,
                ],
            )
            return result
        except Exception as e:
            print(f"Error analyzing image: {e}")
            return None

    def print_results(self):
        result = self.analyze_image()
        if result is None:
            print("No result to display")
            return

        # Print caption
        if result.caption is not None:
            print("\nCaption:")
            print(
                f"'{result.caption.text}' (confidence: {result.caption.confidence * 100:.2f}%)"
            )

        # Print people
        if len(result.people["values"]) > 1:
            for person in result.people["values"]:
                if person["confidence"] > 0.5:
                    print(f" person: {person}")

        # Print objects
        for detected_object in result.objects.list:
            print("\nObjects:")
            print(
                f"{detected_object.tags[0].name} (confidence: {detected_object.tags[0].confidence:.2f}%)"
            )

    def annotate_texts(
        self,
        outputfile: str = "results/text.jpg",
        drawLinePolygon=False,
        drawWordPolygon=True,
    ):
        # Fetch the result object
        result = self.analyze_image()
        if result.read is not None:
            # Prepare image for drawing
            image = Image.open(self.image_path)
            fig = plt.figure(figsize=(image.width / 100, image.height / 100))
            plt.axis("off")
            draw = ImageDraw.Draw(image)
            color = "cyan"

            # Lines in the text
            print("\nText:")
            for line in result.read.blocks[0].lines:
                print(line.text)
                r = line.bounding_polygon
                bounding_polygon = (
                    (r[0].x, r[0].y),
                    (r[1].x, r[1].y),
                    (r[2].x, r[2].y),
                    (r[3].x, r[3].y),
                )
                print(f"   Bounding Polygon: {bounding_polygon}")

                # Draw line bounding polygon
                if drawLinePolygon:
                    draw.polygon(bounding_polygon, outline=color, width=3)

                # words in each line
                for word in line.words:
                    r = word.bounding_polygon
                    bounding_polygon = (
                        (r[0].x, r[0].y),
                        (r[1].x, r[1].y),
                        (r[2].x, r[2].y),
                        (r[3].x, r[3].y),
                    )
                    print(
                        f"    Word: '{word.text}', Bounding Polygon: {bounding_polygon}, Confidence: {word.confidence:.4f}"
                    )

                    # Draw word bounding polygon
                    if drawWordPolygon:
                        draw.polygon(bounding_polygon, outline=color, width=3)

                # Save image
            plt.imshow(image)
            plt.tight_layout(pad=0)
            fig.savefig(outputfile)
            print("\n  Results saved in", outputfile)


if __name__ == "__main__":
    analyzer = AzureImageAnalyzer.from_config(
        config_path="config.ini", image_path="images/statue.jpg"
    )
    analyzer.print_results()
    analyzer.annotate_texts(outputfile="results/text.jpg")
