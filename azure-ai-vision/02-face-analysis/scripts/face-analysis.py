import configparser
import os
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential
import azure.ai.vision as sdk
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw


class FaceDetection:
    def __init__(self, ai_endpoint, ai_key, image_path: str):
        self.cv_client = sdk.VisionServiceOptions(ai_endpoint, ai_key)
        self.image_path = image_path

    @classmethod
    def from_config(cls, config_path: str, image_path: str):
        config = configparser.ConfigParser(interpolation=None)
        config.read(config_path)
        ai_endpoint = config["azure-ai-services"]["AI_SERVICE_ENDPOINT"]
        ai_key = config["azure-ai-services"]["AI_SERVICE_KEY"]
        return cls(ai_endpoint=ai_endpoint, ai_key=ai_key, image_path=image_path)

    def analyze_image(self):
        # Define an object of type image analysis options
        analysis_options = sdk.ImageAnalysisOptions()
        # Addition features (people)
        analysis_options.features = sdk.ImageAnalysisFeature.PEOPLE
        # Define vision source
        image = sdk.VisionSource(self.image_path)
        # Analyze vision source
        image_analyzer = sdk.ImageAnalyzer(self.cv_client, image, analysis_options)
        result = image_analyzer.analyze()
        return result

    def annotate_people(self, confidence_threshold=0.5):
        result = self.analyze_image()
        if result.reason == sdk.ImageAnalysisResultReason.ANALYZED:
            # Get people in the image
            if result.people is not None:
                print("\nPeople in image:")
                # Prepare image for drawing
                image = Image.open(self.image_path)
                fig = plt.figure(figsize=(image.width / 100, image.height / 100))
                plt.axis("off")
                draw = ImageDraw.Draw(image)
                color = "cyan"

                for detected_people in result.people:
                    # Draw object bounding box if confidence > 50%
                    if detected_people.confidence > confidence_threshold:
                        # Draw object bounding box
                        r = detected_people.bounding_box
                        bounding_box = ((r.x, r.y), (r.x + r.w, r.y + r.h))
                        draw.rectangle(bounding_box, outline=color, width=3)
                        # Return the confidence of the person detected
                        print(
                            f"{detected_people.bounding_box} (confidence: {detected_people.confidence * 100:.2f}%)"
                        )

                # Save annotated image
                plt.imshow(image)
                plt.tight_layout(pad=0)
                outputfile = "results/detected_people.jpg"
                fig.savefig(outputfile)
                print("Results saved in", outputfile)
            else:
                print("No, people")

        else:
            error_details = sdk.ImageAnalysisErrorDetails.from_result(result)
            print(" Analysis failed.")
            print("   Error reason: {}".format(error_details.reason))
            print("   Error code: {}".format(error_details.error_code))
            print("   Error message: {}".format(error_details.message))


if __name__ == "__main__":
    analyzer = FaceDetection.from_config(
        config_path="config.ini", image_path="images/people.jpg"
    )
    analyzer.annotate_people(confidence_threshold=0.5)
