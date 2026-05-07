from flask import Flask, render_template, request
from transformers import pipeline
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import os

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

text_generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-base"
)

processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

blip_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

def generate_text(prompt):

    result = text_generator(
        prompt,
        max_length=200,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95
    )

    return result[0]['generated_text']

def generate_caption(image_path):

    raw_image = Image.open(image_path).convert('RGB')

    inputs = processor(
        raw_image,
        return_tensors="pt"
    )

    output = blip_model.generate(**inputs)

    caption = processor.decode(
        output[0],
        skip_special_tokens=True
    )

    return caption

@app.route('/', methods=['GET', 'POST'])

def home():

    ai_response = None

    image_caption = None

    uploaded_file = None

    if request.method == 'POST':

        if request.form.get("prompt"):

            prompt = request.form.get("prompt")

            ai_response = generate_text(
                "Answer this question clearly: " + prompt
            )

        if 'image' in request.files:

            image = request.files['image']

            if image.filename != "":

                image_path = os.path.join(
                    app.config['UPLOAD_FOLDER'],
                    image.filename
                )

                image.save(image_path)

                image_caption = generate_caption(
                    image_path
                )

                uploaded_file = image.filename

    return render_template(
        "index.html",
        ai_response=ai_response,
        image_caption=image_caption,
        uploaded_file=uploaded_file
    )

if __name__ == '__main__':
    app.run(debug=True, port=5004)