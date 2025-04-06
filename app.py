import os
from flask import Flask, render_template, request
import torch
from test import predict_lesion, EfficientNetWithMeta, load_metadata_info

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load model and metadata
class_names, localization_encoder, median_age, num_localizations = load_metadata_info()
model = EfficientNetWithMeta(model_name='efficientnet-b2', num_classes=len(class_names))
model.load_state_dict(torch.load('skin_lesion_model_b2.pth', map_location='cpu'))
model.eval()

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = None
    probabilities = None
    if request.method == 'POST':
        age = request.form.get('age', type=float)
        sex = request.form.get('sex')
        locality = request.form.get('locality')
        image = request.files.get('image')  # ← safer

        if image:
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
            image.save(image_path)

            pred_class, probs = predict_lesion(
                image_path=image_path,
                age=age or median_age,
                sex=sex,
                localization=locality,
                model=model,
                class_names=class_names,
                localization_encoder=localization_encoder,
                num_localizations=num_localizations
            )

            prediction = pred_class
            confidence = f"{max(probs)*100:.2f}%"
            
            # Create probabilities dictionary to pass to template
            probabilities = {class_name: float(prob) for class_name, prob in zip(class_names, probs)}

    return render_template('index.html', prediction=prediction, confidence=confidence, probabilities=probabilities)


if __name__ == '__main__':
    app.run(debug=True)