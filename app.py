
from fastai.vision.all import *
import gradio as gr




print(f"croix Starting...")


print(f"croix load learner...")
learner = load_learner('exported_models/20240830-artist-styleof-curated.pkl')
print(f"croix loaded learner done.")

labels = learner.dls.vocab

def predict(img):
    img = PILImage.create(img)
    predictedLabel,predictionProbIndex,probabilityArr = learner.predict(img)
    return dict(zip(labels, map(float, probabilityArr)))

input=gr.Image(height=512, width = 512)
output=gr.Label(num_top_classes=3)
examples=['generated_input/style_da_vinci.png', 'generated_input/style_picasso.png', 'generated_input/style_van_gogh.png']

print(f"interface launch starting....")
i1 = gr.Interface(fn=predict,examples=examples, inputs=input, outputs=output, title="Style Matcher").launch(share=True)

print(f"interface launch done.")