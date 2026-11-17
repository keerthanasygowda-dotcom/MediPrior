import gradio as gr
import torch
import torchxrayvision as xrv

model = xrv.models.DenseNet(
    weights="densenet121-res224-nih",
    apply_sigmoid=True
)
model.eval()

def analyze(image_path):
    if image_path is None:
        return "Please upload a chest X-ray."

    try:
        image = xrv.utils.load_image(image_path)
        tensor = torch.from_numpy(image).unsqueeze(0)

        with torch.no_grad():
            scores = model(tensor)[0]

        results = sorted(
            zip(model.pathologies, scores.tolist()),
            key=lambda x: x[1],
            reverse=True
        )

        output = "### Model findings\n\n"
        for name, score in results[:5]:
            output += f"- **{name}:** {score * 100:.1f}%\n"

        output += "\n---\n**Important:** Experimental AI screening only. "
        output += "These scores are not a diagnosis. A qualified "
        output += "healthcare professional must review the image."

        return output

    except Exception as error:
        return f"Analysis failed: {error}"

demo = gr.Interface(
    fn=analyze,
    inputs=gr.Image(type="filepath", label="Upload chest X-ray"),
    outputs=gr.Markdown(label="Analysis"),
    title="MediPrior — Chest X-ray Screening",
    description="AI-assisted screening to help prioritize cases. "
                "For research demonstration only; not for clinical use."
)

demo.launch()
