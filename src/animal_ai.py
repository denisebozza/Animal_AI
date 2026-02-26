'''
Codice principale
'''
# src/app_animals_gradio.py
import os
import torch
import gradio as gr
import numpy as np
from PIL import Image

from src.ai import AnimalRecognAI
from src.animal_recogn import training_saving 

MODEL_PATH = "./persistent_data/animal_model.pth"   # cambia se lo salvi altrove
CSS = """
body { background-color: #121212; color: #eee; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
.gradio-container { max-width: 900px; margin: auto; padding: 20px; border-radius: 15px; background-color: #1e1e1e; box-shadow: 0 0 20px #222; }
h1, h2 { color: #FFA500; text-align: center; }
"""
DATASET_FOLDER = "./data/animals"
DEFAULT_EPOCHS = 5

def load_ai(model_path: str = MODEL_PATH) -> AnimalRecognAI:
    if not os.path.exists(model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        # allena e salva il modello
        training_saving(
            download_dataset=False,   # metti True solo se vuoi che scarichi da kaggle qui
            epochs=DEFAULT_EPOCHS,
            save_path=model_path,
            dataset_path=DATASET_FOLDER,
        )
    return AnimalRecognAI(model_path=model_path, rgb=True)

AI_INSTANCE = None

def predict(image: np.ndarray, topk: int = 3):
    """
    image arriva da Gradio come numpy array (H,W,3) in RGB.
    """
    global AI_INSTANCE
    if AI_INSTANCE is None:
        AI_INSTANCE = load_ai(MODEL_PATH)

    if image is None:
        return "Nessuna immagine", {}

    # numpy -> PIL
    pil = Image.fromarray(image.astype(np.uint8))

    # stessa trasformazione della classe
    x = AI_INSTANCE.transform(pil).unsqueeze(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    AI_INSTANCE.model.to(device)
    x = x.to(device)

    AI_INSTANCE.model.eval()
    with torch.no_grad():
        logits = AI_INSTANCE.model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0)

    # top-k
    k = max(1, min(int(topk), probs.numel()))
    top_probs, top_idx = torch.topk(probs, k=k)

    # mapping indice->nome (usa la tua mappa fissa a 15 classi)
    idx_to_name = AI_INSTANCE.CLASS_MAP

    pred_name = idx_to_name[int(top_idx[0].item())]
    pred_prob = float(top_probs[0].item())

    # gradio Label vuole dict {classe: prob}
    scores = {idx_to_name[int(i.item())]: float(p.item()) for p, i in zip(top_probs, top_idx)}

    return f"{pred_name} ({pred_prob*100:.1f}%)", scores


def main():
    with gr.Blocks(css=CSS) as demo:
        gr.Markdown("# Animal Recognition 🐾")
        gr.Markdown("Carica un’immagine e il modello predice la classe (Beetle, Butterfly, Cat, ... Zebra).")

        with gr.Row():
            inp = gr.Image(type="numpy", label="Immagine (upload)", height=320)

        with gr.Row():
            topk = gr.Slider(minimum=1, maximum=5, value=3, step=1, label="Top-K")

        with gr.Row():
            out_text = gr.Textbox(label="Predizione")
        with gr.Row():
            out_label = gr.Label(num_top_classes=5, label="Probabilità (Top-K)")

        btn = gr.Button("Predici")
        btn.click(fn=predict, inputs=[inp, topk], outputs=[out_text, out_label])

    # per Docker/remote
    demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()