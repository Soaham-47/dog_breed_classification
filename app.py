import gradio as gr
import torch
import torchvision.transforms as transforms
from PIL import Image
from model_utils import create_res_model

# 1. Load your breed translation list from the local text file
with open("class_names.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# 2. Re-instantiate the architecture and load your downloaded weights
model, _, _ = create_res_model(output_shape=len(class_names))
model.load_state_dict(torch.load("8June_dog_breed_resnet_50.pth", map_location="cpu"))
model.eval()

# 3. Define the exact same image transforms you used for your validation loader
inference_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 4. Define the prediction execution loop
def predict_dog_breed(inp_image):
    if inp_image is None:
        return "Please upload an image to start."
        
    # Convert incoming array interface to PIL RGB format
    img = Image.fromarray(inp_image.astype('uint8'), 'RGB')
    img_tensor = inference_transforms(img).unsqueeze(0)
    
    # Compute confidence probabilities without tracking gradients
    with torch.no_grad():
        logits = model(img_tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
    
    # Extract the top 5 highest guesses to build dynamic progress bars in Gradio
    topk_probs, topk_indices = torch.topk(probabilities, k=5)
    
    results = {}
    for i in range(5):
        breed_label = class_names[topk_indices[i].item()].replace('_', ' ').title()
        results[breed_label] = float(topk_probs[i].item())
        
    return results

# 5. Build and configure the user interface window
demo = gr.Interface(
    fn=predict_dog_breed,
    inputs=gr.Image(label="📸 Upload Dog Image Here"),
    outputs=gr.Label(num_top_classes=5, label="🎯 Top 5 Identified Breeds"),
    title="🐶 Fine-Grained Dog Breed Classifier",
    description="Drop a picture of any dog below, and our custom fine-tuned ResNet50 vision model will identify its breed with granular confidence metrics!",
)

# Launch application
if __name__ == "__main__":
    demo.launch()