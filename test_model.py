import torch
from model import AODnet
import numpy as np
from PIL import Image
import cv2

# Load model safely
with torch.serialization.safe_globals([AODnet]):
    model = torch.load("AOD-Net_epoch10.pth", map_location="cpu", weights_only=False)

model.eval()
print("Model loaded successfully!")

# Dehaze function
def dehaze_dl(img_pil):
    img = np.array(img_pil)
    tensor = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).float()/255.0
    with torch.no_grad():
        out = model(tensor)
    out_img = out.squeeze(0).permute(1,2,0).numpy()
    return np.clip(out_img*255, 0, 255).astype(np.uint8)

# Load hazy image
image = Image.open(r"C:\Users\divya\OneDrive\Desktop\DehazeIt\hazy1.jpg")

# Dehaze it
result = dehaze_dl(image)

# Save dehazed result
cv2.imwrite("dehazed_test.png", cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
print("Dehazed image saved as dehazed_test.png")