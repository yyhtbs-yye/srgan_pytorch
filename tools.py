import os
import cv2
import numpy as np
import torch
from torchvision.utils import save_image
from torchvision.transforms.functional import to_tensor, to_pil_image

def evaluate(G, checkpoint_dir, save_dir, hr_img_path, device):
    ###====================== PRE-LOAD DATA ===========================###
    # Assuming valid_hr_imgs is a list of file paths for simplicity
    valid_hr_imgs = [cv2.imread(path) for path in hr_img_path]

    ###========================LOAD WEIGHTS ============================###
    # Make sure to replace 'g.npz' with the actual PyTorch model filename, like 'g_epoch.pth'
    G.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'g.pth'), map_location=device))
    G.eval()

    # Select an image for demonstration (ensure your dataset has this index)
    imid = 0  # Adjust the index based on your dataset
    valid_hr_img = valid_hr_imgs[imid]
    hr_size1 = valid_hr_img.shape[:2]

    # Downscale the image to create a low-resolution version
    valid_lr_img = cv2.resize(valid_hr_img, (hr_size1[1] // 4, hr_size1[0] // 4))

    # Convert images to tensor
    valid_lr_img_tensor = to_tensor(valid_lr_img).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        out = G(valid_lr_img_tensor).cpu().clamp(0, 1)

    # Process and save the output
    out = to_pil_image(out.squeeze(0))
    out.save(os.path.join(save_dir, 'valid_gen.png'))
    
    # Save the bicubic upsampled image
    out_bicu = cv2.resize(valid_lr_img, (hr_size1[1], hr_size1[0]), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(os.path.join(save_dir, 'valid_hr_cubic.png'), out_bicu)

    # Save the LR and HR images for reference
    cv2.imwrite(os.path.join(save_dir, 'valid_lr.png'), valid_lr_img)
    cv2.imwrite(os.path.join(save_dir, 'valid_hr.png'), valid_hr_img)

    print(f"LR size: {valid_lr_img.shape[:2]} / generated HR size: {out.size}")
    print("[*] Images saved.")
