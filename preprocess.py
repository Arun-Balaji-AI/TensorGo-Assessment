import cv2
import os
from torchvision import transforms
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim



model = SuperResolutionDiffusionModel()
model.load_state_dict(torch.load('super_res_diffusion_model.pth'))  # Load pre-trained weights
model.eval()
def extract_frames(video_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(output_dir, f"frame_{frame_count:04d}.png")
        cv2.imwrite(frame_path, frame)
        frame_count += 1

    cap.release()
    return frame_count

def enhance_frame(frame_path, model, device):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((1080, 1920)),  # Resize to HD
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    img = Image.open(frame_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        enhanced_tensor = model(img_tensor)
    
    enhanced_img = transforms.ToPILImage()(enhanced_tensor.squeeze(0).cpu().clamp(0, 1))
    return enhanced_img

def enhance_frames(input_dir, output_dir, model, device):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    frame_files = sorted(os.listdir(input_dir))
    for frame_file in frame_files:
        frame_path = os.path.join(input_dir, frame_file)
        enhanced_img = enhance_frame(frame_path, model, device)
        enhanced_frame_path = os.path.join(output_dir, frame_file)
        enhanced_img.save(enhanced_frame_path)


if __name__ == "__main__":
    video_path = "path_to_your_sd_video.mp4"  # Replace with your video file path
    extracted_frames_dir = "extracted_frames"
    enhanced_frames_dir = "enhanced_frames"
    output_video_path = "output_hd_video.mp4"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Step 1: Extract frames from the video
    frame_count = extract_frames(video_path, extracted_frames_dir)
    
    # Step 2: Enhance the frames
    enhance_frames(extracted_frames_dir, enhanced_frames_dir, model, device)
    
    # Step 3: Reconstruct the video from enhanced frames
    reconstruct_video(enhanced_frames_dir, output_video_path)

