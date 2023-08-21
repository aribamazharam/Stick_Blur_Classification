
from stickblurclassifier_refactored import train_model, classify_video, transforms

def main():
    # Example usage:

    # 1. Train the model
    train_path = "/path/to/train_data"
    valid_path = "/path/to/valid_data"
    model_save_path = "/path/to/save/model.pth"
    trained_model, train_accuracies, val_accuracies = train_model(train_path, valid_path, num_epochs=5, model_save_path=model_save_path)

    # Print training and validation accuracies for reference
    print("Training Accuracies:", train_accuracies)
    print("Validation Accuracies:", val_accuracies)

    # 2. Classify a video
    video_path = "/path/to/video.mp4"
    blur_save_dir = "/path/to/save/blur_frames"
    no_blur_save_dir = "/path/to/save/no_blur_frames"

    # Use the same transform as used during training
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    classify_video(trained_model, video_path, blur_save_dir, no_blur_save_dir, transform)

if __name__ == "__main__":
    main()
