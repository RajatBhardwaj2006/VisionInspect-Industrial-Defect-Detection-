from torchvision import transforms


def get_train_transform(image_size=256):
    """
    Preprocessing used for normal MVTec training images.
    """

    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])