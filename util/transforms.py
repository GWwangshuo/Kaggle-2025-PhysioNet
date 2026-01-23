import albumentations as A


def build_transforms() -> A.Compose:
    transforms = A.Compose(
        [
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 9), p=0.5),
                A.MotionBlur(blur_limit=7, p=0.5),
            ], p=0.3),

            A.OneOf([
                A.GaussNoise(p=0.5),
                A.MultiplicativeNoise(multiplier=(0.8, 1.2), elementwise=True, p=0.5),
            ], p=0.3),

            A.RandomBrightnessContrast(p=0.3),
            A.ImageCompression(quality_range=(30, 90), p=0.3),
            A.ToGray(p=0.5)
        ]
    )
    return transforms


