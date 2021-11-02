import albumentations as A
import warnings

class Augmentor:
    """Image augmentation class for Computer Vision classification problems"""

    def __init__(self, t_type="soft"):
        self.type = t_type
        self.transform = self.get_transform()

    def get_transform(self):
        """Augment images for desired change level (soft-medium-hard) """
        if self.type == "soft":
            return A.Compose(
                [
                    A.RGBShift(r_shift_limit=(-30, 10), g_shift_limit=(-30, 10), b_shift_limit=(-30, 10), p=1.0),
                    A.OneOf([
                        A.RandomGamma(gamma_limit=(80, 120), eps=1e-07, p=0.6),
                        A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.3, 0.3),
                                                   brightness_by_max=False, p=0.6)
                    ],
                        p=1.0),
                    A.OneOf(
                        [
                            A.Blur(blur_limit=(3, 5), p=0.6),
                            A.MedianBlur(blur_limit=5, p=0.6),
                            A.GaussianBlur(blur_limit=(3, 5), sigma_limit=0, p=0.6)
                        ],
                        p=1.0),
                    A.OneOf(
                        [
                            A.MultiplicativeNoise(multiplier=(0.70, 1.50), per_channel=False, elementwise=False, p=0.6),
                            A.GaussNoise(var_limit=(40.0, 80.0), p=0.6)
                        ],
                        p=1.0),

                ]
            )

        elif self.type == "medium":
            return A.Compose(
                [
                    A.RGBShift(r_shift_limit=(-40, 20), g_shift_limit=(-40, 20), b_shift_limit=(-40, 20), p=1.0),
                    A.OneOf([
                        A.RandomGamma(gamma_limit=(80, 120), eps=1e-07, p=0.8),
                        A.RandomBrightnessContrast(brightness_limit=(-0.3, 0.3), contrast_limit=(-0.3, 0.3),
                                                   brightness_by_max=False, p=0.8)
                    ],
                        p=1.0),
                    A.SomeOf(
                        [
                            A.Blur(blur_limit=(3, 5), p=0.6),
                            A.MedianBlur(blur_limit=5, p=0.6),
                            A.GaussianBlur(blur_limit=(3, 5), sigma_limit=0, p=0.6)
                        ],
                        n=2,
                        p=1.0),
                    A.OneOf(
                        [
                            A.MultiplicativeNoise(multiplier=(0.50, 1.80), per_channel=False, elementwise=False, p=0.8),
                            A.GaussNoise(var_limit=(50.0, 90.0), p=0.8)
                        ],
                        p=1.0)

                ]
            )
        elif self.type == "hard":
            return A.Compose(
                [
                    A.RGBShift(r_shift_limit=(-40, 20), g_shift_limit=(-40, 20), b_shift_limit=(-40, 20), p=1.0),
                    A.OneOf([
                        A.RandomGamma(gamma_limit=(80, 120), eps=1e-07, p=0.9),
                        A.RandomBrightnessContrast(brightness_limit=(-0.3, 0.3), contrast_limit=(-0.3, 0.3),
                                                   brightness_by_max=False, p=0.9)
                    ],
                        p=1.0),
                    A.SomeOf(
                        [
                            A.Blur(blur_limit=(3, 7), p=0.7),
                            A.MedianBlur(blur_limit=7, p=0.7),
                            A.GaussianBlur(blur_limit=(3, 7), sigma_limit=0, p=0.7)
                        ],
                        n=2,
                        p=1.0),
                    A.MultiplicativeNoise(multiplier=(0.50, 1.80), per_channel=False, elementwise=False, p=0.9),
                    A.GaussNoise(var_limit=(90.0, 140.0), p=0.9)
                ]
            )
        else:
            warnings.warn(
                "There is not any transformation on image")
            return A.Compose([A.RGBShift(r_shift_limit=0, g_shift_limit=0, b_shift_limit=0, p=0.0)])
    def transform_image(self,image)->dict:
        """Augment images with corresponding hardness level transformations"""
        return self.transform(image=image)
