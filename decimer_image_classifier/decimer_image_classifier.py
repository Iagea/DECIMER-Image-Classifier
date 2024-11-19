import os
from copy import copy
from PIL import Image
import tensorflow as tf
import tensorflow.keras as keras


class DecimerImageClassifier:
    """Class that wraps up the functionalities of the image classifier"""

    def __init__(self):
        # Establish GPU growth, load default GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        gpus = tf.config.experimental.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        # Load model
        model_path = os.path.join(os.path.split(__file__)[0], "model")
        self.model = keras.models.load_model(model_path)

    def is_chemical_structure(self, img=False, threshold: float = 0.000089) -> bool:
        """
        This function determines whether or not a given image (given as
        PIL.Image or as a path of an image (str)) is a chemical structure
        depiction.

        Args:
            img (PIL.Image)/ img (str): Image object or Path of image that is supposed to get classified
            # img or img path needs to be specified!
            threshold (float): Threshold for classification

        Returns:
            Result (bool): the image predicted score.
        """
        score = self.get_classifier_score(img)
        if score <= threshold:
            return True
        else:
            return False

    def get_classifier_score(
        self,
        img=False,
    ) -> float:
        """
        Function to compute the classifier score for a particular image.

        Args:
            img (PIL.Image)/ img (str): Image object or Path of image that is supposed to get classified
            # img or img path needs to be specified!

        Returns:
            score (float): the image predicted score.
        """
        if type(img) == str:
            img = Image.open(img)
        img = self._get_resized_grayscale_image(img)
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        img_array = keras.applications.efficientnet.preprocess_input(img_array)
        predictions = self.model.predict(img_array)
        score = tf.nn.sigmoid(predictions[0])
        return score.numpy()[0]

    def _get_resized_grayscale_image(
        self, img: Image, desired_size: int = 224
    ) -> Image:
        """
        This function takes a PIL.Image object, converts it to grayscale and
        resizes it to a square image of length/height a given desired_size.
        It returns the resized grayscale image.

        Args:
            img (Image): PIL.Image object
            desired_size (int, optional): Desired image height/length.
                                          Defaults to 224.

        Returns:
            Image: Resized grayscale image
        """
        grayscale_image = self._get_RGB_grayscale_image(img)
        resized_grayscale_image = self._get_resized_image(grayscale_image, desired_size)
        return resized_grayscale_image

    def _get_resized_image(self, img: Image, desired_size: int = 224) -> Image:
        """
        This function takes a PIL.Image object, resizes it to a
        square image of length/height a given desired_size.
        It returns the resized grayscale image.

        Args:
            img (Image): PIL.Image object (mode: "L" (grayscale))
            desired_size (int, optional): Desired image height/length.
                                          Defaults to 224.

        Returns:
            Image: Resized grayscale image
        """
        old_size = img.size
        if old_size[0] or old_size[1] != desired_size:
            ratio = float(desired_size) / max(old_size)
            new_size = tuple([int(x * ratio) for x in old_size])
            img = img.resize(new_size, Image.LANCZOS)
        resized_image = Image.new("RGB", (desired_size, desired_size), "white")
        resized_image.paste(
            img, ((desired_size - new_size[0]) // 2, (desired_size - new_size[1]) // 2)
        )
        return resized_image

    def _get_RGB_grayscale_image(
        self, img: Image, save_modified_rgba: bool = False
    ) -> Image:
        """
        This function takes a PIL.Image object, and returns the
        same Image object as a grayscale RGB image (no colours,
        but three channels).


        Args:
            img (Image): PIL.Image object (mode: "RGB")

        Returns:
            Image: Grayscale image
        """
        if img.mode == "RGBA":
            orig_img = copy(img)
            orig_img.load()
            img = Image.new("RGB", orig_img.size, (255, 255, 255))
            img.paste(orig_img, mask=orig_img.split()[3])
            if save_modified_rgba:
                img.save("caffeine_mod.png")
        img = img.convert("L").convert("RGB")
        return img
