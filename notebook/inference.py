"""Inference from the all-models export. Requires its complete models/ and shared/ folders."""
from pathlib import Path
import sys, json
import numpy as np
import joblib
import tensorflow as tf
from PIL import Image
from sklearn.preprocessing import normalize

@tf.keras.utils.register_keras_serializable(package="DeepWeeds")
class ResNetPreprocess(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.keras.applications.resnet50.preprocess_input(inputs)

class Predictor:
    def __init__(self, export_dir, model_id):
        self.root = Path(export_dir)
        self.directory = self.root / "models" / model_id
        self.config = json.loads((self.directory / "metadata.json").read_text())
        self.classes = self.config["classes"]
        path = self.directory / self.config["artifact"]
        self.classifier = (tf.keras.models.load_model(path, compile=False)
                           if path.suffix == ".keras" else joblib.load(path))
        self.scaler = joblib.load(self.directory / self.config["scaler"]) if self.config.get("scaler") else None
        self.extractor = None
        extractor_name = self.config.get("extractor")
        if extractor_name == "dinov2":
            import os
            os.environ.setdefault("XFORMERS_DISABLED", "1")
            import torch
            from torchvision import transforms
            sys.path.insert(0, str(self.root / "shared/dinov2_source"))
            from dinov2.hub.backbones import dinov2_vits14
            self.extractor = dinov2_vits14(pretrained=False).eval()
            weights = torch.load(self.root / "shared/dinov2_vits14_pretrain.pth",
                                 map_location="cpu", weights_only=True)
            self.extractor.load_state_dict(weights)
            self.transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
        elif extractor_name:
            self.extractor = tf.keras.models.load_model(self.root / "shared" / (extractor_name + ".keras"), compile=False)

    def predict_classifier_input(self, array):
        """Already prepared classifier inputs."""
        array = np.asarray(array)
        if self.config["artifact"].endswith(".joblib"):
            return self.classifier.predict_proba(array)
        return self.classifier(array, training=False).numpy()

    def predict_features(self, features):
        features = np.asarray(features)
        if self.config.get("l2"):
            features = normalize(features, norm="l2")
        if self.scaler is not None:
            features = self.scaler.transform(features)
        return self.predict_classifier_input(features)

    def predict_array(self, rgb):
        """One decoded RGB uint8 image -> probabilities in self.classes order."""
        rgb = np.asarray(rgb, dtype=np.uint8)
        family = self.config["family"]
        size = self.config["input_size"]
        if family == "features":
            name = self.config["extractor"]
            if name == "dinov2":
                import torch
                x = self.transform(Image.fromarray(rgb)).unsqueeze(0)
                with torch.inference_mode():
                    features = self.extractor(x).numpy()
            else:
                x = tf.image.resize(rgb, (size, size))[None]
                if name == "resnet50":
                    x = tf.keras.applications.resnet50.preprocess_input(x)
                elif name == "inceptionv3":
                    x = tf.keras.applications.inception_v3.preprocess_input(x)
                features = self.extractor(x, training=False).numpy()
            return self.predict_features(features)[0]
        x = tf.image.resize(rgb, (size, size)).numpy()[None]
        if family == "pixels":
            x = x / 255.
            if self.config.get("flatten"):
                return self.predict_classifier_input(x.reshape(1, -1))[0]
            if self.config.get("tta"):
                views = []
                for k in range(4):
                    rotated = np.rot90(x, k, axes=(1, 2)).copy()
                    views.extend([self.predict_classifier_input(rotated),
                                  self.predict_classifier_input(rotated[:, :, ::-1].copy())])
                return np.mean(views, axis=0)[0]
        return self.predict_classifier_input(x)[0]

    def predict_file(self, path):
        rgb = np.asarray(Image.open(path).convert("RGB"))
        probabilities = self.predict_array(rgb)
        return {"class": self.classes[int(probabilities.argmax())],
                "probabilities": dict(zip(self.classes, probabilities.astype(float).tolist()))}
