from transformers import ViTModel, ViTForImageClassification


def load_imagenet_vit(device="cuda"):
    return ViTModel.from_pretrained(
        "google/vit-base-patch16-224-in21k", output_hidden_states=True
    ).to(device)


def load_expression_vit(device="cuda"):
    return ViTForImageClassification.from_pretrained(
        "trpakov/vit-face-expression", output_hidden_states=True
    ).to(device)
