import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from tqdm import tqdm
import numpy as np
import warnings
from src.MyDataset import MyDataset
from src.BertSentimentClassifier import BertSentimentClassifier

warnings.filterwarnings("ignore")


def load_label_encoder(label_encoder_path):
    """Load the label encoder."""
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.load(label_encoder_path, allow_pickle=True)
    return label_encoder


def load_bert_model(weight_path, bert_model_name, num_classes, device):
    """Load the BERT model with pre-trained weights."""
    model = BertSentimentClassifier(bert_model_name, num_classes)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    return model.to(device)


def prepare_test_loader(test_data_path, tokenizer, max_len, batch_size):
    """Prepare the DataLoader for test data."""
    test_df = pd.read_csv(test_data_path, index_col=0)
    test_dataset = MyDataset(test_df, tokenizer, max_len, inference=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader


def predict_labels_and_probs(model, test_loader, label_encoder, device):
    """Perform inference and return predicted labels and probabilities."""
    model.eval()
    predicted_labels = []
    predicted_probs = []

    with torch.no_grad():  # Disable gradient computation
        for data in tqdm(test_loader, desc="Predicting"):
            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)

            # Get model outputs
            outputs = model(input_ids, attention_mask)

            # Predicted labels and probabilities
            preds = torch.argmax(outputs, dim=1).tolist()
            probs = torch.nn.functional.softmax(outputs, dim=1)[0].tolist()

            predicted_labels.extend(label_encoder.inverse_transform(preds))
            predicted_probs.append(probs)

    return predicted_labels, predicted_probs


def main(test_data_path, label_encoder_path, weight_path, bert_model_name, max_len=200, batch_size=1, device="cuda"):
    """Main function to run the inference."""
    # Load resources
    tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    label_encoder = load_label_encoder(label_encoder_path)
    num_classes = len(label_encoder.classes_)
    model = load_bert_model(weight_path, bert_model_name, num_classes, device)

    # Prepare test data
    test_loader = prepare_test_loader(test_data_path, tokenizer, max_len, batch_size)

    # Perform inference
    predicted_labels, predicted_probs = predict_labels_and_probs(model, test_loader, label_encoder, device)

    return predicted_labels, predicted_probs


if __name__ == "__main__":
    # Inputs
    test_data_path = 'data/incidents_val.csv'
    label_encoder_path = 'models/ST1/hazard_category/hazard_category_label_encoder.npy'
    weight_path = 'models/ST1/hazard_category/best_st1_hazard_category_large_200.pt'
    bert_model_name = "microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract"

    # Run inference
    predicted_labels, predicted_probs = main(test_data_path, label_encoder_path, weight_path, bert_model_name)

    # Print results
    print(f"Predicted Labels: {predicted_labels}")
    print(f"Predicted Probabilities: {predicted_probs}")
