import base64
import io
from typing import Any, Callable, Dict, List

import torch
from flask import Flask, jsonify, request
from PIL import Image
from torchvision import transforms
from transformers import BertTokenizerFast

from utils.vocab import Vocab


def load_bert_resnet152_mul_model(device='cpu') -> Callable[[Image.Image, str], List[str]]:
    from models.bert_resnet152_mul_model import VQANet

    device = torch.device(device)

    model_path = './models/bert_resnet152_mul_'
    answers = Vocab.load(model_path + 'answers.json')
    weights = torch.load(model_path + 'weights.pth', map_location=device)
    model = VQANet(len(answers))
    model.load_state_dict(weights)
    model.to(device)
    model.eval()

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    image_transform = transforms.Compose(
        [
            transforms.Resize(int(224 / 0.875)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    def generate_batch(data_batch):
        images, questions = zip(*data_batch)
        images = torch.stack(images, 0).to(device)
        question_inputs = tokenizer(list(questions), padding=True)
        question_inputs = {k: torch.tensor(v, dtype=torch.long).to(device) for k, v in question_inputs.items()}
        return images, question_inputs

    def predict(image: Image.Image, question: str) -> List[str]:
        with torch.no_grad():
            # preprocess
            image = image_transform(image)
            batch = generate_batch([[image, question]])
            # predict
            output = model(*batch)
            output = torch.argmax(output).item()
            answer = answers.itos[int(output)]
            return [answer]

    return predict


app = Flask(__name__)

model_loaders = {
    'bert_resnet152_mul': (load_bert_resnet152_mul_model, {}),
}

models = {}


@ app.route('/api/vqa', methods=['POST'])
def vqa():
    request_dict: Dict[str, Any] = request.get_json()  # type: ignore

    # Read image
    try:
        if 'image' not in request_dict:
            raise ValueError('`image` not found in the request form')
        image_uri = request_dict['image']
        _, image_b64 = image_uri.split(',', 1)
        image_bytes = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    except Exception as e:
        msg = f'failed to read image due to error: {e}'
        app.logger.error(msg)
        return jsonify({'message': msg}), 400

    # Read question
    question = request_dict.get('question', 'what is this?')

    # Get model
    model_name = request_dict.get('model', 'bert_resnet152_mul')
    if model_name in model_loaders:
        if model_name not in models:
            loader, args = model_loaders[model_name]
            models[model_name] = loader(**args)
        model = models[model_name]
    else:
        msg = f'unknown model: {model_name}'
        app.logger.error(msg)
        return jsonify({'message': msg}), 404

    # Answer question
    answer = model(image, question)

    app.logger.info(f'Q: {question} A: {answer[-1]}')
    return jsonify({
        'question': question,
        'answer': answer
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
