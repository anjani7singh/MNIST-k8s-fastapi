from PIL import Image
import torch
from torch import load
from torchvision import transforms
from model import MNISTClassifier


class mnist_classifier():
    def __init__(self):
        # Load the saved model:
        self.device = "cpu"
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classifier = MNISTClassifier()
        with open('./model/model_state.pt', 'rb') as f:
            self.classifier.load_state_dict(load(f, map_location=torch.device("cpu")))
        print(next(self.classifier.parameters()).is_cuda)

    def predict(self,img):
        img_transform = transforms.Compose([transforms.ToTensor()])
        img_tensor = img_transform(img).unsqueeze(0)
        output = self.classifier(img_tensor)
        predicted_label = torch.argmax(output)
        return predicted_label.item()
        

if __name__=='__main__':
    img = Image.open('image.jpg')
    classifier = mnist_classifier()
    output = classifier.predict(img)
    print(output)
