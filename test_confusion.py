from sklearn.metrics import confusion_matrix
from model import *
from util import *
from path import *
import torch
import itertools
import matplotlib.pyplot as plt
import warnings
import sys
import torchvision.transforms as transforms
import torchvision.models as models
from data_loader import Classify_Dataset

CV = 1
DATASET = 'flickr'
BATCH_SIZE = 32
class_names = ['Amusement','Awe','Contentment','Excitement','Anger','Disgust','Fear','Sadness']
torch.cuda.set_device(1)
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    file_name = sys.argv[1]
    metrics = torch.nn.Sigmoid()
    val_csv_file = val_csv_path.format(DATASET,CV)
    base_model = models.squeezenet1_1(pretrained=True)
    model = Classification(base_model).to(device)
    model.load_state_dict(torch.load(file_name))
    model.to(device)
    model.eval()
    val_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()])

    valset = Classify_Dataset(csv_file=val_csv_file, root_dir=img_path, transform=val_transform)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=BATCH_SIZE,shuffle=False)
    Acc = 0
    for batch_idx, data in enumerate(val_loader):
        images = data['image'].to(device)
        labels = data['annotations'].squeeze(2).to(device).float()
        with torch.no_grad():
            outputs = model(images)
        output_label = torch.argmax(metrics(outputs),1)
        target_label = torch.argmax(labels,1)
        Acc += np.sum((output_label.cpu() == target_label.cpu()).numpy())
        if batch_idx == 0:
            predict = output_label.cpu().numpy().tolist()
            target = target_label.cpu().numpy().tolist()
        else:
            predict += output_label.cpu().numpy().tolist()
            target += target_label.cpu().numpy().tolist()
        print('\r{}'.format(np.array(predict).shape),end='')
    print('\nAcc: {}'.format(Acc/len(val_loader.dataset)))
    cnf_matrix = confusion_matrix(np.array(target),np.array(predict))
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True, title='Confusion Matrix')
    plt.show()


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{:.3f}'.format(cm[i, j]),
                 horizontalalignment="center",
                 color="red" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
if __name__ == '__main__':
    warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)  
    main()