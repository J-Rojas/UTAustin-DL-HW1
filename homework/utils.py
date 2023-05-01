from PIL import Image
from csv import reader
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

LABEL_NAMES = ['background', 'kart', 'pickup', 'nitro', 'bomb', 'projectile']

ImageToTensor=transforms.transforms.ToTensor()

class SuperTuxDataset(Dataset):
    
    def __init__(self, dataset_path):
        """
        Your code here
        Hint: Use the python csv library to parse labels.csv

        WARNING: Do not perform data normalization here. 
        """
        self._rows = []
        self._path = dataset_path
        with open("{}/labels.csv".format(dataset_path), newline='') as fp:
            csv_reader = reader(fp)
            # skip header
            next(csv_reader, None)
            for row in csv_reader:
                self._rows.append(row)

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, idx):
        """
        Your code here
        return a tuple: img, label
        """        
        (file, label, track) = self._rows[idx]
        return (load_image("{}/{}".format(self._path,file)), LABEL_NAMES.index(label))

def load_image(img_path):
    im = Image.open(img_path)    
    retval = ImageToTensor(im)
    im.close()
    del im
    return retval

def load_data(dataset_path, num_workers=0, batch_size=128):
    dataset = SuperTuxDataset(dataset_path)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=False)


def accuracy(outputs, labels):
    outputs_idx = outputs.max(1)[1].type_as(labels)
    return outputs_idx.eq(labels).float().mean()
