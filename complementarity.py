import pandas as pd
from torch.utils import data
from utils.datasets import ImageNet, CIFAR10, INTEL, FashionMNIST
from utils.models_lists import imagenet_models, cifar10_models
from torch import cuda, hub, inference_mode, argmax, load
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, Grayscale
from torchvision.models import get_model
from argparse import ArgumentParser
from tqdm import tqdm

def main():
    parser = ArgumentParser()

    parser.add_argument("-D", "--dataset", help="Define which dataset models to use.", choices=['cifar10', 'imagenet','intel', 'fashionmnist'], default='cifar-10', required=True)
    parser.add_argument("-f", "--dataset-root", help="The root file path of the validation or test dataset. (e.g. For CIFAR-10 the directory containing the 'cifar-10-batches-py' folder, etc.)", required=True)
    parser.add_argument("-t", "--train", help="Define whether to use the training or test split, for datasets that require that parameter.", default=False, action="store_true")
    parser.add_argument("-w", "--weights", help="Optional. The path directory of custom weights for all the models used in the process. The files should be in '.pth' extension and named after the original CIFAR-10 model name (e.g. 'resnet20.pth'). If not set the default pretrained CIFAR-10 model weights will be used.", default=None)
    
    args = parser.parse_args()

    device = 'cuda' if cuda.is_available() else 'cpu'
    if device == 'cuda':
        cuda.empty_cache()     

    # Setup parameters
    if args.dataset == 'cifar10':
        transform = Compose([
            ToTensor(),
            Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.201)) # cifar10
        ])

        model_list = cifar10_models

        dataset = CIFAR10(root=args.dataset_root, train=args.train, transform=transform, seed=42)
        
        BATCH_SIZE=64

        dataloader = data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)        
    
    
    elif args.dataset == 'imagenet':
        transform = Compose([
            ToTensor(),
            Normalize(mean =(0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)) # imagenet
        ])

        model_list = imagenet_models

        dataset = ImageNet(root=args.dataset_root, transform=transform, seed=42)

        BATCH_SIZE=64

        dataloader = data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)


    elif args.dataset == 'intel':
        transform = Compose([
            Resize((32,32)),
            ToTensor(),    
            Normalize((0.4302, 0.4575, 0.4539), (0.2386, 0.2377, 0.2733))
        ])

        model_list = cifar10_models

        split = "test" if args.train == True else "train"

        dataset = INTEL(root=args.dataset_root, split=split, transform=transform, seed=42)

        BATCH_SIZE=64

        dataloader = data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)  


    elif args.dataset == 'fashionmnist':
        transform = Compose([  
            Resize((32,32)),
            Grayscale(num_output_channels=3),
            ToTensor(),    
            Normalize((0.2856, 0.2856, 0.2856), (0.3385, 0.3385, 0.3385))
        ])

        model_list = cifar10_models

        dataset = FashionMNIST(root=args.dataset_root, split=split, transform=transform, seed=42)

        BATCH_SIZE=64

        dataloader = data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, train=args.train)  


    all_results = {}

    print("Getting predictions from all models...")
    for model_name in model_list:
        predictions = []
        truth = []

        if args.dataset in ['cifar10', 'intel', 'fashionmnist']:
            model = hub.load("chenyaofo/pytorch-cifar-models", model=f'cifar10_{model_name}', pretrained=True).to(device)
            if args.weights is not None:
                model = load(f'./{args.weights}/{model_name}.pth', map_location="cpu", weights_only=False)

        else:
            model = get_model(model_name, weights=imagenet_models[model_name]).to(device)

        model.to(device)
        model.eval()

        with inference_mode():
            for X_test, y_test in tqdm(dataloader, desc=f'{model_name}'):
                
                X_test = X_test.to(device)

                preds = model(X_test)

                for y_pred in preds:
                    predictions.append(argmax(y_pred).item())
                
                if len(truth) < len(dataset):
                    for true in y_test:
                        truth.append(true.item())

        all_results[f'{model_name}'] = predictions

    if 'truth' not in all_results.keys():
        all_results['true'] = truth

    df = pd.DataFrame(all_results)

    print("Calculating complementarity matrix...")
    complementarity_matrix = pd.DataFrame([])
    for model1 in df.columns[:-1]:
        for model2 in df.columns[:-1]:
            volume = 0
            for i, j, k in zip(df[model1], df[model2], df['true']):
                volume += (i == k or j == k) and i != j
            complementarity_matrix.at[model1, model2] = volume / len(df)

    print("Saving results ...")
    complementarity_matrix.to_csv("./complementarity.csv", index=False, float_format='%.4f')

if __name__ == '__main__':
    main()