import pandas as pd
from torch.utils import data
from utils.datasets import ImageNetC, CIFAR10C
from utils.models_lists import imagenet_models, cifar10_models
from torch import cuda, hub, inference_mode, argmax
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision.models import get_model
from argparse import ArgumentParser
from tqdm.auto import tqdm
from utils.plotter import plot_heterogeneity_matrix

def main():
    parser = ArgumentParser()

    parser.add_argument("-d", "--dataset", help="Define which dataset models to use.", choices=['cifar10', 'imagenet'], default='cifar-10', required=True)
    parser.add_argument("-f", "--filepath", help="The path of the correspondig CIFAR-10 or ImageNet validation dataset.", required=True)
    parser.add_argument("-t", "--train", help="Only applicable to the CIFAR-10 dataset. Define whether to use the training or test dataset.", default=False, action="store_true")
    
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

        dataset = CIFAR10C(root=args.filepath, train=args.train, transform=transform, random_seed=42)
        
        BATCH_SIZE=64

        dataloader = data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)        
    elif args.dataset == 'imagenet':

        transform = Compose([
            ToTensor(),
            Normalize(mean =(0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)) # imagenet
        ])

        model_list = imagenet_models

        dataset = ImageNetC(root=args.filepath, transform=transform, random_seed=42)

        BATCH_SIZE=64

        dataloader = data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    all_results = {}

    for model_name in model_list:
        predictions = []
        truth = []

        if args.dataset == 'cifar10':
            model = hub.load("chenyaofo/pytorch-cifar-models", model=f'cifar10_{model_name}', pretrained=True).to(device)
        else:
            model = get_model(model_name, weights=imagenet_models[model_name]).to(device)

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


    print("Calculating heterogeneity matrix...")
    heterogeneity_matrix = pd.DataFrame([])
    for model1 in df.columns[:-1]:
        for model2 in df.columns[:-1]:
            volume = 0
            for i, j, k in zip(df[model1], df[model2], df['true']):
                volume += (i == k or j == k) and i != j
            heterogeneity_matrix.at[model1, model2] = volume / len(df)

    print("Saving results ...")
    plot_heterogeneity_matrix(heterogeneity_matrix)

if __name__ == '__main__':
    main()