from sklearn.metrics import classification_report
from torch import cuda, load
from torch.nn import Module
from os import system
from pandas import DataFrame
from yaml import safe_load, YAMLError
from utils.models_lists import cifar10_models, imagenet_models
from argparse import ArgumentParser

def main():
    #TODO required arguments when providing yml file should be ignored
    deciding_args_parser = ArgumentParser(add_help=False)
    deciding_args_parser.add_argument("-y", "--yml-file", help="Use .yml configuration file instead of cli arguments. In this case you must provide the location of the .yml file and the rest of the arguments are ignored.", type=str, required=False, default=None)
    deciding_args, _ = deciding_args_parser.parse_known_args()

    parser = ArgumentParser(parents=[deciding_args_parser])
    # parser.add_argument("-y", "--yml-file", help="Use .yml configuration file instead of cli arguments. In this case you must provide the location of the .yml file and the rest of the arguments are ignored.", required=False, default=None)
    parser.add_argument("-D", "--dataset", help="The dataset to use.", choices=['cifar10', 'imagenet', 'intel', 'fashionmnist'], required=deciding_args.yml_file is None, default=None)
    parser.add_argument("-m1", "--model-1", help="The first model name, required. It must be included in the provided lists of available models.", required=deciding_args.yml_file is None)   
    parser.add_argument("-m2", "--model-2", help="The second model. It must be included in the provided lists of available models.", default=None, required=False)
    parser.add_argument("-w1", "--weights-1", help="Optional. A file path to the first model's weights file.", default=None, required=False)
    parser.add_argument("-w2", "--weights-2", help="Optional. A file path to the second model's weights file.", default=None, required=False)
    parser.add_argument("-f", "--dataset-path", help="The file path of the validation or test dataset.", required=deciding_args.yml_file is None)
    parser.add_argument("-s", "--scorefn", help="Score function to use.", choices=['maxp', 'difference', 'entropy', 'oracle'], default=None, required=False)
    parser.add_argument("-t", "--threshold", help="The threshold value to use for the threshold check. (λ parameter)", type=float, required=False)
    parser.add_argument("-p", "--postcheck", help="Enable post check. Default is false.", default=False, action="store_true")
    parser.add_argument("-m", "--memory", help="Enable memory component. Default is None.", choices=['dhash', 'invariants'], default=None)
    parser.add_argument("-d", "--duplicates", help="Set the percentage of the original training set for duplication. Default is 0 (No duplicates). Range [0-1]", type=float, default=0)
    parser.add_argument("-r", "--rotations", help="If set the duplicated samples will be randomly rotated or mirrored.", action='store_true', default=False)
    parser.add_argument("-rp", "--root-password", help="If provided the password will be used to command the computer to shutdown after finishing.", required=False, default=None)

    args = parser.parse_args()

    if deciding_args.yml_file is None:        
        DATASET = args.dataset
        MODEL_1 = args.model_1
        MODEL_2 = args.model_2
        SCORE_FN = args.scorefn
        THRESHOLD = args.threshold
        POSTCHECK = args.postcheck
        WEIGHTS_1 = args.weights_1
        WEIGHTS_2 = args.weights_2
        DATASET_FILEPATH = args.dataset_path
        MEMORY = True if args.memory is not None else False
        MEMORY_METHOD = args.memory
        DUPLICATES = args.duplicates
        ROTATIONS = args.rotations
        ROOT_PASSWORD = args.root_password
    else:
        with open("./config.yml") as stream:
            try:
                config = safe_load(stream)
            except YAMLError as exc:
                print(exc)

        DATASET = config['dataset']
        MODEL_1 = config['first_model']['name']
        MODEL_2 = config['second_model']['name'] if config['second_model']['enable'] == True else None
        SCORE_FN = config['second_model']['scorefn']
        THRESHOLD = config['second_model']['threshold']
        POSTCHECK = config['second_model']['postcheck']
        WEIGHTS_1 = config['first_model']['weights_file']
        WEIGHTS_2 = config['second_model']['weights_file']
        DATASET_FILEPATH = config['dataset_path']
        MEMORY = config['memory']['enable']
        MEMORY_METHOD = config['memory']['method']
        DUPLICATES = config['duplicates']
        ROTATIONS = config['transforms']
        ROOT_PASSWORD = config['root_password']

    print(DATASET)
    device = 'cuda' if cuda.is_available() else 'cpu'

    if MODEL_2 and not SCORE_FN and THRESHOLD == None:
        raise ValueError("Score function and/or threshold value must be provided if model_2 is used.")

    if DATASET == 'cifar10':        
        from torch import hub
        from utils.datasets import CIFAR10
        from torchvision.transforms import Compose, ToTensor, Normalize

        dataset = "CIFAR-10"

        transform = Compose([
                ToTensor(),
                Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.201)) # CIFAR10
            ])
        
        print("Loading dataset...")

        valset = CIFAR10(root=DATASET_FILEPATH, train=False, return_numpy=MEMORY, transform=transform, duplicate_ratio=DUPLICATES, rotations=ROTATIONS, seed=42)

        print("Loading models...")
        model_a = hub.load("chenyaofo/pytorch-cifar-models", model=f'cifar10_{MODEL_1}', pretrained=True).to(device)

        if MODEL_2 != None:
            if not MODEL_2 in cifar10_models:
                raise ValueError("Second model must be from the same CIFAR10 dataset.")

            model_b = hub.load("chenyaofo/pytorch-cifar-models", model=f'cifar10_{MODEL_2}', pretrained=True).to(device)

    
    elif DATASET == 'imagenet':
        from torchvision.models import get_model
        from utils.datasets import ImageNet
        from torchvision.transforms import Compose, ToTensor, Normalize

        dataset = "ImageNet"

        transform = Compose([
            ToTensor(),
            Normalize(mean =(0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)), #ImageNet
        ])
        print("Loading dataset...")

        valset = ImageNet(root=DATASET_FILEPATH, split='val', transform=transform, return_numpy=MEMORY, duplicate_ratio=DUPLICATES, rotations=ROTATIONS, seed=42)
        

        print("Loading models...")
        model_a = get_model(MODEL_1, weights=imagenet_models[MODEL_1])

        if MODEL_2:
            if not MODEL_2 in imagenet_models.keys():
                raise ValueError("Second model must be from the same ImageNet dataset.")

            model_b = get_model(MODEL_2, weights=imagenet_models[MODEL_2])

    
    elif DATASET == 'intel':
        from torch import hub
        from utils.datasets import INTEL
        from torchvision.transforms import Compose, ToTensor, Normalize, Resize

        transform = Compose([
            Resize((32,32)),
            ToTensor(),    
            Normalize((0.4302, 0.4575, 0.4539), (0.2386, 0.2377, 0.2733))
        ])

        dataset = "INTEL"

        valset = INTEL(root=DATASET_FILEPATH, split='test', transform=transform, return_numpy=MEMORY, duplicate_ratio=DUPLICATES, rotations=ROTATIONS, seed=42)        

        model_a = hub.load("chenyaofo/pytorch-cifar-models", model=f'cifar10_{MODEL_1}', pretrained=False)     

        if MODEL_2 != None:
            if not MODEL_2 in cifar10_models:
                raise ValueError("Second model must be from the same CIFAR10 dataset.")

            model_b = hub.load("chenyaofo/pytorch-cifar-models", model=f'cifar10_{MODEL_2}', pretrained=False).to(device)


    elif DATASET == 'fashionmnist':
        from torch import hub
        from utils.datasets import FashionMNIST
        from torchvision.transforms import Compose, ToTensor, Normalize, Grayscale, Resize


        transform = Compose([  
            Resize((32,32)),
            Grayscale(num_output_channels=3),
            ToTensor(),    
            Normalize((0.2856, 0.2856, 0.2856), (0.3385, 0.3385, 0.3385))
        ])


        dataset = "FashionMNIST"

        print(f"Loading dataset {dataset}...")
        valset = FashionMNIST(root=DATASET_FILEPATH, train=False, transform=transform, return_numpy=MEMORY, duplicate_ratio=DUPLICATES, rotations=ROTATIONS, seed=42)

        model_a = hub.load("chenyaofo/pytorch-cifar-models", model=f'cifar10_{MODEL_1}', pretrained=False)     

        if MODEL_2 != None:
            if not MODEL_2 in cifar10_models:
                raise ValueError("Second model must be from the same CIFAR10 dataset.")

            model_b = hub.load("chenyaofo/pytorch-cifar-models", model=f'cifar10_{MODEL_2}', pretrained=False).to(device)
    else:
        raise "Wrong dataset provided."
    
    # Load default weights for intel and fashion mnist models
    if DATASET == 'intel':
        prefix = 'intel'
        if WEIGHTS_1 is None:
            model_a = load(f'./{DATASET}_weights/{prefix}_{MODEL_1}.pth', map_location="cpu", weights_only=False)
        if MODEL_2 is not None and WEIGHTS_2 is None:
            model_b = load(f'./{DATASET}_weights/{prefix}_{MODEL_2}.pth', map_location="cpu", weights_only=False)
    
    if DATASET == 'fashion_mnist':
        prefix = 'FM'
        if WEIGHTS_1 is None:
            model_a = load(f'./{DATASET}_weights/{prefix}_{MODEL_1}.pth', map_location="cpu", weights_only=False)
        if MODEL_2 is not None and WEIGHTS_2 is None:
            model_b = load(f'./{DATASET}_weights/{prefix}_{MODEL_2}.pth', map_location="cpu", weights_only=False)

    # Load custom weights if provided
    if WEIGHTS_1 is not None:
        print(f"Loading weights for {MODEL_1}...")
        loaded = load(WEIGHTS_1, map_location="cpu", weights_only=False)
        # In case the whole nn.Module is saved
        if isinstance(loaded, Module):
            model_a = loaded
        # In case only the state dictionary of the module is saved
        else:
            model_a.load_state_dict(loaded)
        
    if MODEL_2 is not None and WEIGHTS_2 is not None:
        print(f"Loading weights for {MODEL_2}...")
        loaded = load(WEIGHTS_2, map_location="cpu", weights_only=False)
        # In case the whole nn.Module is saved
        if isinstance(loaded, Module):
            model_b = loaded
        # In case only the state dictionary of the module is saved
        else:
            model_b.load_state_dict(loaded)

    model_a.to(device=device)
    model_a.eval()
    if MODEL_2:
        model_b.to(device=device)
        model_b.eval()    
    

    print("\n------------Configuration------------")
    print(f"Dataset: {dataset}")
    if MODEL_2:
        print(f"First Model: {MODEL_1}, {sum(p.numel() for p in model_a.parameters()) / 1000000:.2f}M parameters.")
        print(f"Second Model: {MODEL_2}, {sum(p.numel() for p in model_b.parameters())/ 1000000:.2f}M parameters.")
    else:
        print(f'Model: {MODEL_1}, {sum(p.numel() for p in model_a.parameters())/ 1000000:.2f}M parameters.')

    print(f"Device: {device}")

    if MODEL_2 and THRESHOLD and SCORE_FN:
        print(f"Score function: {SCORE_FN}")
        print(f"Threshold parameter: {THRESHOLD}")

    if MODEL_2 and POSTCHECK:
        print("Post-check enabled")

    if MEMORY:
        print(f"Using memory component: {MEMORY_METHOD}.")

    if MEMORY:
        print(f"Using {DUPLICATES * 100:.2f}% duplicated samples for a total of {len(valset)} samples.")
    
    print("-" * 37)

    

    if not MODEL_2:
        from utils.workloadfunctions import single
        response_times, trues, preds = single(model_a, valset=valset, device=device)
    else:
        if SCORE_FN == 'oracle':
            from utils.workloadfunctions import double_oracle
            response_times, trues, preds, usage = double_oracle(model_a, model_b, valset=valset, device=device)
        else:
            if not POSTCHECK:
                from utils.workloadfunctions import double
                response_times, trues, preds, usage = double(model_a=model_a, model_b=model_b, valset=valset, threshold=THRESHOLD, score_function=SCORE_FN, device=device)
            else:
                if not MEMORY:
                    from utils.workloadfunctions import double_ps
                    response_times, trues, preds, usage = double_ps(model_a=model_a, model_b=model_b, valset=valset, threshold=THRESHOLD, score_function=SCORE_FN, device=device)
                else:
                    from utils.workloadfunctions import double_ps_mem
                    response_times, trues, preds, usage = double_ps_mem(model_a=model_a, model_b=model_b, valset=valset, threshold=THRESHOLD, score_function=SCORE_FN, memory=MEMORY_METHOD, device=device)
        
    
    print("\n---Classification report---")
    print(classification_report(trues, preds, output_dict=False, zero_division=0, digits=4))
    if MODEL_2:
        print(f"Second model usage: {usage * 100:.2f}%")
    df = DataFrame(response_times)
    df.drop(index=df.index[0:25], axis=0, inplace=True)
    print("\n---Response times---")
    print(f"Mean: {df['response_time'].mean() * 1000:.3f} ms")
    print(f"95th: {df['response_time'].quantile(0.95) * 1000:.3f} ms")
    print(f"99th: {df['response_time'].quantile(0.99) * 1000:.3f} ms")

    if ROOT_PASSWORD:
        system(f"echo {ROOT_PASSWORD} | sudo -S poweroff")
    else:
        # playsound('./alarm.wav')
        pass


if __name__ == '__main__':
    main()
