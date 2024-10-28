from sklearn.metrics import classification_report
from torch import cuda, load
from torch.nn import Module
from os import system
from utils.models_lists import cifar10_models, imagenet_models
from pandas import DataFrame
from yaml import safe_load, YAMLError

def main():
    with open("./config.yml") as stream:
        try:
            config = safe_load(stream)
        except YAMLError as exc:
            print(exc)

    DATASET = config['dataset']
    MODEL_1 = config['first_model']['name']
    MODEL_2 = config['second_model']['name'] if config['second_model']['enable'] == True else None
    SCORE_FN = config['second_model']['score_fn']
    THRESHOLD = config['second_model']['threshold']
    POSTCHECK = config['second_model']['postcheck']
    WEIGHTS_1 = config['first_model']['weights_file']
    WEIGHTS_2 = config['second_model']['weights_file']
    DATASET_FILEPATH = config['dataset_path']
    MEMORY = config['memory']['enable']
    MEMORY_METHOD = config['memory']['method']
    DUPLICATES = config['memory']['duplicates']
    ROTATIONS = config['memory']['transforms']    
    ROOT_PASSWORD = config['root_password']


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

        valset = CIFAR10(root=DATASET_FILEPATH, train=True, return_numpy=MEMORY, transform=transform, duplicate_ratio=DUPLICATES, transform_prob=ROTATIONS, random_seed=42)

        print("Loading models...")
        model_a = hub.load("chenyaofo/pytorch-cifar-models", model=f'cifar10_{MODEL_1}', pretrained=True).to(device)

        if MODEL_2 != None:
            if not MODEL_2 in cifar10_models:
                raise ValueError("Second model must be from the same CIFAR10 dataset.")

            model_b = hub.load("chenyaofo/pytorch-cifar-models", model=f'cifar10_{MODEL_2}', pretrained=True).to(device)

    
    elif DATASET == 'imagenet':
        from torchvision.models import get_model
        from utils.datasets import ImageNetC
        from torchvision.transforms import Compose, ToTensor, Normalize
        # from utils.models_lists import imagenet_models

        dataset = "ImageNet"

        transform = Compose([
            ToTensor(),
            Normalize(mean =(0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)), #ImageNet
        ])
        print("Loading dataset...")

        valset = ImageNetC(root=DATASET_FILEPATH, transform=transform, return_numpy=MEMORY, duplicate_ratio=DUPLICATES, transform_prob=0.0, random_seed=42)
        

        print("Loading models...")
        model_a = get_model(MODEL_1, weights=imagenet_models[MODEL_1])

        if MODEL_2:
            if not MODEL_2 in imagenet_models.keys():
                raise ValueError("Second model must be from the same ImageNet dataset.")

            model_b = get_model(MODEL_2, weights=imagenet_models[MODEL_2])

    
    # TODO: Testing
    elif DATASET == 'intel':
        from torch import hub
        from utils.datasets import INTEL
        from torchvision.transforms import Compose, ToTensor, Normalize, ToPILImage, Resize
        # from utils.models_lists import imagenet_models
        # n_classes = 5
        # entropy_norm_factor = entropy(tensor([1/n_classes for _ in range(n_classes)]).unsqueeze(dim=0))

        transform = Compose([
            ToPILImage(),
            Resize((32,32)),
            ToTensor(),    
            Normalize((0.4302, 0.4575, 0.4539), (0.2386, 0.2377, 0.2733))
        ])

        dataset = "INTEL"

        valset = INTEL(root=DATASET_FILEPATH, split='test', transform=transform, seed=42, return_numpy=MEMORY, duplicate_ratio=DUPLICATES, transform_prob=0.0)        

        model_a = hub.load("chenyaofo/pytorch-cifar-models", model=f'cifar10_{MODEL_1}', pretrained=False)     

        if MODEL_2 != None:
            if not MODEL_2 in cifar10_models:
                raise ValueError("Second model must be from the same CIFAR10 dataset.")

            model_b = hub.load("chenyaofo/pytorch-cifar-models", model=f'cifar10_{MODEL_2}', pretrained=False).to(device)


    elif DATASET == 'fashion_mnist':
        from torch import hub        
        # from torchvision.datasets import FashionMNIST
        from utils.datasets import FashionMNISTc
        from torchvision.transforms import Compose, ToTensor, Normalize, Grayscale, Resize, ToPILImage
        # from utils.models_lists import imagenet_models

        # n_classes = 10
        # entropy_norm_factor = entropy(tensor([1/n_classes for _ in range(n_classes)]).unsqueeze(dim=0))


        transform = Compose([
            # ToPILImage(),    
            Resize((32,32)),
            Grayscale(num_output_channels=3),
            ToTensor(),    
            Normalize((0.2856, 0.2856, 0.2856), (0.3385, 0.3385, 0.3385))
        ])


        dataset = "FashionMNIST"

        print(f"Loading dataset {dataset}...")
        valset = FashionMNISTc(root=DATASET_FILEPATH, split='test', transform=transform, seed=42, return_numpy=MEMORY, duplicate_ratio=DUPLICATES, transform_prob=0.0)

        model_a = hub.load("chenyaofo/pytorch-cifar-models", model=f'cifar10_{MODEL_1}', pretrained=False)     

        if MODEL_2 != None:
            if not MODEL_2 in cifar10_models:
                raise ValueError("Second model must be from the same CIFAR10 dataset.")

            model_b = hub.load("chenyaofo/pytorch-cifar-models", model=f'cifar10_{MODEL_2}', pretrained=False).to(device)
    else:
        raise "Wrong dataset provided."
    

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

    # Load weights if provided
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
        from Deliverable.utils.workloadfunctions import single
        response_times, trues, preds = single(model_a, valset=valset, device=device)
    else:
        if SCORE_FN == 'oracle':
            from Deliverable.utils.workloadfunctions import double_oracle
            response_times, trues, preds, usage = double_oracle(model_a, model_b, valset=valset, device=device)
        else:
            if not POSTCHECK:
                from Deliverable.utils.workloadfunctions import double
                response_times, trues, preds, usage = double(model_a=model_a, model_b=model_b, valset=valset, threshold=THRESHOLD, score_function=SCORE_FN, device=device)
            else:
                if not MEMORY:
                    from Deliverable.utils.workloadfunctions import double_ps
                    response_times, trues, preds, usage = double_ps(model_a=model_a, model_b=model_b, valset=valset, threshold=THRESHOLD, score_function=SCORE_FN, device=device)
                else:
                    from Deliverable.utils.workloadfunctions import double_ps_mem
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
