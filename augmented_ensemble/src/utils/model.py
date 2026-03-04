import torch


def build_model(settings):
    model_name = settings['model']['name']
    if model_name == 'ReFineNet':
        from building_footprint_segmentation.seg.binary.models import ReFineNet
        from torch.utils import model_zoo

        model = ReFineNet()
        device = f"cuda:{settings['trainer']['devices'][0]}" if settings['trainer']['accelerator'] == 'gpu' else 'cpu'
        if settings['model']['args']['weights'] == 'inria':
            state_dict = model_zoo.load_url(
                "https://github.com/fuzailpalnak/building-footprint-segmentation/releases/download/alpha/refine.zip",
                progress=True,
            )
            model.load_state_dict(state_dict)
        if settings['task']['loss']['name'] == 'AleatoricUQLoss':
            model.final_layer = torch.nn.Conv2d(model.final_layer.in_channels, out_channels=2, kernel_size=1, stride=1)
        else:
            # add a sigmoid layer to the final layer
            model.final_layer = torch.nn.Sequential(model.final_layer, torch.nn.Sigmoid())
        model.to(device)
    else:
        import segmentation_models_pytorch as smp
        # assume the model is from segmentation_models_pytorch
        assert hasattr(smp, model_name), f'Model {model_name} not found in segmentation_models_pytorch'
        model = getattr(smp, model_name)(**settings['model']['args'])

        if settings['task']['loss']['name'] == 'AleatoricUQLoss':
            raise NotImplementedError('AleatoricUQLoss not implemented for segmentation_models_pytorch')

    return model
