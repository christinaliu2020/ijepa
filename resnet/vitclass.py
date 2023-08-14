import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from src.models.vision_transformer import VisionTransformer, vit_predictor, vit_small, MLP
# Define your ContextEncoder, Predictor, and TargetEncoder classes
from src.helper import init_model, load_checkpoint, init_opt

checkpoint = torch.load('ijepa/logs/jepa-latest.pth.tar')

# Create an instance of the ContextEncoder, Predictor, and TargetEncoder
# encoder = vit_small()
# predictor = vit_predictor()
# target_encoder = vit_small()

encoder, predictor = init_model(
    device='cuda',
    patch_size=16,
    model_name='vit_small', #changed from vit_base
    crop_size=224,
    pred_depth=6,
    pred_emb_dim=384
)
target_encoder = copy.deepcopy(encoder)
# param_groups = [
#         {
#             'params': (p for n, p in encoder.named_parameters()
#                        if ('bias' not in n) and (len(p.shape) != 1))
#         }, {
#             'params': (p for n, p in predictor.named_parameters()
#                        if ('bias' not in n) and (len(p.shape) != 1))
#         }, {
#             'params': (p for n, p in encoder.named_parameters()
#                        if ('bias' in n) or (len(p.shape) == 1)),
#             'WD_exclude': True,
#             'weight_decay': 0
#         }, {
#             'params': (p for n, p in predictor.named_parameters()
#                        if ('bias' in n) or (len(p.shape) == 1)),
#             'WD_exclude': True,
#             'weight_decay': 0
#         }
#     ]
#
# encoder, predictor, target_encoder, opt, scaler, epoch = load_checkpoint(device ='cuda', r_path='/home/christina/PycharmProjects/ijepa/logs/jepa-latest.pth.tar', encoder=encoder, predictor=predictor, target_encoder=target_encoder, opt = torch.optim.AdamW(param_groups), scaler = None)
new_enc_state_dict = {}
for key, value in checkpoint['encoder'].items():
    if key.startswith('module.'):
        new_key = key[len('module.'):]
        new_enc_state_dict[new_key] = value
    else:
        new_enc_state_dict[key] = value

# Load the new state_dict into your model
encoder.load_state_dict(new_enc_state_dict)
print(new_enc_state_dict.keys())
print(checkpoint['predictor'].keys())
# Create a new state_dict with modified keys
new_pred_state_dict = {}
for key, value in checkpoint['predictor'].items():
    if key.startswith('module.'):
        new_key = key[len('module.'):]
    else:
        new_key = key
    new_pred_state_dict[new_key] = value
print(new_pred_state_dict.keys())

def remove_keys_between_indices(dictionary, start_index, end_index):
    new_dict = {k: v for i, (k, v) in enumerate(dictionary.items()) if i < start_index or i > end_index}
    return new_dict
print(len(new_pred_state_dict.keys()))
new_pred_state_dict = remove_keys_between_indices(new_pred_state_dict, 76, 147)
print(new_pred_state_dict.keys())
#Unexpected key(s) in state_dict: "predictor_blocks.6.norm1.weight", "predictor_blocks.6.norm1.bias"
predictor.load_state_dict(new_pred_state_dict)

new_tar_state_dict = {}
for key, value in checkpoint['target_encoder'].items():
    if key.startswith('module.'):
        new_key = key[len('module.'):]
        new_tar_state_dict[new_key] = value
    else:
        new_tar_state_dict[key] = value
target_encoder.load_state_dict(new_tar_state_dict)
# print(new_tar_state_dict.keys())
# Replace predictor with classification head

#mlp = MLP(in_features = 1536, out_features = 10)
classification_head = nn.Linear(in_features=predictor.embed_dim, out_features = 10)
predictor = classification_head
#target_encoder = nn.Identity()
# classification_head = nn.Linear(in_features=384, out_features=num_classes)
# predictor = classification_head

# Load and preprocess CIFAR-10 dataset for classification
# # Define data transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

classification_dataset = CIFAR10(root='data', train=True, transform=transform, download=True)
classification_dataloader = DataLoader(classification_dataset, batch_size=32, shuffle=True)

# Define optimizer and loss function
optimizer = optim.Adam(predictor.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# Fine-tuning classification head on CIFAR-10
num_epochs = 10
loss_values = []
for epoch in range(num_epochs):
    for batch_idx, batch in enumerate(classification_dataloader):
        inputs, targets = batch

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        logits = encoder(inputs.to('cuda'))
        pred_output = predictor(logits.to('cpu'))
        logits_pooled = torch.mean(pred_output, dim=1)
        #print(logits.shape)
        loss = criterion(logits_pooled, targets)

        # Backpropagation and optimization
        loss.backward()
        optimizer.step()

        # Print progress
        if (batch_idx + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(classification_dataloader)}], Loss: {loss.item()}')
    loss_values.append(loss)
# Save the fine-tuned classification head
loss_values = loss_values.detach().numpy()
epochs = range(1, len(loss_values) + 1)
import matplotlib.pyplot as plt
# Plot the loss curve
plt.plot(epochs, loss_values, 'b', label='Training Loss')
plt.title('Training Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

torch.save(encoder.state_dict(), 'resnet/fine_tuned_model.pth')

