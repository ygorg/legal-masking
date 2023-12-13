import torch

#device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
#print("====================================================================")
#print(device)
#print(torch.cuda.device_count())
print(torch.cuda.is_available())
print(torch.version.cuda)
print(torch.cuda.device_count())
