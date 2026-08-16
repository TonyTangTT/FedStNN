This is the Github project of the paper Federated Learning on Stochastic Neural Networks (https://arxiv.org/abs/2506.08169). 

FedSNN_1D_Function.ipynb generate a 1D function and perturb the data with a Gaussian noise. Then distribute the data two clients and do FedStNN to reproduce the function and the noise. The trained model is saved in FedSNN_1D_global.pth. 

FedSNN_2D_Function.ipynb generate a 2D function and perturb the data with a Gaussian noise. Then distribute the data two clients and do FedStNN to reproduce the function and the noise. The trained model is saved in FedSNN_2D_Function.pth. 

FedSNN_2d_Image files read a 2D image. The image consists of three letters, and the pixels of the letter are assigned to clients as the local dataset. Then use FedStNN to regenerate the image. 
