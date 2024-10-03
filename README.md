![alt text](https://github.com/Baiaopires/Portuguese-Fake-News-Classification-with-BERT-models/blob/main/Z-Images/UnB_Logo.png)
# Portuguese-Fake-News-Classification-with-BERT-models
Repository with the python scripts used in the training and fine-tuning of the models present in the paper "Portuguese Fake News Classification with BERT models"

# Attention:
- Before running tests on the python scripts in this repository, check if you have a sufficiently powerfull GPU to run these scripts. If you do have, be sure to select it by changing the 'x' in "device = torch.device('cuda:x' if torch.cuda.is_available() else 'cpu')", founf in every python script, to the ID of your GPU. That ID can be found in the output of the terminal command 'nvidia-smi', just like below:

![alt text](https://github.com/Baiaopires/Portuguese-Fake-News-Classification-with-BERT-models/blob/main/Z-Images/Nvidia_Smi_Result.png)

- Check if you have enough space where the models, logs and plots will be placed;
- Check if all the necessary dependencies are installed, like Python and it's libraries.
