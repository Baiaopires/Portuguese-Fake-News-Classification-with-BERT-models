![alt text](https://github.com/Baiaopires/Portuguese-Fake-News-Classification-with-BERT-models/blob/main/z-Images/UnB_Logo.png)
# Portuguese-Fake-News-Classification-with-BERT-models
Repository with the python scripts used in the training and fine-tuning of the models present in the paper "Portuguese Fake News Classification with BERT models"

# Attention:
- <div align="justify">Before running tests on the python scripts in this repository, check if you have a sufficiently powerful NVIDIA GPU to run these scripts. If you do have, be sure to select it by changing the 'x' in "device = torch.device('cuda:x' if torch.cuda.is_available() else 'cpu')", found in every python script in this repository, to the ID of your GPU. That ID can be found in the output of the terminal command 'nvidia-smi', just like below: </div>

<div align="center">
	<img src="https://github.com/Baiaopires/Portuguese-Fake-News-Classification-with-BERT-models/blob/main/z-Images/Nvidia_Smi_Result.png">
</div>

- <div align="justify">Check if you have enough space where the models, logs and plots will be saved;</div>
- <div align="justify">Check if all the necessary dependencies are installed, like Python and it's libraries.</div>
