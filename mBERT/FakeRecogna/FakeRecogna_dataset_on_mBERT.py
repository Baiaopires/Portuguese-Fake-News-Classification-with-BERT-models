import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import tqdm as tqdm
import transformers as ppb
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

############################################################################################################################################################################################################################################################

df = pd.read_csv('https://raw.githubusercontent.com/Baiaopires/datasets_used/main/datasets_used/FakeRecogna/FakeRecogna.csv', delimiter=',', skiprows = lambda x: x in [0], header=None)
df.drop(columns=[0, 1, 3, 4, 5, 6], inplace=True)
df.drop(
    labels=[7337],
    axis=0,
    inplace=True
)
df.reset_index(inplace=True, drop=True)
df = df.rename(columns={2: 0, 7: 1})
df[1] = df[1].astype(int)
labels = df[1]

############################################################################################################################################################################################################################################################

#converting arrays to tensors so that Torch functions work properly
tokenizer = ppb.BertTokenizer.from_pretrained(
    'bert-base-multilingual-uncased',
    do_lower_case = True
    )

tokenized = []
attention_masks = []

def preprocessing(input_text, tokenizer):
  return tokenizer.encode_plus(
                        input_text,
                        add_special_tokens = True,
                        max_length = 128,
                        pad_to_max_length = True,
                        return_attention_mask = True,
                        return_tensors = 'pt'
                   )

for sample in df[0].values:
    encoding_dict = preprocessing(sample, tokenizer)
    tokenized.append(encoding_dict['input_ids'])
    attention_masks.append(encoding_dict['attention_mask'])

tokenized = torch.cat(tokenized, dim = 0)
attention_masks = torch.cat(attention_masks, dim = 0)
labels = torch.tensor(labels)

val_plus_test_ratio = 0.4
test_ratio = 0.5 # 50% of 40%, which is 20% of the total
batch_size = 128

train_idx, val_plus_test_idx = train_test_split(
    np.arange(len(labels)),
    test_size = val_plus_test_ratio,
    shuffle = True,
    stratify = labels)

val_idx, test_idx = train_test_split(
    val_plus_test_idx,
    test_size = test_ratio,
    shuffle = True)

train_set = TensorDataset(tokenized[train_idx],
                          attention_masks[train_idx],
                          labels[train_idx])

val_set = TensorDataset(tokenized[val_idx],
                        attention_masks[val_idx],
                        labels[val_idx])

train_dataloader = DataLoader(
                   train_set,
                   sampler = RandomSampler(train_set),
                   batch_size = batch_size
                )

validation_dataloader = DataLoader(
                  val_set,
                  sampler = RandomSampler(val_set),
                  batch_size = batch_size
                )

############################################################################################################################################################################################################################################################

#true positives
def b_tp(preds, labels):
  return sum([preds == labels and preds == 1 for preds, labels in zip(preds, labels)])

#false positives
def b_fp(preds, labels):
  return sum([preds != labels and preds == 1 for preds, labels in zip(preds, labels)])

#true negatives
def b_tn(preds, labels):
  return sum([preds == labels and preds == 0 for preds, labels in zip(preds, labels)])

#false negatives
def b_fn(preds, labels):
  return sum([preds != labels and preds == 0 for preds, labels in zip(preds, labels)])

def b_metrics(preds, labels):
  '''
  Returns the following metrics:
    - accuracy    = (TP + TN) / N
    - precision   = TP / (TP + FP)
    - recall      = TP / (TP + FN)
    - specificity = TN / (TN + FP)
  '''
  preds = np.argmax(preds, axis = 1).flatten()
  labels = labels.flatten()

  tp = b_tp(preds, labels)
  fp = b_fp(preds, labels)
  tn = b_tn(preds, labels)
  fn = b_fn(preds, labels)

  b_accuracy = (tp + tn)/len(labels)
  b_precision = tp / (tp + fp) if (tp + tp) > 0 else 'nan'
  b_recall = tp / (tp + fn) if (tp + fn) > 0 else 'nan'
  b_specificity = tn / (tn + fp) if (tp + fp) > 0 else 'nan'

  return b_accuracy, b_precision, b_recall, b_specificity

############################################################################################################################################################################################################################################################

model = ppb.BertForSequenceClassification.from_pretrained(
    'bert-base-multilingual-uncased',
    num_labels = 2,
    output_attentions = False,
    output_hidden_states = False
)

#use of gpu if available (put "if torch.cuda.is_available() else 'cpu' after the 'cpu' argument in torch.device()" to choose cpu if gpu doesn't work)
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

model.cuda(device)

optimizer = torch.optim.AdamW(model.parameters(),
                            lr = 5e-5,
                            eps = 5e-08
                        )

############################################################################################################################################################################################################################################################

class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)


        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False


    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False


        if np.isnan(metrics):
            return True


        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1


        if self.num_bad_epochs >= self.patience:
            print("\n terminating because of early stopping!")
            return True


        return False


    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                            best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                            best * min_delta / 100)

############################################################################################################################################################################################################################################################

val_loss = 1
best_val_accuracy = 0
best_epoch = 0
epoch_counter = 1
bool_value = 0

x = []
y = []

es = EarlyStopping(patience=100)

def generator():
  while not(es.step(val_loss)):
    yield

for _ in tqdm.tqdm(generator()):

  model.train()

  bool_value = 0
  tr_loss = 0
  nb_tr_examples, nb_tr_steps = 0, 0

  for step, batch in enumerate(train_dataloader):
    batch = tuple(t.to(device) for t in batch)
    b_input_ids, b_input_mask, b_labels = batch
    optimizer.zero_grad()

    train_output = model(b_input_ids,
                          token_type_ids = None,
                          attention_mask = b_input_mask,
                          labels = b_labels
                          )

    train_output.loss.backward()
    optimizer.step()

    tr_loss = tr_loss + train_output.loss.item()
    nb_tr_examples = nb_tr_examples + b_input_ids.size(0)
    nb_tr_steps = nb_tr_steps + 1

############################################################################################################################################################################################################################################################

  model.eval()

  val_accuracy = []
  val_precision = []
  val_recall = []
  val_specificity = []

  for batch in validation_dataloader:
    batch = tuple(t.to(device) for t in batch)
    b_input_ids, b_input_mask, b_labels = batch
    with torch.no_grad():
      eval_output = model(b_input_ids,
                          token_type_ids = None,
                          attention_mask = b_input_mask)

    logits = eval_output.logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()

    b_accuracy, b_precision, b_recall, b_specificity = b_metrics(logits, label_ids)
    val_accuracy.append(b_accuracy)
    if b_precision != 'nan': val_precision.append(b_precision)
    if b_recall != 'nan': val_recall.append(b_recall)
    if b_specificity != 'nan': val_specificity.append(b_specificity)

  y.append(sum(val_accuracy)/len(val_accuracy))
  x.append(epoch_counter)

  epoch_counter += 1
  val_loss = 1 - (sum(val_accuracy)/len(val_accuracy))

  if((sum(val_accuracy)/len(val_accuracy)) > best_val_accuracy):
    best_val_accuracy = (sum(val_accuracy)/len(val_accuracy)),
    best_epoch = epoch_counter
    bool_value = 1
    model.save_pretrained("./Models/FakeRecogna_on_mBERT_FineTunned_Model_epoch_{0}_val_acc_{1}".format(best_epoch, best_val_accuracy))

  print(' - Epoch: ', (epoch_counter - 1), end = '')

  if(bool_value):
      print(' - Occurence of Best Validation Accuracy In This Epoch')
  else: print('\n', end = '')

  #printing the average accuracy, precision, recall and specificity metrics obtained in validation

  print('\t - Train loss: {:.8f}'.format(tr_loss / nb_tr_steps))
  print('\t - Validation Accuracy: {:.8f}'.format(sum(val_accuracy)/len(val_accuracy)))
  print('\t - Validation Precision: {:.8f}'.format(sum(val_precision)/len(val_precision)) if len(val_precision)>0 else '\t - Validation Precision: NaN')
  print('\t - Validation Recall: {:.8f}'.format(sum(val_recall)/len(val_recall)) if len(val_recall)>0 else '\t - Validation Recall: NaN')
  print('\t - Validation Specificity: {:.8f}\n'.format(sum(val_specificity)/len(val_specificity)) if len(val_specificity)>0 else '\t - Validation Specificity: NaN')

############################################################################################################################################################################################################################################################

plt.plot(x, y)
plt.xlabel('Number of Epochs')
plt.ylabel('Validation Accuracy')
#plt.ylim(0, 1)
plt.xlim(0, epoch_counter)
plt.savefig("./Plots/Plot_FakeRecogna_Dataset_On_mBERT_val_acc_{0}.png".format(best_val_accuracy), bbox_inches='tight')
plt.show()

############################################################################################################################################################################################################################################################

model = ppb.AutoModelForSequenceClassification.from_pretrained(
    "./Models/FakeRecogna_on_mBERT_FineTunned_Model_epoch_{0}_val_acc_{1}".format(best_epoch, best_val_accuracy),
    num_labels = 2,
    output_attentions = False,
    output_hidden_states = False
)

test_sum_of_true_positives = 0
test_sum_of_false_positives = 0
test_sum_of_true_negatives = 0
test_sum_of_false_negatives = 0
test_accuracy = 0
test_precision = 0
test_recall = 0
test_specificity = 0

for i in range(len(test_idx)):
  new_sentence = df[0][test_idx[i]]

  sentence_id = []
  sentence_mask = []

  new_encoding = preprocessing(new_sentence, tokenizer)
  sentence_id.append(new_encoding['input_ids'])
  sentence_mask.append(new_encoding['attention_mask'])

  sentence_id = torch.cat(sentence_id, dim = 0)
  sentence_mask = torch.cat(sentence_mask, dim = 0)

  device = torch.device('cpu')
  model.cpu()

  with torch.no_grad():
    output = model(sentence_id, token_type_ids = None, attention_mask = sentence_mask.to(device))

  prediction = 1 if np.argmax(output.logits.cpu().numpy()).flatten().item() == 1 else 0

  if((prediction == 1) and (df[1][test_idx[i]] == 1)): test_sum_of_true_positives += 1
  if((prediction == 1) and (df[1][test_idx[i]] == 0)): test_sum_of_false_positives += 1
  if((prediction == 0) and (df[1][test_idx[i]] == 0)): test_sum_of_true_negatives += 1
  if((prediction == 0) and (df[1][test_idx[i]] == 1)): test_sum_of_false_negatives += 1

test_accuracy = ((test_sum_of_true_positives + test_sum_of_true_negatives)/(len(test_idx)))
test_precision = test_sum_of_true_positives / (test_sum_of_true_positives + test_sum_of_false_positives) if (test_sum_of_true_positives + test_sum_of_true_positives) > 0 else 'nan'
test_recall = test_sum_of_true_positives / (test_sum_of_true_positives + test_sum_of_false_negatives) if (test_sum_of_true_positives + test_sum_of_false_negatives) > 0 else 'nan'
test_specificity = test_sum_of_true_negatives / (test_sum_of_true_negatives + test_sum_of_false_positives) if (test_sum_of_true_positives + test_sum_of_false_positives) > 0 else 'nan'

print(' - Test Accuracy: {:.8f}'.format(test_accuracy))
print(' - Test Precision: {:.8f}'.format(test_precision) if len(test_idx) > 0 else '\t - Test Precision: NaN')
print(' - Test Recall: {:.8f}'.format(test_recall) if len(test_idx) > 0 else '\t - Test Recall: NaN')
print(' - Test Specificity: {:.8f}\n'.format(test_specificity) if len(test_idx) > 0 else '\t - Test Specificity: NaN')
