# from conf.args import Arguments
from model import MCQAModel
# from dataset import MCQADataset
# import pytorch_lightning
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers import CSVLogger
import pytorch_lightning as pl
# from pytorch_lightning.core.step_result import TrainResult,EvalResult
from pytorch_lightning import Trainer
from torch.utils.data import Dataset
from dataclasses import dataclass
import torch,os,sys
import pandas as pd
from tqdm import tqdm
from transformers import AutoModel
from multiprocessing import Process
import time,argparse


EXPERIMENT_DATASET_FOLDER = "../data/Q2_(2)_dev_data.json"
WB_PROJECT = "MEDMCQA"

def eval(gpu,
          args,
          exp_dataset_folder,
          experiment_name,
          models_folder,
          version):

    pl.seed_everything(42)
    
    EXPERIMENT_FOLDER = os.path.join(models_folder,experiment_name)
    os.makedirs(EXPERIMENT_FOLDER,exist_ok=True)
    experiment_string = experiment_name+'-{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}'


    wb = WandbLogger(project=WB_PROJECT,name=experiment_name,version=version)
    csv_log = CSVLogger(models_folder, name=experiment_name, version=version)

    # train_dataset = MCQADataset(args.train_csv,args.use_context)
    # test_dataset = MCQADataset(args.test_csv,args.use_context)
    val_dataset = MCQADataset(args.dev_csv,args.use_context)

    es_callback = pl.callbacks.EarlyStopping(monitor='val_loss',
                                    min_delta=0.00,
                                    patience=2,
                                    verbose=True,
                                    mode='min')

    cp_callback = pl.callbacks.ModelCheckpoint(monitor='val_loss',
                                               filepath=os.path.join(EXPERIMENT_FOLDER,experiment_string),
                                               save_top_k=1,
                                               save_weights_only=True,
                                               mode='min')

    mcqaModel = MCQAModel(model_name_or_path=args.pretrained_model_name,
                      args=args.__dict__)
    
    mcqaModel.val_dataset=val_dataset

    trainer = Trainer(gpus=gpu,
                    distributed_backend='ddp' if not isinstance(gpu,list) else None,
                    logger=[wb,csv_log],
                    callbacks= [es_callback,cp_callback],
                    max_epochs=args.num_epochs)
    
    # trainer.fit(mcqaModel)
    # print(f"Training completed")

    ckpt = [f for f in os.listdir(EXPERIMENT_FOLDER) if f.endswith('.ckpt')]

    # self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    inference_model = AutoModel.from_pretrained("../output/checkpoint-10000")
    inference_model = inference_model.to("cuda")
    inference_model = inference_model.eval()

    # _,test_results = trainer.test(ckpt_path=os.path.join(EXPERIMENT_FOLDER,ckpt[0]))
    # wb.log_metrics(test_results)
    # csv_log.log_metrics(test_results)


    # #Persist test dataset predictions
    # test_df = pd.read_csv(args.test_csv)
    # test_df.loc[:,"predictions"] = [pred+1 for pred in run_inference(inference_model,mcqaModel.test_dataloader(),args)]
    # test_df.to_csv(os.path.join(EXPERIMENT_FOLDER,"test_results.csv"),index=False)
    # print(f"Test predictions written to {os.path.join(EXPERIMENT_FOLDER,'test_results.csv')}")

    val_df = pd.read_json(args.dev_csv, lines=True)
    val_df.loc[:,"predictions"] = [pred+1 for pred in run_inference(inference_model,mcqaModel.val_dataloader(),args)]
    val_df.to_csv(os.path.join(EXPERIMENT_FOLDER,"dev_results.csv"),index=False)
    print(f"Val predictions written to {os.path.join(EXPERIMENT_FOLDER,'dev_results.csv')}")

    del mcqaModel
    del inference_model
    del trainer
    torch.cuda.empty_cache()
    
def run_inference(model,dataloader,args):
    predictions = []
    for idx,(inputs,labels) in tqdm(enumerate(dataloader)):
        batch_size = len(labels)
        for key in inputs.keys():
            inputs[key] = inputs[key].to(args.device)
        with torch.no_grad():
            outputs = model(**inputs)
        # print(outputs)
        prediction_idxs = torch.argmax(outputs.last_hidden_state,axis=1).cpu().detach().numpy()
        predictions.extend(list(prediction_idxs))
    return predictions

class MCQADataset(Dataset):

  def __init__(self,
               csv_path,
               use_context=True):
#     self.dataset = dataset['train'] if training == True else dataset['test']
    self.dataset = pd.read_json(csv_path, lines=True)
    self.use_context = use_context

  def __len__(self):
    return len(self.dataset)
  
  def __getitem__(self,idx):
    return_tuple = tuple()
    if self.use_context:
      context = self.dataset.loc[idx,'exp']
      return_tuple+=(context,)
    question = self.dataset.loc[idx,'question']
    options = self.dataset.loc[idx,['opa', 'opb', 'opc', 'opd']].values
    label = self.dataset.loc[idx,'cop'] - 1
    return_tuple+=(question,options,label)
    return return_tuple

@dataclass    
class Arguments:
    # train_csv:str
    # test_csv:str 
    dev_csv:str
    batch_size:int = 16
    max_len:int = 192
    checkpoint_batch_size:int = 32
    print_freq:int = 100
    pretrained_model_name:str = "bert-base-uncased"
    learning_rate:float = 2e-4
    hidden_dropout_prob:float =0.4
    hidden_size:int=768
    num_epochs:int = 5
    num_choices:int = 4
    device:str='cuda'
    gpu='6'
    use_context:bool=True

if __name__ == "__main__":

    models = ["allenai/scibert_scivocab_uncased","bert-base-uncased"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",default="bert-base-uncased",help="name of the model")
    parser.add_argument("--dataset_folder_name", default="../data/Q2_(2)_dev_data.json",help="dataset folder")
    parser.add_argument("--use_context",default=False,action='store_true',help="mention this flag to use_context")
    cmd_args = parser.parse_args()

    # exp_dataset_folder = os.path.join(EXPERIMENT_DATASET_FOLDER,cmd_args.dataset_folder_name)
    model = cmd_args.model
    # if((model == "allenai/scibert_scivocab_uncased" and os.path.basename(exp_dataset_folder) == "single_high_pubmed_exp") or 
    #    (model == "allenai/scibert_scivocab_uncased" and os.path.basename(exp_dataset_folder) == "multi_high_pubmed_exp")):
    #     exit()
    print(f"Training started for model - {model} variant - {EXPERIMENT_DATASET_FOLDER} use_context - {str(cmd_args.use_context)}")

    args = Arguments(dev_csv=os.path.join("../data/Q2_(2)_dev_data.json"),
                    pretrained_model_name=model,
                    use_context=cmd_args.use_context)
    
    exp_name = f"{model}@@@{os.path.basename(EXPERIMENT_DATASET_FOLDER)}@@@use_context{str(cmd_args.use_context)}@@@seqlen{str(args.max_len)}".replace("/","_")


    eval(gpu=args.gpu,
        args=args,
        exp_dataset_folder=EXPERIMENT_DATASET_FOLDER,
        experiment_name=exp_name,
        models_folder="./results",
        version=exp_name)
    
    time.sleep(60)