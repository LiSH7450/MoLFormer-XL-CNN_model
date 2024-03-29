import time
import args
import random
import os
from functools import partial

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics import classification
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
from fast_transformers.masking import LengthMask as LM
from fast_transformers.feature_maps import GeneralizedRandomFeatures
from apex import optimizers
import numpy as np
import pandas as pd

from rotate_attention.rotate_builder import RotateEncoderBuilder as rotate_builder
from tokenizer.tokenizer import MolTranBertTokenizer
from pytorch_lightning.callbacks import EarlyStopping,ModelCheckpoint
from sklearn import metrics


class CNN(nn.Module):
    def __init__(self, smiles_embed_dim):  
        super(CNN, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels=smiles_embed_dim, out_channels=smiles_embed_dim*3, kernel_size=3, padding=1),
            nn.BatchNorm1d(num_features=smiles_embed_dim*3),
            nn.ReLU(),
            nn.Conv1d(in_channels=smiles_embed_dim*3, out_channels=smiles_embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(num_features=smiles_embed_dim),
            nn.ReLU(),
            # nn.AdaptiveMaxPool1d(1)
            )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv_block(x)
        return x



class LightningModule(pl.LightningModule):
    """  the full GPT language model, with a context size of block_size """
    def __init__(self, config, tokenizer):
        super(LightningModule, self).__init__()
        self.config = config
        self.hparams.update(vars(config))
        self.mode = config.mode
        self.tokenizer=tokenizer
        self.min_loss = {
            self.hparams.measure_name + "min_valid_loss": torch.finfo(torch.float32).max,
            self.hparams.measure_name + "min_epoch": 0,
        }
        self.max_rocauc = {"max_train_rocauc": 0, "max_valid_rocauc": 0, "max_test_rocauc": 0}
        # 用来存每个validation的输出
        self.validation_outputs = [[],[]]
        self.training_outputs = []
        # Word embeddings layer
        n_vocab, d_emb = len(tokenizer.vocab), config.n_embd
        # input embedding stem
        builder = rotate_builder.from_kwargs(
            n_layers=config.n_layer,
            n_heads=config.n_head,
            query_dimensions=config.n_embd//config.n_head,
            value_dimensions=config.n_embd//config.n_head,
            feed_forward_dimensions=config.n_embd,
            attention_type='linear',
            feature_map=partial(GeneralizedRandomFeatures, n_dims=config.num_feats),
            activation='gelu',
            )
        self.pos_emb = None
        self.tok_emb = nn.Embedding(n_vocab, config.n_embd)
        self.drop = nn.Dropout(config.d_dropout)
        ## transformer
        self.blocks = builder.get()
        self.lang_model = self.lm_layer(config.n_embd, n_vocab)
        self.train_config = config
        #if we are starting from scratch set seeds
        #########################################
        # protein_emb_dim, smiles_embed_dim, dims=dims, dropout=0.2):
        #########################################

        self.fcs = []  # nn.ModuleList()
        self.loss = torch.nn.CrossEntropyLoss()
        # self.loss = torch.nn.BCELoss(reduction='sum')

        self.net = CNN(config.n_embd)
        
        self.ffn = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd),
            nn.Dropout(config.d_dropout),
            nn.ReLU(),
            nn.Linear(config.n_embd, self.hparams.num_classes),)

        self.validation_outputs = [[], []]



    class lm_layer(nn.Module):
        def __init__(self, n_embd, n_vocab):
            super().__init__()
            self.embed = nn.Linear(n_embd, n_embd)
            self.ln_f = nn.LayerNorm(n_embd)
            self.head = nn.Linear(n_embd, n_vocab, bias=False)
        def forward(self, tensor):
            tensor = self.embed(tensor)
            tensor = F.gelu(tensor)
            tensor = self.ln_f(tensor)
            tensor = self.head(tensor)
            return tensor

    def get_loss(self, smiles_emb, measures):
        z_pred = self.ffn(smiles_emb)
        measures = measures.long()
        # print('z_pred: {0}, measures: {1}'.format(z_pred, measures))
        loss = self.loss(z_pred, measures)
        return loss, z_pred, measures

    def on_save_checkpoint(self, checkpoint):
        #save RNG states each time the model and states are saved
        out_dict = dict()
        out_dict['torch_state']=torch.get_rng_state()
        out_dict['cuda_state']=torch.cuda.get_rng_state()
        if np:
            out_dict['numpy_state']=np.random.get_state()
        if random:
            out_dict['python_state']=random.getstate()
        checkpoint['rng'] = out_dict

    def on_load_checkpoint(self, checkpoint):
        #load RNG states each time the model and states are loaded from checkpoint
        rng = checkpoint['rng']
        for key, value in rng.items():
            if key =='torch_state':
                torch.set_rng_state(value)
            elif key =='cuda_state':
                torch.cuda.set_rng_state(value)
            elif key =='numpy_state':
                np.random.set_state(value)
            elif key =='python_state':
                random.setstate(value)
            else:
                print('unrecognized state')

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self):
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding, torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)


        if self.pos_emb != None:
            no_decay.add('pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.0},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        if self.hparams.measure_name == 'r2':
            betas = (0.9, 0.999)
        else:
            betas = (0.9, 0.99)
        print('betas are {}'.format(betas))
        learning_rate = self.train_config.lr_start * self.train_config.lr_multiplier
        # optimizer = optimizers.FusedLAMB(optim_groups, lr=learning_rate, betas=betas, weight_decay = self.train_config.weight_decay)

        #optimizer = optimizers.FusedAdam(optim_groups, lr=learning_rate, betas=betas, weight_decay = self.train_config.weight_decay, adam_w_mode = True)
        from torch.optim import AdamW
        optimizer = AdamW(optim_groups, lr=learning_rate, betas=betas, weight_decay=0.01)
        
        return {"optimizer": optimizer}

    def training_step(self, batch, batch_idx):
        idx = batch[0]
        mask = batch[1]
        targets = batch[2]
        loss = 0
        loss_tmp = 0
        b = idx.shape[0]
        token_embeddings = self.tok_emb(idx) # each index maps to a (learnable) vector
        x = self.drop(token_embeddings)
        x = self.blocks(x, length_mask=LM(mask.sum(-1)))
        token_embeddings = x
        input_mask_expanded = mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        token_embeddings = token_embeddings * input_mask_expanded
        token_embeddings = self.net(token_embeddings)
        token_embeddings = token_embeddings.reshape(b, -1, 768)
        
        sum_embeddings = torch.sum(token_embeddings, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        loss_input = sum_embeddings / sum_mask
        loss, pred, actual = self.get_loss(loss_input, targets)

        self.log('train_loss', loss, on_step=True, prog_bar=True, logger=True)

        logs = {"train_loss": loss}
        self.training_outputs.append({
            "pred": pred.detach(),
            "actual": actual.detach(),
        })

        return {"loss": loss}
    
    def on_train_epoch_end(self):
        tensorboard_logs = {}
        outputs = self.training_outputs
        batch_outputs = outputs
        dataset = "training"
        accuracy, roc_auc, recall, F1, precision, mcc, auprc, bacc, sp, se  = self.epoch_end(batch_outputs)
        
        tensorboard_logs.update(
            {
                # dataset + "_avg_val_loss": avg_loss,
                dataset + "_acc": accuracy,
                dataset + "_rocauc": roc_auc,
            }
        )
        self.log("train_acc", accuracy, on_epoch=True, prog_bar=True)
        self.log("train_rocauc", roc_auc, on_epoch=True, prog_bar=True)
        self.log("train_recall", recall, on_epoch=True, prog_bar=False)
        self.log("train_F1", F1, on_epoch=True, prog_bar=False)
        self.log("train_precision", precision, on_epoch=True, prog_bar=False)
        self.log("train_mcc", mcc, on_epoch=True, prog_bar=True)
        self.log("train_bacc", bacc, on_epoch=True, prog_bar=True)
        self.log("train_auprc", auprc, on_epoch=True, prog_bar=True)
        self.log("train_sp", sp, on_epoch=True, prog_bar=True)
        self.log("train_se", se, on_epoch=True, prog_bar=True)
        self.training_outputs.clear()
        
    def validation_step(self, val_batch, batch_idx, dataloader_idx):
        idx = val_batch[0]
        mask = val_batch[1]
        targets = val_batch[2]
        
        loss = 0
        loss_tmp = 0
        b, t = idx.size()
        
        token_embeddings = self.tok_emb(idx) # each index maps to a (learnable) vector
        x = self.drop(token_embeddings)
        x = self.blocks(x, length_mask=LM(mask.sum(-1)))
        token_embeddings = x
        input_mask_expanded = mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        token_embeddings = token_embeddings * input_mask_expanded
        token_embeddings = self.net(token_embeddings)
        token_embeddings = token_embeddings.reshape(b, -1, 768)
        
        sum_embeddings = torch.sum(token_embeddings, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        loss_input = sum_embeddings / sum_mask
        loss, pred, actual = self.get_loss(loss_input, targets)
        
        self.validation_outputs[dataloader_idx].append({
            "val_loss": loss,
            "pred": pred.detach(),
            "actual": actual.detach(),
            "dataset_idx": dataloader_idx,
        })
        return
        
    def on_validation_epoch_end(self):
        # results_by_dataset = self.split_results_by_dataset(outputs)
        
        tensorboard_logs = {}
        outputs = self.validation_outputs
        self.validation_outputs = [[] for _ in range(len(outputs))]
        for dataset_idx, batch_outputs in enumerate(outputs):
            dataset = self.hparams.dataset_names[dataset_idx]
            preds = torch.cat([x["pred"] for x in batch_outputs])
            actuals = torch.cat([x["actual"] for x in batch_outputs])
            avg_loss = torch.stack([x["val_loss"] for x in batch_outputs]).mean()
            val_loss = self.loss(preds, actuals)
            
            accuracy, roc_auc, recall, F1, precision, mcc, auprc, bacc, sp, se  = self.epoch_end(batch_outputs)
            # print(dataset,':rocauc:', roc_auc, '\tacc:', accuracy, '\tloss:', val_loss)
            
            self.log(dataset + "_loss", avg_loss, on_epoch=True, prog_bar=True)
            self.log(dataset + "_acc", accuracy, on_epoch=True, prog_bar=False)
            self.log(dataset + "_rocauc", roc_auc, on_epoch=True, prog_bar=True)
            self.log(dataset + "_recall", recall, on_epoch=True, prog_bar=False)
            self.log(dataset + "_F1", F1, on_epoch=True, prog_bar=False)
            self.log(dataset + "_precision", precision, on_epoch=True, prog_bar=False)
            self.log(dataset + "_mcc", mcc, on_epoch=True, prog_bar=False)
            self.log(dataset + "_bacc", bacc, on_epoch=True, prog_bar=False)
            self.log(dataset + "_auprc", auprc, on_epoch=True, prog_bar=False)
            self.log(dataset + "_sp", sp, on_epoch=True, prog_bar=False)
            self.log(dataset + "_se", se, on_epoch=True, prog_bar=False)
            if self.max_rocauc["max_"+ dataset + "_rocauc"] < roc_auc:
                self.max_rocauc["max_"+ dataset + "_rocauc"] = roc_auc
                self.log("max_"+ dataset + "_rocauc", roc_auc, on_epoch=True, prog_bar=False)            
        # print("Validation: Current Epoch", self.current_epoch)
                
            # self.validation_outputs[dataset_idx].clear()
        return {"avg_val_loss": avg_loss}
    
    def epoch_end(self, batch_outputs):
        preds = torch.cat([x["pred"] for x in batch_outputs])
        preds = F.softmax(preds, dim=1)
        preds_probs=preds[:,1]
        preds = preds.argmax(dim=1)
        actuals = torch.cat([x["actual"] for x in batch_outputs])
      
        tn, fp, fn, tp = metrics.confusion_matrix(actuals.cpu().numpy(), preds.cpu().numpy()).ravel()   
        sp_score=tn/(tn+fp)
        se_score=tp/(tp+fn)
        bacc_score=(se_score+sp_score)/2      
  
        bacc=0
        sp=0
        se=0         
        bacc=bacc_score
        sp=sp_score
        se=se_score
       
        if self.hparams.num_classes==2:
            Accuracy = classification.Accuracy(task="binary", num_classes=2).to(self.hparams.device)
            Recall = classification.Recall(task="binary", num_classes=2).to(self.hparams.device)
            F1Score = classification.F1Score(task="binary", num_classes=2).to(self.hparams.device)
            Precision = classification.Precision(task="binary", num_classes=2).to(self.hparams.device)
            AUROC = classification.AUROC(task="binary", num_classes=2).to(self.hparams.device)
            Mcc = classification.MatthewsCorrCoef(task="binary", num_classes=2, threshold=0.5).to(self.hparams.device)
            Auprc = classification.AveragePrecision(task="binary", num_classes=2).to(self.hparams.device)
            
        else:
            Accuracy = classification.Accuracy(task='multiclass', num_classes=self.hparams.num_classes).to(self.hparams.device)
            Recall = classification.Recall(task='multiclass', num_classes=self.hparams.num_classes).to(self.hparams.device)
            F1Score = classification.F1Score(task='multiclass', num_classes=self.hparams.num_classes).to(self.hparams.device)
            Precision = classification.Precision(task='multiclass', num_classes=self.hparams.num_classes).to(self.hparams.device)
            AUROC = classification.AUROC(task='multiclass', num_classes=self.hparams.num_classes).to(self.hparams.device)
            Mcc = classification.MatthewsCorrCoef(task='multiclass', num_classes=self.hparams.num_classes).to(self.hparams.device)
            Average_Precision = classification.AveragePrecision(task="binary", num_classes=2).to(self.hparams.device)
                           
        accuracy = Accuracy(preds, actuals)
        recall = Recall(preds, actuals)
        F1 = F1Score(preds, actuals)
        precision = Precision(preds, actuals)
        roc_auc = AUROC(preds, actuals)
        mcc = Mcc(preds, actuals)
        auprc = Auprc(preds_probs, actuals)
        
        return accuracy, roc_auc, recall, F1, precision, mcc, auprc, bacc, sp, se

def get_dataset(data_root, filename,  dataset_len, aug, measure_name):
    df = pd.read_csv(os.path.join(data_root, filename))
    print("Length of dataset:", len(df))
    
    if dataset_len:
        df = df.head(dataset_len)
        print("Warning entire dataset not used:", len(df))
    dataset = PropertyPredictionDataset(df=df, measure_name=measure_name, aug=aug)
    return dataset

class PropertyPredictionDataset(torch.utils.data.Dataset):
    def __init__(self, df, measure_name, tokenizer=None, aug=False):
        df = df[['canonical_smiles', measure_name]]
        df = df.dropna()
        self.measure_name = measure_name
        # if aug:
        #     df['canonical_smiles'] = df['smiles'].apply(lambda smi: normalize_smiles(smi, canonical=True, isomeric=False))
        #     df['canonical_smiles'] = df['canonical_smiles'].apply(lambda smi: randomize_smiles(smi))
        #     # df['canonical_smiles'] = df['canonical_smiles'].apply(lambda smi: normalize_smiles(smi, canonical=True, isomeric=False))
        # else:
        #     df['canonical_smiles'] = df['smiles'].apply(lambda smi: normalize_smiles(smi, canonical=True, isomeric=False))
        df_good = df.dropna(subset=['canonical_smiles'])  # TODO - Check why some rows are na

        len_new = len(df_good)
        print('Dropped {} invalid smiles'.format(len(df) - len_new))
        self.df = df_good
        self.df = self.df.reset_index(drop=True)

    def __getitem__(self, index):
        canonical_smiles = self.df.loc[index, 'canonical_smiles']
        measures = self.df.loc[index, self.measure_name]
        return canonical_smiles, measures

    def __len__(self):
        return len(self.df)

class PropertyPredictionDataModule(pl.LightningDataModule):
    def __init__(self, hparams):
        super(PropertyPredictionDataModule, self).__init__()
        # if type(hparams) is dict:
        #     hparams = Namespace(**hparams)
        self.save_hyperparameters(hparams)
        self.smiles_emb_size = hparams.n_embd
        self.tokenizer = MolTranBertTokenizer('bert_vocab.txt')
        self.dataset_name = hparams.dataset_name

    def get_split_dataset_filename(dataset_name, split):
        return split + ".csv"

    def prepare_data(self):
        print("Inside prepare_dataset")
        self.train_filename = PropertyPredictionDataModule.get_split_dataset_filename(
            self.dataset_name, "train"
        )

        valid_filename = PropertyPredictionDataModule.get_split_dataset_filename(
            self.dataset_name, "valid"
        )

        test_filename = PropertyPredictionDataModule.get_split_dataset_filename(
            self.dataset_name, "test"
        )

        train_ds = get_dataset(
            self.hparams.data_root,
            self.train_filename,
            self.hparams.train_dataset_length,
            aug = True,
            measure_name=self.hparams.measure_name,
        )

        val_ds = get_dataset(
            self.hparams.data_root,
            valid_filename,
            self.hparams.eval_dataset_length,
            aug=False,
            measure_name=self.hparams.measure_name,
        )

        test_ds = get_dataset(
            self.hparams.data_root,
            test_filename,
            self.hparams.eval_dataset_length,
            aug=False,
            measure_name=self.hparams.measure_name,
        )

        self.train_ds = train_ds
        self.val_ds = [val_ds] + [test_ds]

    def collate(self, batch):
        tokens = self.tokenizer.batch_encode_plus([ smile[0] for smile in batch], padding=True, add_special_tokens=True)
        
        return (torch.tensor(tokens['input_ids']), torch.tensor(tokens['attention_mask']), torch.tensor([smile[1] for smile in batch]))

    def val_dataloader(self):
        return [
            DataLoader(
                ds,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                shuffle=True,
                collate_fn=self.collate,
            )
            for ds in self.val_ds
        ]

    def train_dataloader(self):
        self.train_ds = get_dataset(
            self.hparams.data_root,
            self.train_filename,
            self.hparams.train_dataset_length,
            aug = True,
            measure_name=self.hparams.measure_name,
        )
        return DataLoader(
            self.train_ds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            collate_fn=self.collate,
        )


def main():
    margs = args.parse_args()
    print("Using " + str(
        torch.cuda.device_count()) + " GPUs---------------------------------------------------------------------")
    pos_emb_type = 'rot'
    print('pos_emb_type is {}'.format(pos_emb_type))

    run_name_fields = [
        margs.dataset_name,
        margs.measure_name,
        pos_emb_type,
        margs.fold,
        margs.mode,
        "lr",
        margs.lr_start,
        "batch",
        margs.batch_size,
        "drop",
        margs.dropout,
        margs.dims,
    ]
    run_name = "_".join(map(str, run_name_fields))

    print(run_name)
    datamodule = PropertyPredictionDataModule(margs)
    margs.dataset_names = "valid test".split()
    margs.run_name = run_name

    checkpoints_folder = margs.checkpoints_folder
    checkpoint_root = os.path.join(checkpoints_folder, margs.measure_name)
    margs.checkpoint_root = checkpoint_root
    margs.run_id=np.random.randint(30000)
    os.makedirs(checkpoints_folder, exist_ok=True)
    checkpoint_dir = os.path.join(checkpoint_root, "models_"+str(margs.run_id))
    results_dir = os.path.join(checkpoint_root, "results")
    margs.results_dir = results_dir
    margs.checkpoint_dir = checkpoint_dir
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    

    checkpoint_path = os.path.join(checkpoints_folder, margs.measure_name)
    # checkpoint_callback = pl.callbacks.ModelCheckpoint(period=1, save_last=True, dirpath=checkpoint_dir, filename='checkpoint', verbose=True)

    print(margs)

    logger = TensorBoardLogger(
        save_dir=checkpoint_root,
        version="lr:"+str(margs.lr_start)+"_decay:"+str(margs.weight_decay),
        name="lightning_logs",
        default_hp_metric=False,
    )

    tokenizer = MolTranBertTokenizer('bert_vocab.txt')
    pl.seed_everything(margs.seed)

    # Load pre-trained model
    if margs.seed_path == '':
        print("# training from scratch")
        model = LightningModule(margs, tokenizer)
    else:
        print("# loaded pre-trained model from {args.seed_path}")
        model = LightningModule(margs, tokenizer).load_from_checkpoint(margs.seed_path, strict=False, config=margs, tokenizer=tokenizer, vocab=len(tokenizer.vocab))


    last_checkpoint_file = os.path.join(checkpoint_dir, "last.ckpt")
    resume_from_checkpoint = None
    if os.path.isfile(last_checkpoint_file):
        print(f"resuming training from : {last_checkpoint_file}")
        resume_from_checkpoint = last_checkpoint_file
    else:
        print(f"training from scratch")

    early_stop_callback = EarlyStopping(
        monitor='valid_loss',
        min_delta=0.00,
        patience=5,
        verbose=True,
        mode='min'
    )
    model_checkpoint = ModelCheckpoint(
    monitor='valid_loss',
    filename='model-{epoch:02d}-{val_loss:.2f}',
    save_top_k=1,
    mode='min',
    verbose=True
    )

    trainer = pl.Trainer(
        callbacks=[model_checkpoint],
        max_epochs=margs.max_epochs,
        default_root_dir=checkpoint_root,
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,
        logger=logger,
        num_sanity_val_steps=0,
        accumulate_grad_batches=5,
        gradient_clip_val=0.1,
        sync_batchnorm=True,
        # log_every_n_steps=8,
        )
    
    if margs.froze_transformer:
        print("Freeze transformer")
        for _, param in model.blocks.named_parameters():
            param.requires_grad = False
            
    tic = time.perf_counter()
    
    # Train the model ⚡
    trainer.fit(
        model, 
        datamodule, 
        # ckpt_path=margs.seed_path,
        )
    toc = time.perf_counter()
    print('Time was {}'.format(toc - tic))


if __name__ == '__main__':
    main()
