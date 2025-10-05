import torch
import torch.nn.functional as F
import numpy as np
import time

import os
import json
import glob
from collections import OrderedDict
from tqdm import tqdm
from typing import Dict, Any, List, Optional
import torch.distributed as dist
from torch.utils.data import Subset
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
import io
import zipfile
import gc
import inspect


from nn.state_space_llm import apply_lora_to_model
from fairscale.nn.model_parallel.layers import RowParallelLinear, ColumnParallelLinear


def print_tensor_memory(tensor, name="tensor"):
    """Print memory usage of a tensor"""
    if torch.cuda.is_available() and tensor.is_cuda:
        size_mb = tensor.element_size() * tensor.numel() / 1024**2
        print(f"  {name}: {tensor.shape} -> {size_mb:.2f} MB")
        if tensor.requires_grad:
            print(f"    (requires_grad=True - will store gradients!)")


def count_occurence(x,y):
  coord_counts = {}
  for i in range(len(x)):
      coord = (x[i], y[i])
      if coord in coord_counts:
          coord_counts[coord] += 1
      else:
          coord_counts[coord] = 1


class Trainer(object):
    """
    A class that encapsulates the training loop for a PyTorch model.
    """
    def __init__(self, model, optimizer, criterion, train_dataloader, device, world_size=1, output_dim=2,
                 scheduler=None, val_dataloader=None,   max_iter=-1, scaler=None, use_amp=False,
                  grad_clip=False, max_grad_norm=1, log_path=None, exp_name=None, plot_every=None,
                  cos_inc=False, range_update=None, accumulation_step=1, wandb_log=False, num_quantiles=1,
                  update_func=lambda x: x, validation_interval=None):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scaler = scaler
        self.use_amp = use_amp
        self.max_grad_norm = max_grad_norm
        self.grad_clip = grad_clip
        self.cos_inc = cos_inc
        self.output_dim = output_dim
        self.scheduler = scheduler
        self.train_dl = train_dataloader
        self.val_dl = val_dataloader
        self.train_sampler = self.get_sampler_from_dataloader(train_dataloader)
        self.val_sampler = self.get_sampler_from_dataloader(val_dataloader)
        self.max_iter = max_iter
        self.device = device
        self.world_size = world_size
        self.exp_name = exp_name
        self.log_path = log_path
        self.best_state_dict = self.model.state_dict()
        self.plot_every = plot_every
        self.logger = None
        self.range_update = range_update
        self.accumulation_step = accumulation_step
        self.wandb = wandb_log
        self.num_quantiles = num_quantiles
        self.update_func = update_func
        self.epoch = 0
        self.validation_interval = validation_interval
        # if log_path is not None:
        #     self.logger =SummaryWriter(f'{self.log_path}/exp{self.exp_num}')
        #     # print(f"logger path: {self.log_path}/exp{self.exp_num}")

        # print("logger is: ", self.logger)
    
    def get_sampler_from_dataloader(self, dataloader):
        if hasattr(dataloader, 'sampler'):
            if isinstance(dataloader.sampler, torch.utils.data.DistributedSampler):
                return dataloader.sampler
            elif hasattr(dataloader.sampler, 'sampler'):
                return dataloader.sampler.sampler
        
        if hasattr(dataloader, 'batch_sampler') and hasattr(dataloader.batch_sampler, 'sampler'):
            return dataloader.batch_sampler.sampler
        
        return None
    
    def fit(self, num_epochs, device,  early_stopping=None, start_epoch=0, best='loss', conf=False,
            initial_min_loss=None, initial_best_acc=None):
        """
        Fits the model for the given number of epochs.
        """
        min_loss = float(initial_min_loss) if initial_min_loss is not None else np.inf
        best_acc = float(initial_best_acc) if initial_best_acc is not None else 0.0
        train_loss, val_loss,  = [], []
        train_acc, val_acc = [], []
        lrs = []
        epochs = []
        self.train_aux_loss_1 = []
        self.train_aux_loss_2 = []
        self.train_aux_loss_3 = []
        self.val_aux_loss_1 = []
        self.val_aux_loss_2 = []
        self.val_aux_loss_3 = []
        self.train_logits_mean = []
        self.train_logits_std = []
        self.val_logits_mean = []
        self.val_logits_std = []
        # self.optim_params['lr_history'] = []
        epochs_without_improvement = 0
        main_proccess = (torch.distributed.is_initialized() and torch.distributed.get_rank() == 0) or self.device == 'cpu'

        print(f"Starting training for {num_epochs} epochs")
        print("is main process: ", main_proccess, flush=True)
        global_time = time.time()
        self.epoch = 0
        for epoch in range(start_epoch, start_epoch + num_epochs):
            epochs.append(epoch)
            self.epoch = epoch
            start_time = time.time()
            plot = (self.plot_every is not None) and (epoch % self.plot_every == 0)
            t_loss, t_acc = self.train_epoch(device, epoch=epoch)
            t_loss_mean = np.nanmean(t_loss)
            train_loss.extend(t_loss)
            global_train_accuracy, global_train_loss = self.process_loss(t_acc, t_loss_mean)
            if main_proccess:  # Only perform this on the master GPU
                train_acc.append(global_train_accuracy.mean().item())
                
            v_loss, v_acc = self.eval_epoch(device, epoch=epoch)
            v_loss_mean = np.nanmean(v_loss)
            val_loss.extend(v_loss)
            global_val_accuracy, global_val_loss = self.process_loss(v_acc, v_loss_mean)
            if main_proccess:  # Only perform this on the master GPU                
                val_acc.append(global_val_accuracy.mean().item())
                
                current_objective = global_val_loss if best == 'loss' else global_val_accuracy.mean()
                improved = False
                
                if best == 'loss':
                    if current_objective < min_loss:
                        min_loss = current_objective
                        improved = True
                else:
                    if current_objective > best_acc:
                        best_acc = current_objective
                        improved = True
                
                if improved and (not dist.is_initialized() or dist.get_rank() == 0):
                    # Save best composite checkpoint (rank 0 only)
                    model_name = f'{self.log_path}/{self.exp_name}.pth'
                    resume_best = f'{self.log_path}/{self.exp_name}_resume_best.pth'
                    print(f"saving model at {model_name}...")
                    try:
                        # atomic save: write temp then move
                        tmp_model = model_name + '.tmp'
                        torch.save(self.model.state_dict(), tmp_model)
                        os.replace(tmp_model, model_name)
                        self.best_state_dict = self.model.state_dict()
                    except Exception as e:
                        print(f"Warning saving state_dict only: {e}")
                    try:
                        tmp_best = resume_best + '.tmp'
                        torch.save({
                            'epoch': epoch,
                            'model': self.model.state_dict(),
                            'optimizer': self.optimizer.state_dict() if self.optimizer is not None else None,
                            'scheduler': self.scheduler.state_dict() if self.scheduler is not None else None,
                            'scaler': self.scaler.state_dict() if self.scaler is not None else None,
                            'min_loss': float(min_loss) if isinstance(min_loss, np.generic) else min_loss,
                            'best_acc': float(best_acc) if isinstance(best_acc, np.generic) else best_acc,
                        }, tmp_best)
                        os.replace(tmp_best, resume_best)
                    except Exception as e:
                        print(f"Warning saving best composite checkpoint: {e}")
                    # model_path, output_filename = save_compressed_checkpoint(
                    #                            self.model, model_name, res, use_zip=True )
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                res = {"epochs": epochs, "train_loss": train_loss, "val_loss": val_loss,
                        "train_acc": train_acc, "val_acc": val_acc, "train_aux_loss_1": self.train_aux_loss_1,
                        "train_aux_loss_2":self.train_aux_loss_2, "train_aux_loss_3":self.train_aux_loss_3,
                         "val_aux_loss_1":self.val_aux_loss_1, "val_aux_loss_2": self.val_aux_loss_2,
                          "val_aux_loss_3": self.val_aux_loss_3, "train_logits_mean": self.train_logits_mean,
                         "train_logits_std": self.train_logits_std, "val_logits_mean": self.val_logits_mean,
                          "val_logits_std": self.val_logits_std, "lrs": lrs}

                current_lr = self.optimizer.param_groups[0]['lr'] if self.scheduler is None \
                            else self.scheduler.get_last_lr()[0]
                
                lrs.append(current_lr)
                
                if (not dist.is_initialized()) or dist.get_rank() == 0:
                    output_filename = f'{self.log_path}/{self.exp_name}.json'
                    tmp_json = output_filename + '.tmp'
                    with open(tmp_json, "w") as f:
                        json.dump(res, f, indent=2)
                    os.replace(tmp_json, output_filename)
                    print(f"saved results at {output_filename}")
                
                print(f'Epoch {epoch}, lr {current_lr}, Train Loss: {global_train_loss:.6f}, Val Loss:'\
                
                        f'{global_val_loss:.6f}, Train Acc: {global_train_accuracy.round(decimals=4).tolist()}, '\
                f'Val Acc: {global_val_accuracy.round(decimals=4).tolist()},'\
                  f'Time: {time.time() - start_time:.2f}s, Total Time: {(time.time() - global_time)/3600} hr', flush=True)
                if ((not dist.is_initialized()) or dist.get_rank() == 0) and (epoch % 10 == 0):
                    print(os.system('nvidia-smi'))

                if epochs_without_improvement == early_stopping:
                    print('early stopping!', flush=True)
                    break
                if time.time() - global_time > (23.83 * 3600):
                    print("time limit reached")
                    break 

            # Always save last composite checkpoint to allow exact resume (rank 0 only)
            if (not dist.is_initialized()) or dist.get_rank() == 0:
                try:
                    last_path = f'{self.log_path}/{self.exp_name}_resume_last.pth'
                    tmp_last = last_path + '.tmp'
                    torch.save({
                        'epoch': epoch,
                        'model': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict() if self.optimizer is not None else None,
                        'scheduler': self.scheduler.state_dict() if self.scheduler is not None else None,
                        'scaler': self.scaler.state_dict() if self.scaler is not None else None,
                        'min_loss': float(min_loss) if isinstance(min_loss, np.generic) else min_loss,
                        'best_acc': float(best_acc) if isinstance(best_acc, np.generic) else best_acc,
                    }, tmp_last)
                    os.replace(tmp_last, last_path)
                except Exception as e:
                    print(f"Warning saving last composite checkpoint: {e}")

        return {"epochs":epochs, "train_loss": train_loss,
                 "val_loss": val_loss, "train_acc": train_acc,
                "val_acc": val_acc, "train_aux_loss_1": self.train_aux_loss_1,
                "train_aux_loss_2":self.train_aux_loss_2, "train_aux_loss_3":self.train_aux_loss_3,
                    "val_aux_loss_1":self.val_aux_loss_1, "val_aux_loss_2": self.val_aux_loss_2,
                    "val_aux_loss_3": self.val_aux_loss_3, "train_logits_mean": self.train_logits_mean,
                 "train_logits_std": self.train_logits_std, "val_logits_mean": self.val_logits_mean,
                  "val_logits_std": self.val_logits_std, "lrs": lrs}

    def process_loss(self, acc, loss_mean):
        if  torch.cuda.is_available() and torch.distributed.is_initialized():
            global_accuracy = torch.tensor(acc).cuda()  # Convert accuracy to a tensor on the GPU
            torch.distributed.reduce(global_accuracy, dst=0, op=torch.distributed.ReduceOp.SUM)
            global_loss = torch.tensor(loss_mean).cuda()  # Convert loss to a tensor on the GPU
            torch.distributed.reduce(global_loss, dst=0, op=torch.distributed.ReduceOp.SUM)
            
            # Divide both loss and accuracy by world size
            world_size = torch.distributed.get_world_size()
            global_loss /= world_size
            global_accuracy /= world_size
        else:
            global_loss = torch.tensor(loss_mean)
            global_accuracy = torch.tensor(acc)
        return global_accuracy, global_loss

    def load_best_model(self, to_ddp=True, from_ddp=True):
        data_dir = f'{self.log_path}/exp{self.exp_num}'
        # data_dir = f'{self.log_path}/exp29' # for debugging

        state_dict_files = glob.glob(data_dir + '/*.pth')
        print("loading model from ", state_dict_files[-1])
        
        state_dict = torch.load(state_dict_files[-1]) if to_ddp else torch.load(state_dict_files[0],map_location=self.device)
    
        if from_ddp:
            print("loading distributed model")
            # Remove "module." from keys
            new_state_dict = OrderedDict()
            for key, value in state_dict.items():
                if key.startswith('module.'):
                    while key.startswith('module.'):
                        key = key[7:]
                new_state_dict[key] = value
            state_dict = new_state_dict
        # print("state_dict: ", state_dict.keys())
        # print("model: ", self.model.state_dict().keys())

        self.model.load_state_dict(state_dict, strict=False)

    def check_gradients(self):
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                if grad_norm > 100:
                   print(f"Large gradient in {name}: {grad_norm}")

    def train_epoch(self, device, epoch):
        """
        Trains the model for one epoch.
        """
        if self.train_sampler is not None:
            try:
                self.train_sampler.set_epoch(epoch)
            except AttributeError:
                pass
        self.model.train()
        train_loss = []
        train_acc = 0
        total = 0
        all_accs = torch.zeros(self.output_dim, device=device)
        total_batches = len(self.train_dl)
        if self.max_iter is not None and self.max_iter >= 0:
            total_batches = min(total_batches, self.max_iter + 1)
        pbar = tqdm(self.train_dl, total=total_batches)
        for i, batch in enumerate(pbar):
            if self.optimizer is not None:
                self.optimizer.zero_grad()
            loss, acc , y = self.train_batch(batch, i + epoch * len(self.train_dl), device)
            train_loss.append(loss.item())
            all_accs = all_accs + acc
            total += len(y)
            pbar.set_description(f"train_acc: {acc}, train_loss:  {loss.item():.4f}")      
            if (self.max_iter is not None) and (self.max_iter >= 0) and (i >= self.max_iter):
                break

            if self.validation_interval and (i + 1) % self.validation_interval == 0:
                self._run_mid_epoch_validation(epoch, i + 1, device)
        print("number of train_accs: ", all_accs, "total: ", total)
        return train_loss, all_accs/total
    
    def train_batch(self, batch, batch_idx, device):
        pass

    def eval_epoch(self, device, epoch):
        """
        Evaluates the model for one epoch.
        """
        self.model.eval()
        val_loss = []
        val_acc = 0
        total = 0
        all_accs = torch.zeros(self.output_dim, device=device)
        total_batches = len(self.val_dl)
        if self.max_iter is not None and self.max_iter >= 0:
            total_batches = min(total_batches, self.max_iter + 1)
        pbar = tqdm(self.val_dl, total=total_batches)
        for i,batch in enumerate(pbar):
            loss, acc, y = self.eval_batch(batch, i + epoch * len(self.val_dl), device)
            val_loss.append(loss.item())
            all_accs = all_accs + acc
            total += len(y)
            pbar.set_description(f"val_acc: {acc}, val_loss:  {loss.item():.4f}")
            if (self.max_iter is not None) and (self.max_iter >= 0) and (i >= self.max_iter):
                break

        if dist.is_initialized():
            torch.cuda.synchronize()
        self.model.train()

        avg_acc = all_accs / total if total > 0 else all_accs
        return val_loss, avg_acc

    def _run_mid_epoch_validation(self, epoch: int, iteration: int, device):
        if self.val_dl is None or self.validation_interval is None:
            return
        self.model.eval()
        losses = []
        sample_examples = []
        with torch.no_grad():
            for i, batch in enumerate(self.val_dl):
                mode = batch.get('mode', self.mode)
                self._set_active_mode(mode)
                loss, _, _ = self.eval_batch(batch, i + epoch * len(self.val_dl), device)
                losses.append(loss.item())
                if len(losses) <= 3:
                    sample = self._collect_validation_sample(batch, device)
                    if sample:
                        sample_examples.append(sample)
                if i >= 9:
                    break

        if not losses:
            return

        avg_loss = float(np.mean(losses))
        if self.log_path and self.exp_name:
            interim_path = os.path.join(
                self.log_path,
                f"{self.exp_name}_midval_epoch{epoch}_iter{iteration}.json"
            )
            with open(interim_path, 'w') as fh:
                json.dump({
                    'epoch': epoch,
                    'iteration': iteration,
                    'avg_val_loss': avg_loss,
                    'num_batches': len(losses),
                    'samples': sample_examples
                }, fh, indent=2)

        if dist.is_initialized():
            torch.cuda.synchronize()
        self.model.train()

    def _collect_validation_sample(self, batch: Dict[str, Any], device) -> Optional[Dict[str, Any]]:
        try:
            tokenizer = getattr(self, 'tokenizer', None)
            model = self.model.module if hasattr(self.model, 'module') else self.model
            batch_idx = 0
            generated_text, input_text, target_text, _ = model.generate_response_from_batch(
                batch_data=batch,
                batch_idx=batch_idx,
                tokenizer=tokenizer,
                max_new_tokens=50,
                temperature=0.2,
                top_p=0.8,
            )

            if not input_text:
                input_text = batch.get('input_texts', [''])[batch_idx] if batch.get('input_texts') else ''
                if not input_text:
                    input_text = batch.get('question_texts', [''])[batch_idx] if batch.get('question_texts') else ''

            if not target_text:
                target_text = batch.get('target_texts', [''])[batch_idx] if batch.get('target_texts') else ''

            obsid_key = 'obsids' if 'obsids' in batch else 'obsid'
            obsid_val = None
            if obsid_key in batch:
                obs_entry = batch[obsid_key][batch_idx] if isinstance(batch[obsid_key], (list, tuple)) else batch[obsid_key]
                if isinstance(obs_entry, torch.Tensor):
                    obsid_val = obs_entry.item() if obs_entry.numel() == 1 else obs_entry.tolist()
                else:
                    obsid_val = obs_entry

            sample = {
                'mode': batch.get('mode', getattr(model, 'mode', 'single_star')),
                'obsid': obsid_val,
                'question': input_text,
                'true_answer': target_text,
                'generated_answer': generated_text,
            }
            return sample
        except Exception as err:
            print(f"Warning: failed to collect validation sample ({err})")
            return None

    def eval_batch(self, batch, batch_idx, device):
        pass

    def predict(self, test_dataloader, device, load_best=True):
        """
        Returns the predictions of the model on the given dataset.
        """
        pass

class MaskedRegressorTrainer(Trainer):
    def __init__(self, w_name, w_init_val, ssl_criterion, ssl_weight=0.5, **kwargs):
        super().__init__(**kwargs)
        self.w_name = w_name
        self.ssl_criterion = ssl_criterion
        self.w_init_val = w_init_val
        self.ssl_weight = ssl_weight  # Weight to balance between SSL and regression
        self.drop_first_y = False
        
    def train_batch(self, batch, batch_idx, device):
        x_masked, x, y, mask, _, info= batch
        x_masked, x, y, mask = x_masked.to(device), x.to(device), y.to(device), mask.to(device)
        b = x_masked.shape[0]
        
        if self.w_name is None:
            w = torch.ones(x_masked.size(0)).to(device)
        else:
            w = torch.tensor([i[self.w_name] for i in info]).to(device)
        # Forward pass for both tasks
        reg_out,ssl_out,features = self.model(x_masked, x)

        # Calculate SSL loss (masked filling)
        if (len(x.shape) == 3) and (x.shape[1] > 1):
            x = x[:, 0, :]
        ssl_loss = self.ssl_criterion(ssl_out, x)
        ssl_acc = self.mask_accuracy(ssl_out, x, mask)

        
        # Calculate regression loss
        reg_out = reg_out.view(b, -1, self.num_quantiles)
        out_diff = int(y.shape[1] - reg_out.shape[1])
        y = y[:, out_diff:]
        if self.drop_first_y:
            reg_out = reg_out[:, 1:]
            y = y[:, 1:]
        reg_loss = self.criterion(reg_out, y)
        reg_loss = (reg_loss * w.unsqueeze(-1)).mean()
        out_median = reg_out[..., reg_out.shape[-1]//2]
        reg_acc = (torch.abs(out_median - y) < y * 0.1).sum(0)
        self.train_aux_loss_1.append(ssl_loss.item())
        self.train_aux_loss_2.append(reg_loss.item())
        # Combine losses
        loss = (self.ssl_weight * ssl_loss) + ((1 - self.ssl_weight) * reg_loss)
        
         # Backward pass with gradient scaling
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            if (batch_idx + 1) % self.accumulation_step == 0:
                self.scaler.unscale_(self.optimizer)
                if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                if self.scheduler is not None:
                    self.scheduler.step()
                self.check_gradients()  # Monitor gradients
        else:
            loss.backward()
            if (batch_idx + 1) % self.accumulation_step == 0:
                if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                self.check_gradients()  # Monitor gradients
                
        return loss, reg_acc, x

    def eval_batch(self, batch, batch_idx, device):
        x_masked, x, y, mask, _, info = batch
        x_masked, x, y, mask = x_masked.to(device), x.to(device), y.to(device), mask.to(device)
        b = x_masked.shape[0]
        
        if self.w_name is None:
            w = torch.ones(x_masked.size(0)).to(device)
        else:
            w = torch.tensor([i[self.w_name] for i in info]).to(device)

        with torch.no_grad():
            reg_out,ssl_out,features = self.model(x_masked, x)  # Masked filling task
            
            ssl_loss = self.ssl_criterion(ssl_out, x)
            ssl_acc = self.mask_accuracy(ssl_out, x, mask)

            reg_out = reg_out.view(b, -1, self.num_quantiles)
            reg_out = reg_out.view(b, -1, self.num_quantiles)
            out_diff = int(y.shape[1] - reg_out.shape[1])
            y = y[:, out_diff:]
            if self.drop_first_y:
                reg_out = reg_out[:, 1:]
                y = y[:, 1:]
            reg_loss = self.criterion(reg_out, y)
            reg_loss = (reg_loss * w.unsqueeze(-1)).mean()
            out_median = reg_out[..., reg_out.shape[-1]//2]
            reg_acc = (torch.abs(out_median - y) < y * 0.1).sum(0)
            self.val_aux_loss_1.append(ssl_loss.item())
            self.val_aux_loss_2.append(reg_loss.item())

            total_loss = (self.ssl_weight * ssl_loss) + ((1 - self.ssl_weight) * reg_loss)
            
        return total_loss, reg_acc, x
        
    def mask_accuracy(self, result, target, inverse_token_mask, epsilon=1e-5):
        r = result.masked_select(inverse_token_mask)
        t = target.masked_select(inverse_token_mask)
        s = (torch.abs(r - t) < epsilon).sum()
        return s / inverse_token_mask.sum()

    def predict(self, test_dataloader, device):
        """
        Returns the predictions of the model on the given dataset.
        """
        self.model.eval()
        preds = np.zeros((0, self.output_dim, self.num_quantiles))
        targets = np.zeros((0, self.output_dim))
        all_features = []
        all_xs = []
        all_decodes = []
        all_wv = []
        aggregated_info = {}
        pbar = tqdm(test_dataloader)

        for i,(x_masked, x, y, mask, _ , info) in enumerate(pbar):
            x_masked, x, y, mask = x_masked.to(device), x.to(device), y.to(device), mask.to(device)
            b = x_masked.shape[0]
            for item in info:
                for key, value in item.items():
                    # Check if value is a scalar (not an array/tensor)
                    if np.isscalar(value):
                        if key not in aggregated_info:
                            aggregated_info[key] = []
                        aggregated_info[key].append(value)
            if self.w_name is None:
                w = torch.ones(x_masked.size(0)).to(device)
            else:
                w = torch.tensor([i[self.w_name] for i in info]).to(device)
            with torch.no_grad():
                y_pred, decod_out, features = self.model(x_masked, x)
                y_pred = y_pred.view(b, -1, self.num_quantiles)
            out_diff = int(y.shape[1] - y_pred.shape[1])
            y = y[:, out_diff:]
            if self.drop_first_y:
                reg_out = reg_out[:, 1:]
                y = y[:, 1:]
            ssl_loss = self.ssl_criterion(decod_out, x)
            reg_loss = self.criterion(y_pred, y)
            reg_loss = (reg_loss * w.unsqueeze(-1)).mean()
            out_median = y_pred[..., y_pred.shape[-1]//2]
            reg_acc = (torch.abs(out_median - y) < y * 0.1).sum(0) / b
            total_loss = (self.ssl_weight * ssl_loss) + ((1 - self.ssl_weight) * reg_loss)

            pbar.set_description(f"test_loss: {total_loss.item():.4f}, test_acc %: {reg_acc}")

            # if i % 100 == 0:
            #     decode_sample = decod_out[0].cpu().numpy()
            #     plt.plot(x[0].cpu().numpy(), alpha=0.5, label='Original Sample')
            #     plt.plot(decode_sample, alpha=0.3, label='Decoded Sample')
            #     plt.legend()
            #     plt.savefig(f'images/masked_regressor_decode_sample_{i}.png')
            #     plt.close()
            #
            preds = np.concatenate((preds, y_pred.cpu().numpy()))
            targets = np.concatenate((targets, y.cpu().numpy()))
            all_features.append(features.cpu().numpy())
            all_xs.append(x.cpu().numpy())
            all_decodes.append(decod_out.cpu().numpy())
            
            if (self.max_iter is not None) and (self.max_iter >= 0) and (i > self.max_iter):
                break
        print("target len: ", len(targets), "dataset: ", len(test_dataloader.dataset))
        return preds, targets, np.concatenate(all_features, axis=0),\
        np.concatenate(all_xs, axis=0), np.concatenate(all_decodes, axis=0), \
            aggregated_info


class LLMTrainer(Trainer):

    def __init__(self, lora_params, alpha=1, beta=1, gamma=1, max_chunk_size=128, tokenizer=None, mode="single_star", **kwargs):
        super(LLMTrainer, self).__init__(**kwargs)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.max_chunk_size = max_chunk_size
        self.tokenizer = tokenizer
        self.mode = mode  # "single_star" or "two_star"
        self.freeze_strategy = lora_params['freeze_strategy']
        self.lora_rank = lora_params['lora_rank']
        self.lora_alpha = lora_params['lora_alpha']
        self.lora_dropout = lora_params['lora_dropout']
        self.lora_target_modules = lora_params['lora_target_modules']
        self.lora_start_epoch = lora_params['lora_start_epoch']
        self.lora_modules = None
        self._apply_freeze_strategy(self.freeze_strategy)
        self._set_active_mode(self.mode)

    def _set_active_mode(self, mode: str):
        self.mode = mode
        target_model = self.model.module if hasattr(self.model, 'module') else self.model
        if hasattr(target_model, 'mode'):
            target_model.mode = mode

    def _apply_lora(self):
        """Apply LoRA to the model - fixed version"""
        print("Applying LoRA layers...")

        # Get supported linear layer types
        try:
            linear_types = (torch.nn.Linear, RowParallelLinear, ColumnParallelLinear)
            print("Using FairScale parallel layers support")
        except ImportError:
            linear_types = (torch.nn.Linear,)
            print("FairScale not available, using only torch.nn.Linear")

        # First, let's see what modules actually exist in the model
        print("Available modules in model:")
        all_modules = []
        for name, module in self.model.named_modules():
            if isinstance(module, linear_types):
                all_modules.append(name)

        print(f"Found {len(all_modules)} Linear/Parallel modules:")
        for i, name in enumerate(all_modules[:10]):  # Show first 10
            print(f"  {name}")
        if len(all_modules) > 10:
            print(f"  ... and {len(all_modules) - 10} more")

        # Convert wildcard patterns to actual module names
        target_modules = []
        import re

        for pattern in self.lora_target_modules:
            if '*' in pattern:
                # Convert pattern to regex, handling multiple wildcards
                pattern_regex = pattern.replace('.', r'\.').replace('*', r'[^.]+')
                pattern_regex = f"^{pattern_regex}$"

                # Find matching modules
                matched_modules = []
                for name in all_modules:
                    if re.match(pattern_regex, name):
                        matched_modules.append(name)
                        target_modules.append(name)

                print(f"Pattern '{pattern}' matched {len(matched_modules)} modules")
                if matched_modules:
                    print(f"  Examples: {matched_modules[:3]}")
            else:
                if pattern in all_modules:
                    target_modules.append(pattern)
                    print(f"Direct match: {pattern}")
                else:
                    print(f"Warning: Pattern '{pattern}' not found in model")

        print(f"\nTotal target modules for LoRA: {len(target_modules)}")

        if not target_modules:
            print("ERROR: No target modules found! Check your lora_target_modules patterns.")
            return

        self.lora_modules = apply_lora_to_model(
            self.model,
            target_modules,
            rank=self.lora_rank,
            alpha=self.lora_alpha,
            dropout=self.lora_dropout
        )

        print(f"Successfully applied LoRA to {len(self.lora_modules)} modules")

    def _apply_freeze_strategy(self, strategy: str):
        """Apply different freezing strategies to the model"""
        print(f"Applying freeze strategy: {strategy}")

        if strategy == 'encoder_only':
            # Freeze everything except latent encoder AND latent regressor
            for name, param in self.model.named_parameters():
                if 'base_model' not in name and 'fm_model' not in name:
                    param.requires_grad = True
                    print(f"  ✓ Unfrozen: {name}")
                else:
                    param.requires_grad = False

        elif strategy == 'lora':
            # Apply LoRA if not already applied
            if not self.lora_modules and self.epoch == self.lora_start_epoch:
                self._apply_lora()

            # Freeze base model, enable encoder, regressor, AND LoRA parameters
            for name, param in self.model.named_parameters():
                if 'base_model' not in name and 'fm_model' not in name:
                    param.requires_grad = True
                    print(f"  ✓ Unfrozen : {name}")
                elif 'lora' in name:
                    param.requires_grad = True
                    print(f"  ✓ Unfrozen (LoRA): {name}")
                else:
                    param.requires_grad = False

        elif strategy == 'none':
            # Unfreeze everything
            for param in self.model.parameters():
                param.requires_grad = True

        self.current_freeze_state = strategy

        # Verify critical components are trainable
        critical_components = ['projector', 'projector_a', 'projector_b']
        for component in critical_components:
            component_params = [p for name, p in self.model.named_parameters()
                                if component in name and p.requires_grad]
            if not component_params:
                print(f"ERROR: No trainable parameters in {component}!")
            else:
                total_params = sum(p.numel() for p in component_params)
                print(f"✓ {component}: {len(component_params)} layers, {total_params:,} trainable params")

        # Print what's trainable
        trainable_modules = set()
        trainable_count = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                trainable_count += param.numel()
                if 'base_model' not in name:
                    trainable_modules.add(name)
                elif 'lora' in name:
                    trainable_modules.add('lora')
                else:
                    module_type = name.split('.')[0]
                    trainable_modules.add(f'base_{module_type}')

        # print(f"Trainable module types: {list(trainable_modules)}")
        print(f"Total trainable parameters: {trainable_count:,}")
    
    def train_epoch(self, device, epoch):
        if hasattr(self, 'combined_mode') and self.combined_mode:
            if hasattr(self, 'mixed_train_loader'):
                self.mixed_train_loader.set_epoch(epoch)
                self.train_dl = self.mixed_train_loader
            if hasattr(self, 'mixed_val_loader'):
                self.mixed_val_loader.set_epoch(epoch)
                self.val_dl = self.mixed_val_loader
            # Ensure we start each epoch in single-star mode for consistency
            self._set_active_mode('single_star')

        self._apply_freeze_strategy(self.freeze_strategy)
        return super().train_epoch(device, epoch)

    def get_loss(self, logits, target_ids, attention_mask=None):
        # Shift for autoregressive prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = target_ids[..., 1:].contiguous()

        # shift_logits = logits
        # shift_labels = target_ids
        
        # Only compute loss on actual answer tokens (not -100)
        active_mask = (shift_labels != -100)
        
        if not active_mask.any():
            return torch.tensor(0.0, device=logits.device)
        
        flat_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_labels = shift_labels.view(-1)
        flat_mask = active_mask.view(-1)
        
        # Compute loss only on answer tokens
        active_logits = flat_logits[flat_mask]
        active_labels = flat_labels[flat_mask]
        
        # For two_star mode, weight first 2 tokens more heavily
        if hasattr(self, 'mode') and self.mode == 'two_star':
            # Create weights for all active tokens
            weights = torch.ones_like(active_labels, dtype=torch.float, device=logits.device)
            
            # Find positions of active tokens for each sample in batch
            batch_size, seq_len = active_mask.shape
            weight_multiplier = 3.0  # Make first 2 tokens 3x more important
            
            # Count active tokens per sample to identify first 2 tokens
            for batch_idx in range(batch_size):
                sample_active_mask = active_mask[batch_idx]
                if sample_active_mask.any():
                    # Get indices of active tokens for this sample
                    active_indices = torch.nonzero(sample_active_mask, as_tuple=False).squeeze(-1)
                    if len(active_indices) >= 2:
                        # Weight the first 2 active tokens more heavily
                        first_two_global_indices = batch_idx * seq_len + active_indices[:2]
                        # Find these indices in the flattened active_labels
                        flat_active_indices = torch.nonzero(flat_mask, as_tuple=False).squeeze(-1)
                        for global_idx in first_two_global_indices:
                            local_idx = (flat_active_indices == global_idx).nonzero(as_tuple=False)
                            if len(local_idx) > 0:
                                weights[local_idx[0]] = weight_multiplier
            
            # Compute weighted cross entropy loss
            loss = F.cross_entropy(active_logits, active_labels, reduction='none')
            loss = (loss * weights).mean()
        else:
            # Standard cross entropy loss for non-two_star modes
            loss = F.cross_entropy(active_logits, active_labels, reduction='mean')
        
        # Print diagnostic info
        # print(f"Active tokens: {active_mask.sum().item()}/{active_mask.numel()} "
        #     f"({100*active_mask.sum().item()/active_mask.numel():.1f}%)")
        
        return loss
    
    def get_logits(self, batch, device, val=False):
        # DEBUG: Check input tensors
        input_ids = batch['input_ids'].to(device)
        target_ids = batch['target_ids'].to(device)

        
        # Handle different data structures based on mode
        if self.mode == "two_star":
            star_a_spectra = batch['masked_spectra_a'].to(device)
            star_b_spectra = batch['masked_spectra_b'].to(device)
            star_a_indices = batch['star_a_feature_indices'].to(device)
            star_b_indices = batch['star_b_feature_indices'].to(device)

        else:
            special_token_positions = batch['feature_start_indices'].to(device)
            input_spectra = batch['masked_spectra'].to(device) 
        
        # Forward pass with memory tracking and AMP
        mem_before_forward = torch.cuda.memory_allocated(device) / 1024**3
        with autocast(enabled=getattr(self, 'use_amp', False)):
            # Conditional no_grad
            cm = torch.no_grad() if val else torch.enable_grad()
            with cm:
                if self.mode == "two_star":
                    outputs = self.model(
                        input_ids=input_ids,
                        input_spectra=None,
                        special_token_positions=None,
                        star_a_spectra=star_a_spectra,
                        star_b_spectra=star_b_spectra,
                        star_a_indices=star_a_indices,
                        star_b_indices=star_b_indices,
                    )
                else:
                    outputs = self.model(
                        input_ids=input_ids,
                        input_spectra=input_spectra,
                        special_token_positions=special_token_positions,
                    )

        return outputs


    def train_batch(self, batch, batch_idx, device):
        """Training step with AMP and detailed memory debugging"""

        batch_mode = batch.get('mode', self.mode)
        self._set_active_mode(batch_mode)
        
        # Memory before forward pass
        torch.cuda.empty_cache()
        gc.collect()



        outputs = self.get_logits(batch, device, val=False)
        
        target_ids = batch['target_ids'].to(device)
        # Compute loss
        loss = self.get_loss(outputs['logits'], target_ids)
        
        
        # Backward pass with AMP
        if getattr(self, 'use_amp', False) and hasattr(self, 'scaler'):
            # Scale loss and backward
            self.scaler.scale(loss).backward()
            
            # Gradient clipping (optional)
            if hasattr(self, 'max_grad_norm') and self.max_grad_norm > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            
            # Optimizer step with scaler
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # Standard backward pass
            loss.backward()
            
            # Gradient clipping (optional)
            if hasattr(self, 'max_grad_norm') and self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            
            # Standard optimizer step
            self.optimizer.step()
        
        # Learning rate scheduler
        if self.scheduler is not None:
            self.scheduler.step()
            
        return loss, 0, outputs['h']

    def eval_batch(self, batch, batch_idx, device):
        """Validation step with AMP"""
        batch_mode = batch.get('mode', self.mode)
        self._set_active_mode(batch_mode)
        outputs = self.get_logits(batch, device, val=True)

        target_ids = batch['target_ids'].to(device)
        
        loss = self.get_loss(outputs['logits'], target_ids)
        
        return loss, 0, outputs['h']
    
    def eval_epoch(self, device, epoch):
        """
        Enhanced evaluation: first evaluate on sample questions, then run regular eval
        """
        # Evaluate on validation samples first (rank 0 only)
        if (not dist.is_initialized()) or dist.get_rank() == 0:
            if epoch % 1 == 0:  # Every epoch
                self.evaluate_validation_samples(device, epoch)
        
        # Then run regular evaluation
        return super().eval_epoch(device, epoch)
        
    def evaluate_validation_samples(self, device, epoch, num_samples=3,
                                    max_new_tokens=50, temperature=0.2, top_p=0.8):
        """
        Evaluate model on actual validation samples with both teacher-forcing and generation perplexity
        """
        self.model.eval()
        
        print(f"\n{'='*80}")
        print(f"VALIDATION SAMPLE EVALUATION - EPOCH {epoch}")
        print(f"{'='*80}")
        
        tokenizer = getattr(self, 'tokenizer', None)
        if tokenizer is None:
            print("Warning: No tokenizer available for decoding")
        
        val_iter = iter(self.val_dl)
        
        epoch_results = {
            'epoch': epoch,
            'samples': [],
            'avg_teacher_forcing_perplexity': 0.0,
            'avg_generation_perplexity': 0.0
        }
        
        sample_count = 0
        total_tf_perplexity = 0
        total_gen_perplexity = 0
        valid_tf_count = 0
        valid_gen_count = 0
        
        with torch.no_grad():
            while sample_count < num_samples:
                try:
                    batch = next(val_iter)
                except StopIteration:
                    break

                batch_mode = batch.get('mode', self.mode)
                self._set_active_mode(batch_mode)

                batch_idx = 0
                obsid = batch['obsids'][batch_idx] if 'obsids' in batch else "Unknown"
                
                # 1. Calculate teacher-forcing perplexity (how well it predicts the true answer)
                tf_perplexity = self._calculate_teacher_forcing_perplexity(batch, batch_idx, device)

                # 2. Generate response and calculate generation perplexity
                if isinstance(self.model, torch.nn.DataParallel) or isinstance(self.model, torch.nn.parallel.DistributedDataParallel):          
                    generated_text, input_text, target_text, generation_log_probs = self.model.module.generate_response_from_batch(
                        batch_data=batch,
                        batch_idx=batch_idx,
                        tokenizer=tokenizer,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p
                    )
                else:
                    generated_text, input_text, target_text, generation_log_probs = self.model.generate_response_from_batch(
                        batch_data=batch,
                        batch_idx=batch_idx,
                        tokenizer=tokenizer,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p
                    )
                
                # Calculate generation perplexity
                if generation_log_probs:
                    avg_log_prob = np.mean(generation_log_probs)
                    gen_perplexity = np.exp(-avg_log_prob)
                else:
                    gen_perplexity = float('inf')
                
                print(f"\n{'-'*60}")
                print(f"SAMPLE {sample_count + 1} (OBSID: {obsid})")
                print(f"{'-'*60}")
                print(f"QUESTION: {input_text}")
                print(f"TRUE ANSWER: {target_text}")
                print(f"GENERATED ANSWER: {generated_text}")
                print(f"Teacher-Forcing Perplexity: {tf_perplexity:.2f}")
                print(f"Generation Perplexity: {gen_perplexity:.2f}")
                print(f"Generated {len(generation_log_probs)} tokens")
                
                # Store results
                sample_result = {
                    'obsid': obsid,
                    'question': input_text,
                    'true_answer': target_text,
                    'generated_answer': generated_text,
                    'teacher_forcing_perplexity': tf_perplexity,
                    'generation_perplexity': gen_perplexity,
                    'num_generated_tokens': len(generation_log_probs),
                    'mode': batch_mode
                }
                epoch_results['samples'].append(sample_result)
                
                # Track valid perplexities for averaging
                if tf_perplexity != float('inf'):
                    total_tf_perplexity += tf_perplexity
                    valid_tf_count += 1
                if gen_perplexity != float('inf'):
                    total_gen_perplexity += gen_perplexity
                    valid_gen_count += 1
                
                sample_count += 1
        
        # Calculate averages
        if valid_tf_count > 0:
            epoch_results['avg_teacher_forcing_perplexity'] = total_tf_perplexity / valid_tf_count
        else:
            epoch_results['avg_teacher_forcing_perplexity'] = float('inf')
            
        if valid_gen_count > 0:
            epoch_results['avg_generation_perplexity'] = total_gen_perplexity / valid_gen_count
        else:
            epoch_results['avg_generation_perplexity'] = float('inf')
        
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch} SUMMARY:")
        print(f"Avg Teacher-Forcing Perplexity: {epoch_results['avg_teacher_forcing_perplexity']:.2f}")
        print(f"Avg Generation Perplexity: {epoch_results['avg_generation_perplexity']:.2f}")
        print(f"Valid TF Samples: {valid_tf_count}/{sample_count}")
        print(f"Valid Gen Samples: {valid_gen_count}/{sample_count}")
        print(f"{'='*60}")
        
        # Store results
        if not hasattr(self, 'validation_sample_history'):
            self.validation_sample_history = []
        self.validation_sample_history.append(epoch_results)
        
        # Save results
        if hasattr(self, 'log_path') and self.log_path:
            eval_file = os.path.join(self.log_path, f'{self.exp_name}_validation_samples.json')
            with open(eval_file, 'w') as f:
                # Convert inf to None for JSON serialization
                serializable_history = []
                for result in self.validation_sample_history:
                    serializable_result = result.copy()
                    serializable_result['avg_teacher_forcing_perplexity'] = (
                        None if result['avg_teacher_forcing_perplexity'] == float('inf') 
                        else result['avg_teacher_forcing_perplexity']
                    )
                    serializable_result['avg_generation_perplexity'] = (
                        None if result['avg_generation_perplexity'] == float('inf') 
                        else result['avg_generation_perplexity']
                    )
                    for sample in serializable_result['samples']:
                        if sample['teacher_forcing_perplexity'] == float('inf'):
                            sample['teacher_forcing_perplexity'] = None
                        if sample['generation_perplexity'] == float('inf'):
                            sample['generation_perplexity'] = None
                    serializable_history.append(serializable_result)
                
                json.dump(serializable_history, f, indent=2)

    def _calculate_teacher_forcing_perplexity(self, batch, batch_idx, device):
        """
        Calculate perplexity using teacher forcing (how well model predicts the true answer)
        """
        # Extract batch data
        input_ids = batch['input_ids'][batch_idx:batch_idx+1].to(device)
        target_ids = batch['target_ids'][batch_idx:batch_idx+1].to(device)

        # WRAP with autocast for validation too
        with autocast(enabled=getattr(self, 'use_amp', False)):
            if self.mode == "two_star":
                star_a_spectra = batch['masked_spectra_a'][batch_idx:batch_idx+1].to(device)
                star_b_spectra = batch['masked_spectra_b'][batch_idx:batch_idx+1].to(device)
                star_a_indices = batch['star_a_feature_indices'][batch_idx:batch_idx+1].to(device)
                star_b_indices = batch['star_b_feature_indices'][batch_idx:batch_idx+1].to(device)
                outputs = self.model(
                    input_ids=input_ids,
                    input_spectra=None,
                    special_token_positions=None,
                    star_a_spectra=star_a_spectra,
                    star_b_spectra=star_b_spectra,
                    star_a_indices=star_a_indices,
                    star_b_indices=star_b_indices,
                    )
            else:
                special_token_positions = batch['feature_start_indices'].to(device)
                input_spectra = batch['masked_spectra'].to(device) 
                outputs = self.model(
                    input_ids=input_ids,
                    input_spectra=input_spectra,
                    special_token_positions=special_token_positions,
                )
        
        
        logits = outputs['logits']
        
        # Calculate cross-entropy loss only on answer tokens (where target_ids != -100)
        # Shift for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()  # [batch, seq_len-1, vocab_size]
        shift_labels = target_ids[..., 1:].contiguous()   # [batch, seq_len-1]
        
        # Create mask for answer tokens (target_ids != -100)
        answer_mask = (shift_labels != -100)
        
        if not answer_mask.any():
            return float('inf')
        
        # Calculate loss only on answer tokens
        flat_logits = shift_logits.view(-1, shift_logits.size(-1))  # [batch*(seq_len-1), vocab_size]
        flat_labels = shift_labels.view(-1)                          # [batch*(seq_len-1)]
        flat_mask = answer_mask.view(-1)                            # [batch*(seq_len-1)]
        
        # Select only answer token positions
        answer_logits = flat_logits[flat_mask]  # [num_answer_tokens, vocab_size]
        answer_labels = flat_labels[flat_mask]  # [num_answer_tokens]
        
        if len(answer_logits) == 0:
            return float('inf')
        
        # Calculate cross-entropy loss
        loss = F.cross_entropy(answer_logits, answer_labels, reduction='mean')
        perplexity = torch.exp(loss).item()
        
        return perplexity

    def plot_perplexity_trends(self, save_path=None):
        """
        Plot both teacher-forcing and generation perplexity trends over epochs
        """
        if not hasattr(self, 'validation_sample_history') or not self.validation_sample_history:
            print("No validation sample history to plot")
            return
        
        epochs = []
        tf_perplexities = []
        gen_perplexities = []
        
        for result in self.validation_sample_history:
            epochs.append(result['epoch'])
            tf_perp = result['avg_teacher_forcing_perplexity']
            gen_perp = result['avg_generation_perplexity']
            
            # Only include finite values
            tf_perplexities.append(tf_perp if tf_perp != float('inf') else None)
            gen_perplexities.append(gen_perp if gen_perp != float('inf') else None)
        
        # Filter out None values for plotting
        valid_epochs = []
        valid_tf = []
        valid_gen = []
        
        for i, (ep, tf, gen) in enumerate(zip(epochs, tf_perplexities, gen_perplexities)):
            if tf is not None:
                valid_epochs.append(ep)
                valid_tf.append(tf)
            if gen is not None and i < len(valid_gen):
                valid_gen.append(gen)
        
        if not valid_epochs:
            print("No valid perplexity values to plot")
            return
        
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 6))
        
        if valid_tf:
            plt.plot(valid_epochs[:len(valid_tf)], valid_tf, 'b-o', 
                    label='Teacher-Forcing Perplexity', alpha=0.7)
        if valid_gen:
            plt.plot(valid_epochs[:len(valid_gen)], valid_gen, 'r-o', 
                    label='Generation Perplexity', alpha=0.7)
        
        plt.xlabel('Epoch')
        plt.ylabel('Perplexity')
        plt.title('Validation Sample Perplexity Trends')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')  # Log scale for better visualization
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Perplexity plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()

    def sanity_questions(self, device, tokenizer, max_new_tokens=50, temperature=0.7, top_p=0.9):
        # Set token IDs for generation
        self.pad_token_id = getattr(tokenizer, 'pad_id', 0)
        self.eos_token_id = getattr(tokenizer, 'eos_id', 1)  
        self.bos_token_id = getattr(tokenizer, 'bos_id', 2)
        
        q1 = 'Hi, tell me something about stars'
        q2 = 'Who was Albert Einstein'
        q3 = "Given the observed evolutionary stages of this star, what are the primary nuclear burning processes occurring within the stellar core at each stage, and how do these processes influence the star's luminosity, effective temperature, and surface gravity over time?"
        qs = [q1,q2,q3]
        ids = [torch.tensor(tokenizer.encode(q, bos=True, eos=False, allowed_special="all"), device=device).view(1,-1) for q in qs]
        print(f"{'-'*60}")
        print("SANITY QUESTIONS")
        print(f"Using: pad_token_id={self.pad_token_id}, eos_token_id={self.eos_token_id}, bos_token_id={self.bos_token_id}")
        print(f"{'-'*60}")
        for q, input_id in zip(qs, ids):
            current_attention_mask = torch.ones_like(input_id, dtype=torch.long, device=device)
            dummy = torch.zeros(input_id.shape[0], device=device)
            dummy_features = torch.zeros((1,4,2048), device=device)
            dummy_param = torch.zeros((1,4), device=device)
            generated_ids = self._generate_response(
                            input_ids=input_id,
                            features=dummy_features,
                            stage_mask=torch.zeros((1,4)),
                            n_stages=1,
                            ages=dummy_param,
                            masses=dummy_param,
                            metallicities=dummy_param,
                            max_new_tokens=max_new_tokens,
                            temperature=temperature,
                            top_p=top_p,
                            device=device
                        )

        # for i, gen_id in enumerate(generated_ids):
            answer = tokenizer.decode(generated_ids[0].cpu().numpy())
            print("Q: ", q)
            print("A: ", answer)


    
    def predict(self, test_dataloader, device, tokenizer=None,
                                    max_samples=10, max_new_tokens=50, temperature=0.7, top_p=0.9):
        """
        Updated predict function that passes attention masks to the model
        """
        self.model.eval()
        
        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)
            print("Loaded best model state")
        
        all_losses = []
        all_ce_losses = []
        all_evo_losses = []
        all_recon_losses = []

        self.sanity_questions(device, tokenizer, max_new_tokens, temperature, top_p)
        
        sample_count = 0
        pbar = tqdm(test_dataloader, desc="Generating predictions")
        
        print(f"\n{'='*80}")
        print("SAMPLE PREDICTIONS WITH ATTENTION MASKS")
        print(f"{'='*80}")
        
        for i, batch in enumerate(pbar):
            batch_size = batch['input_ids'].shape[0]
            
            with torch.no_grad():
                # Extract attention mask
                attention_mask = batch.get('input_attention_mask')
                
                # Forward pass with attention mask
                outputs = self.model(
                    input_ids=batch['input_ids'].to(device),
                    features=batch['features'].to(device),
                    stage_mask=batch['stage_mask'].to(device),
                    n_stages=batch['n_stages'].to(device),
                    ages=batch['ages'].to(device),
                    masses=batch['masses'].to(device),
                    metallicities=batch['metallicities'].to(device),
                    attention_mask=attention_mask.to(device) if attention_mask is not None else None,  # Pass to model
                    use_enhanced_features=True
                )
                
                # Calculate losses with output attention mask
                output_attention_mask = batch.get('output_attention_mask')
                loss, ce_loss, evo_loss, recon_loss = self.get_loss(
                    outputs, 
                    batch['output_ids'].to(device), 
                    output_attention_mask.to(device) if output_attention_mask is not None else None,
                    verbose=False
                )
                
                all_losses.append(loss.item())
                all_ce_losses.append(ce_loss.item())
                all_evo_losses.append(evo_loss.item())
                all_recon_losses.append(recon_loss.item())
                
                # Generate text for sample display
                if sample_count < max_samples:
                    for batch_idx in range(min(batch_size, max_samples - sample_count)):
                        if sample_count >= max_samples:
                            break
                            
                        # Extract input sequence for generation
                        input_ids = batch['input_ids'][batch_idx:batch_idx+1].to(device)
                        features = batch['features'][batch_idx:batch_idx+1].to(device)
                        stage_mask = batch['stage_mask'][batch_idx:batch_idx+1].to(device)
                        n_stages = batch['n_stages'][batch_idx:batch_idx+1].to(device)
                        ages = batch['ages'][batch_idx:batch_idx+1].to(device)
                        masses = batch['masses'][batch_idx:batch_idx+1].to(device)
                        metallicities = batch['metallicities'][batch_idx:batch_idx+1].to(device)
                        
                        # Generate response
                        generated_ids = self._generate_response(
                            input_ids=input_ids,
                            features=features,
                            stage_mask=stage_mask,
                            n_stages=n_stages,
                            ages=ages,
                            masses=masses,
                            metallicities=metallicities,
                            max_new_tokens=max_new_tokens,
                            temperature=temperature,
                            top_p=top_p,
                            device=device
                        )
                        
                        # Display sample
                        self._display_sample(
                            sample_idx=sample_count + 1,
                            input_ids=input_ids[0],
                            generated_ids=generated_ids[0],
                            target_ids=batch['output_ids'][batch_idx],
                            tokenizer=tokenizer,
                            features=features[0],
                            stage_mask=stage_mask[0],
                            outputs=outputs,
                            attention_mask=attention_mask[batch_idx] if attention_mask is not None else None
                        )
                        
                        sample_count += 1
                
                # Update progress bar
                avg_loss = np.mean(all_losses)
                avg_ce_loss = np.mean(all_ce_losses)
                avg_evo_loss = np.mean(all_evo_losses)
                avg_recon_loss = np.mean(all_recon_losses)
                
                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'ce': f'{avg_ce_loss:.4f}',
                    'evo': f'{avg_evo_loss:.4f}',
                    'recon': f'{avg_recon_loss:.4f}'
                })
                
                if (self.max_iter is not None) and (self.max_iter >= 0) and (i > self.max_iter):
                    break
        
        print(f"\n{'='*80}")
        print("PREDICTION SUMMARY")
        print(f"{'='*80}")
        print(f"Total samples processed: {len(all_losses) * test_dataloader.batch_size}")
        print(f"Average total loss: {np.mean(all_losses):.6f} ± {np.std(all_losses):.6f}")
        print(f"Average CE loss: {np.mean(all_ce_losses):.6f} ± {np.std(all_ce_losses):.6f}")
        print(f"Average evolution loss: {np.mean(all_evo_losses):.6f} ± {np.std(all_evo_losses):.6f}")
        print(f"Average reconstruction loss: {np.mean(all_recon_losses):.6f} ± {np.std(all_recon_losses):.6f}")
        
        return {
            'losses': all_losses,
            'ce_losses': all_ce_losses,
            'evo_losses': all_evo_losses,
            'recon_losses': all_recon_losses,
            'avg_loss': np.mean(all_losses),
            'std_loss': np.std(all_losses)
        }

    def _generate_response(self, input_ids, features, stage_mask, n_stages, ages, masses, 
                          metallicities, max_new_tokens, temperature, top_p, device):
        """
        Enhanced generation with attention masks
        """
        batch_size, seq_len = input_ids.shape
        generated_ids = input_ids.clone()
        
        # Create initial attention mask (all 1s for the input sequence)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
        
        # Get special token IDs
        pad_token_id = getattr(self, 'pad_token_id', None)
        eos_token_id = getattr(self, 'eos_token_id', None)
        
        print(f"Generation using pad_token_id={pad_token_id}, eos_token_id={eos_token_id}")
        
        for step in range(max_new_tokens):
            with torch.no_grad():
                # Update attention mask for current sequence length
                current_attention_mask = torch.ones_like(generated_ids, dtype=torch.long, device=device)
                
                outputs = self.model(
                    input_ids=generated_ids,
                    features=features,
                    stage_mask=stage_mask,
                    n_stages=n_stages,
                    ages=ages,
                    masses=masses,
                    metallicities=metallicities,
                    attention_mask=current_attention_mask,  # Pass attention mask to model
                    use_enhanced_features=True
                )
                
                next_token_logits = outputs['logits'][:, -1, :]
                
                # Suppress padding token
                if pad_token_id is not None:
                    next_token_logits[:, pad_token_id] = float('-inf')
                
                # Apply temperature and sampling
                if temperature > 0:
                    next_token_logits = next_token_logits / temperature
                    
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                        
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        
                        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                        next_token_logits[indices_to_remove] = float('-inf')
                    
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                generated_ids = torch.cat([generated_ids, next_token], dim=1)
                
                # Check stopping conditions
                if eos_token_id is not None and next_token.item() == eos_token_id:
                    print(f"Step {step}: EOS token generated, stopping")
                    break
                
                if pad_token_id is not None and next_token.item() == pad_token_id:
                    print(f"Step {step}: Padding token generated despite suppression, stopping")
                    break
        
        return generated_ids

    def _display_sample(self, sample_idx, input_ids, generated_ids, target_ids, tokenizer, 
                       features, stage_mask, outputs, attention_mask=None):  # Add attention_mask parameter
        """
        Display a sample prediction with question, generated answer, and true answer
        Now supports attention mask for better token cleaning
        """
        print(f"\n{'-'*60}")
        print(f"SAMPLE {sample_idx}")
        print(f"{'-'*60}")
        
        # Clean padding tokens before decoding if we have attention mask
        def clean_tokens_with_attention_mask(token_ids, attn_mask=None):
            """Remove padding tokens using attention mask"""
            if attn_mask is not None:
                # Keep only tokens where attention mask is 1
                valid_positions = attn_mask.bool()
                if valid_positions.any():
                    return token_ids[valid_positions]
            return token_ids
        
        # Convert token IDs to text if tokenizer is available
        if tokenizer is not None:
            try:
                # Clean input tokens
                if attention_mask is not None:
                    clean_input_ids = clean_tokens_with_attention_mask(input_ids, attention_mask)
                else:
                    clean_input_ids = input_ids
                
                # For generated and target, we don't have attention masks, so use the old cleaning method
                def remove_padding_tokens(token_ids, pad_token='!'):
                    """Fallback cleaning method"""
                    # Try to identify padding token ID
                    if hasattr(tokenizer, 'encode'):
                        try:
                            pad_token_id = tokenizer.encode(pad_token, bos=False, eos=False)[0]
                            # Remove padding from the end
                            tokens_list = token_ids.cpu().numpy().tolist()
                            while tokens_list and tokens_list[-1] == pad_token_id:
                                tokens_list.pop()
                            return torch.tensor(tokens_list)
                        except:
                            pass
                    return token_ids
                
                clean_generated_ids = remove_padding_tokens(generated_ids)
                clean_target_ids = remove_padding_tokens(target_ids)
                
                # Decode cleaned tokens
                input_text = tokenizer.decode(clean_input_ids.cpu().numpy())
                generated_text = tokenizer.decode(clean_generated_ids.cpu().numpy())
                target_text = tokenizer.decode(clean_target_ids.cpu().numpy())
                
                # Extract just the generated part (remove input)
                input_len = len(clean_input_ids)
                if len(clean_generated_ids) > input_len:
                    generated_only = clean_generated_ids[input_len:]
                    generated_only_text = tokenizer.decode(generated_only.cpu().numpy())
                else:
                    generated_only_text = "[No generation]"
                
                print(f"INPUT/QUESTION:")
                print(f"  {input_text}")
                print(f"\nGENERATED ANSWER:")
                print(f"  {generated_only_text}")
                print(f"\nTRUE ANSWER:")
                print(f"  {target_text}")
                
                # Show attention mask info if available
                if attention_mask is not None:
                    real_tokens = attention_mask.sum().item()
                    total_tokens = len(attention_mask)
                    print(f"\nATTENTION MASK INFO:")
                    print(f"  Real tokens: {real_tokens}/{total_tokens} ({100*real_tokens/total_tokens:.1f}%)")
                
            except Exception as e:
                print(f"Error decoding tokens: {e}")
                # Fallback to token IDs
                print(f"INPUT IDs: {input_ids.cpu().numpy()}")
                print(f"GENERATED IDs: {generated_ids.cpu().numpy()}")
                print(f"TARGET IDs: {target_ids.cpu().numpy()}")
        else:
            # Display token IDs
            input_len = len(input_ids)
            if len(generated_ids) > input_len:
                generated_only = generated_ids[input_len:]
                print(f"INPUT IDs: {input_ids.cpu().numpy()}")
                print(f"GENERATED IDs (new): {generated_only.cpu().numpy()}")
                print(f"TARGET IDs: {target_ids.cpu().numpy()}")
            else:
                print(f"INPUT IDs: {input_ids.cpu().numpy()}")
                print(f"GENERATED IDs: {generated_ids.cpu().numpy()}")
                print(f"TARGET IDs: {target_ids.cpu().numpy()}")
        
        # Display stellar evolution information
        n_stages_val = stage_mask.sum().item()
        print(f"\nSTELLAR EVOLUTION INFO:")
        print(f"  Number of evolutionary stages: {n_stages_val}")
        print(f"  Feature dimensions: {features.shape}")
        print(f"  Stage mask: {stage_mask.cpu().numpy()}")
        
        # Display some evolution predictions if available
        if 'evolution_predictions' in outputs and outputs['evolution_predictions'] is not None:
            evo_preds = outputs['evolution_predictions'][0]  # First sample in batch
            print(f"  Evolution prediction shape: {evo_preds.shape}")
            if len(evo_preds.shape) > 1 and evo_preds.shape[0] > 0:
                print(f"  First stage prediction (sample): {evo_preds[0, :5].cpu().numpy()}")
        
        print(f"{'-'*60}")


class CLIPTrainer(Trainer):
    """
    CLIP-style contrastive trainer for multimodal stellar model
    """
    
    def __init__(self, 
                 temperature_logging: bool = True,
                 similarity_logging: bool = True,
                 retrieval_eval: bool = False,
                 retrieval_k: List[int] = [1, 5, 10],
                 latent_ids: List[str] = ['Teff', 'logg', 'FeH'],
                 **kwargs):
        """
        Args:
            temperature_logging: Whether to log temperature values
            similarity_logging: Whether to log similarity statistics
            retrieval_eval: Whether to compute retrieval metrics
            retrieval_k: K values for retrieval evaluation
            **kwargs: Arguments for base Trainer class
        """
        super().__init__(**kwargs)
        self.temperature_logging = temperature_logging
        self.similarity_logging = similarity_logging
        self.retrieval_eval = retrieval_eval
        self.retrieval_k = retrieval_k
        self.latent_ids = latent_ids
        
        # Additional tracking for CLIP-specific metrics
        self.train_temperatures = []
        self.val_temperatures = []
        self.train_similarities = []
        self.val_similarities = []
        self.retrieval_metrics = {'train': [], 'val': []}
    
    def get_latent_vars(self, batch):
        metadata = batch['stellar_data']
        latent_vars = []
        for var in self.latent_ids:
            if var not in metadata[0]:
                raise ValueError(f"Latent variable '{var}' not found in batch metadata. keys: {metadata[0].keys()}")
            latent_vars.append(torch.tensor([m[var] for m in metadata]))
        return torch.stack(latent_vars, dim=1)  # Shape: [batch_size, num_latents]
            
        
    def train_batch(self, batch: Dict[str, Any], batch_idx: int, device: torch.device) -> tuple:
        """
        Training step for CLIP model
        
        Args:
            batch: Batch dictionary containing descriptions, features, etc.
            batch_idx: Batch index
            device: Device to run on
            
        Returns:
            (loss, accuracy, targets)
        """
        # Extract data from batch
        descriptions = batch['description_tokens']
        features = batch['features']  # [batch_size, feature_dim]
        if len(self.latent_ids) > 0:
            latent_vars = self.get_latent_vars(batch).to(device)
        else:
            latent_vars = None
        
        if features is None:
            # Skip batch if no features available
            dummy_loss = torch.tensor(0.0, requires_grad=True, device=device)
            dummy_acc = torch.zeros(self.output_dim, device=device)
            return dummy_loss, dummy_acc, descriptions
        
        # Forward pass through multimodal model
        outputs = self.model(descriptions, features, latent_vars)
        
        # Compute contrastive loss (use model.module if wrapped in DataParallel)
        if isinstance(self.model, torch.nn.DataParallel) or isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            loss = self.model.module.compute_contrastive_loss(outputs['similarity_matrix'])
        else:
            loss = self.model.compute_contrastive_loss(outputs['similarity_matrix'])
        
        # Compute accuracy (diagonal elements of similarity matrix)
        similarity_matrix = outputs['similarity_matrix']
        batch_size = similarity_matrix.size(0)
        
        # Top-1 accuracy for text-to-spectral retrieval
        text_to_spectral_acc = (similarity_matrix.argmax(dim=1) == torch.arange(batch_size, device=device)).float().sum()
        
        # Top-1 accuracy for spectral-to-text retrieval  
        spectral_to_text_acc = (similarity_matrix.argmax(dim=0) == torch.arange(batch_size, device=device)).float().sum()
        
        # Average accuracy
        avg_accuracy = (text_to_spectral_acc + spectral_to_text_acc) / (2 * batch_size)
        
        # Create accuracy tensor for compatibility with base class
        acc_tensor = torch.zeros(self.output_dim, device=device)
        acc_tensor[0] = avg_accuracy  # Store in first dimension
        
        # Backward pass
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            if (batch_idx + 1) % self.accumulation_step == 0:
                self.scaler.unscale_(self.optimizer)
                if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                if self.scheduler is not None:
                    self.scheduler.step()
        else:
            loss.backward()
            if (batch_idx + 1) % self.accumulation_step == 0:
                if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
        
        # Log additional metrics
        if self.temperature_logging:
            self.train_temperatures.append(outputs['temperature'].item())
            
        if self.similarity_logging:
            # Log similarity statistics
            sim_stats = {
                'mean': similarity_matrix.mean().item(),
                'std': similarity_matrix.std().item(),
                'max': similarity_matrix.max().item(),
                'min': similarity_matrix.min().item()
            }
            self.train_similarities.append(sim_stats)
        
        return loss, acc_tensor, descriptions
    
    def eval_batch(self, batch: Dict[str, Any], batch_idx: int, device: torch.device) -> tuple:
        """
        Evaluation step for CLIP model
        """
        descriptions = batch['description_tokens']
        features = batch['features']

        if len(self.latent_ids) > 0:
            latent_vars = self.get_latent_vars(batch).to(device)
        else:
            latent_vars = None
        
        if features is None:
            dummy_loss = torch.tensor(0.0, device=device)
            dummy_acc = torch.zeros(self.output_dim, device=device)
            return dummy_loss, dummy_acc, descriptions
        
        with torch.no_grad():
            outputs = self.model(descriptions, features, latent_vars)
            if isinstance(self.model, torch.nn.DataParallel) or isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                loss = self.model.module.compute_contrastive_loss(outputs['similarity_matrix'])
            else:
                loss = self.model.compute_contrastive_loss(outputs['similarity_matrix'])
            
            # Compute accuracy
            similarity_matrix = outputs['similarity_matrix']
            batch_size = similarity_matrix.size(0)
            
            text_to_spectral_acc = (similarity_matrix.argmax(dim=1) == torch.arange(batch_size, device=device)).float().sum()
            spectral_to_text_acc = (similarity_matrix.argmax(dim=0) == torch.arange(batch_size, device=device)).float().sum()
            avg_accuracy = (text_to_spectral_acc + spectral_to_text_acc) / (2 * batch_size)
            
            acc_tensor = torch.zeros(self.output_dim, device=device)
            acc_tensor[0] = avg_accuracy
            
            # Log metrics
            if self.temperature_logging:
                self.val_temperatures.append(outputs['temperature'].item())
                
            if self.similarity_logging:
                sim_stats = {
                    'mean': similarity_matrix.mean().item(),
                    'std': similarity_matrix.std().item(),
                    'max': similarity_matrix.max().item(),
                    'min': similarity_matrix.min().item()
                }
                self.val_similarities.append(sim_stats)
        
        return loss, acc_tensor, descriptions
    
    def compute_retrieval_metrics(self, dataloader, device, split_name='val'):
        """
        Compute retrieval metrics on the entire dataset
        """
        if not self.retrieval_eval:
            return {}
            
        self.model.eval()
        all_text_embeddings = []
        all_spectral_embeddings = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Computing {split_name} retrieval metrics"):
                descriptions = batch['descriptions']
                features = batch['features']
                
                if features is None:
                    continue
                    
                outputs = self.model(descriptions, features)
                all_text_embeddings.append(outputs['text_embeddings'].cpu())
                all_spectral_embeddings.append(outputs['spectral_embeddings'].cpu())
        
        if not all_text_embeddings:
            return {}
            
        # Concatenate all embeddings
        text_embeddings = torch.cat(all_text_embeddings, dim=0)
        spectral_embeddings = torch.cat(all_spectral_embeddings, dim=0)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(text_embeddings, spectral_embeddings.T)
        
        # Compute retrieval metrics
        metrics = {}
        n_samples = similarity_matrix.size(0)
        
        # Text-to-spectral retrieval
        text_to_spectral_ranks = []
        for i in range(n_samples):
            similarities = similarity_matrix[i]
            rank = (similarities.argsort(descending=True) == i).nonzero(as_tuple=True)[0].item() + 1
            text_to_spectral_ranks.append(rank)
        
        # Spectral-to-text retrieval  
        spectral_to_text_ranks = []
        for i in range(n_samples):
            similarities = similarity_matrix[:, i]
            rank = (similarities.argsort(descending=True) == i).nonzero(as_tuple=True)[0].item() + 1
            spectral_to_text_ranks.append(rank)
        
        # Compute recall@k for both directions
        for k in self.retrieval_k:
            text_to_spectral_recall_k = sum(1 for rank in text_to_spectral_ranks if rank <= k) / len(text_to_spectral_ranks)
            spectral_to_text_recall_k = sum(1 for rank in spectral_to_text_ranks if rank <= k) / len(spectral_to_text_ranks)
            
            metrics[f'{split_name}_text_to_spectral_recall@{k}'] = text_to_spectral_recall_k
            metrics[f'{split_name}_spectral_to_text_recall@{k}'] = spectral_to_text_recall_k
            metrics[f'{split_name}_avg_recall@{k}'] = (text_to_spectral_recall_k + spectral_to_text_recall_k) / 2
        
        # Mean rank
        metrics[f'{split_name}_text_to_spectral_mean_rank'] = np.mean(text_to_spectral_ranks)
        metrics[f'{split_name}_spectral_to_text_mean_rank'] = np.mean(spectral_to_text_ranks)
        metrics[f'{split_name}_avg_mean_rank'] = (np.mean(text_to_spectral_ranks) + np.mean(spectral_to_text_ranks)) / 2
        
        # Median rank
        metrics[f'{split_name}_text_to_spectral_median_rank'] = np.median(text_to_spectral_ranks)
        metrics[f'{split_name}_spectral_to_text_median_rank'] = np.median(spectral_to_text_ranks)
        metrics[f'{split_name}_avg_median_rank'] = (np.median(text_to_spectral_ranks) + np.median(spectral_to_text_ranks)) / 2
        
        return metrics
    
    def train_epoch(self, device, epoch):
        """Override train_epoch to add retrieval metrics"""
        # Call parent train_epoch
        train_loss, train_acc = super().train_epoch(device, epoch)
        
        # Compute retrieval metrics periodically
        if self.retrieval_eval and epoch % 5 == 0:  # Every 5 epochs
            retrieval_metrics = self.compute_retrieval_metrics(self.train_dl, device, 'train')
            self.retrieval_metrics['train'].append(retrieval_metrics)
        
        return train_loss, train_acc
    
    def eval_epoch(self, device, epoch):
        """Override eval_epoch to add retrieval metrics"""
        # Call parent eval_epoch
        val_loss, val_acc = super().eval_epoch(device, epoch)
        
        # Compute retrieval metrics
        if self.retrieval_eval and epoch % 5 == 0:  # Every 5 epochs
            retrieval_metrics = self.compute_retrieval_metrics(self.val_dl, device, 'val')
            self.retrieval_metrics['val'].append(retrieval_metrics)
        
        return val_loss, val_acc
    
    def fit(self, num_epochs, device, early_stopping=None, start_epoch=0, best='loss', conf=False):
        """Override fit to save CLIP-specific metrics"""
        results = super().fit(num_epochs, device, early_stopping, start_epoch, best, conf)
        
        # Add CLIP-specific metrics to results
        results.update({
            'train_temperatures': self.train_temperatures,
            'val_temperatures': self.val_temperatures,
            'train_similarities': self.train_similarities,
            'val_similarities': self.val_similarities,
            'retrieval_metrics': self.retrieval_metrics
        })
        
        return results
    
    def predict(self, test_dataloader, device, load_best=True, compute_embeddings=True):
        """
        Enhanced prediction method for CLIP model
        """
        if load_best and hasattr(self, 'best_state_dict'):
            self.model.load_state_dict(self.best_state_dict)
            
        self.model.eval()
        
        all_losses = []
        all_accuracies = []
        all_text_embeddings = []
        all_spectral_embeddings = []
        all_descriptions = []
        all_similarities = []
        
        with torch.no_grad():
            for batch in tqdm(test_dataloader, desc="Generating predictions"):
                descriptions = batch['description_tokens']
                features = batch['features']
                
                if features is None:
                    continue
                
                if len(self.latent_ids) > 0:
                    latent_vars = self.get_latent_vars(batch).to(device)
                else:
                    latent_vars = None
                
                outputs = self.model(descriptions, features, latent_vars)
                if isinstance(self.model, torch.nn.DataParallel) or isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                    loss = self.model.module.compute_contrastive_loss(outputs['similarity_matrix'])
                else:
                    loss = self.model.compute_contrastive_loss(outputs['similarity_matrix'])
                
                # Compute accuracy
                similarity_matrix = outputs['similarity_matrix']
                batch_size = similarity_matrix.size(0)
                text_to_spectral_acc = (similarity_matrix.argmax(dim=1) == torch.arange(batch_size, device=device)).float().sum()
                spectral_to_text_acc = (similarity_matrix.argmax(dim=0) == torch.arange(batch_size, device=device)).float().sum()
                avg_accuracy = (text_to_spectral_acc + spectral_to_text_acc) / (2 * batch_size)
                
                all_losses.append(loss.item())
                all_accuracies.append(avg_accuracy.item())
                
                if compute_embeddings:
                    all_text_embeddings.append(outputs['text_embeddings'].cpu())
                    all_spectral_embeddings.append(outputs['spectral_embeddings'].cpu())
                    all_similarities.append(similarity_matrix.cpu())
                
                all_descriptions.extend(descriptions)
        
        results = {
            'losses': all_losses,
            'accuracies': all_accuracies,
            'descriptions': all_descriptions,
            'avg_loss': np.mean(all_losses),
            'avg_accuracy': np.mean(all_accuracies)
        }
        
        if compute_embeddings and all_text_embeddings:
            results.update({
                'text_embeddings': torch.cat(all_text_embeddings, dim=0),
                'spectral_embeddings': torch.cat(all_spectral_embeddings, dim=0),
                'similarities': torch.cat(all_similarities, dim=0)
            })
            
            # Compute final retrieval metrics
            final_retrieval_metrics = self.compute_retrieval_metrics(test_dataloader, device, 'test')
            results['retrieval_metrics'] = final_retrieval_metrics
        
        return results
