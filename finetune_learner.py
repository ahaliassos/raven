from pytorch_lightning import LightningModule
import torch
from torch.optim import Adam, AdamW

from espnet.asr.asr_utils import add_results_to_json, torch_load
from espnet.nets.batch_beam_search import BatchBeamSearch
from espnet.nets.pytorch_backend.e2e_asr_transformer import E2E
from espnet.nets.pytorch_backend.lm.transformer import TransformerLM
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask
from espnet.nets.scorers.length_bonus import LengthBonus
from metrics import WER
from schedulers.warmup_cosine import WarmupCosineScheduler
from utils import ids_to_str, set_requires_grad, UNIGRAM1000_LIST, get_param_groups


class Learner(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg
        if self.cfg.data.modality == "audio":
            self.backbone_args = self.cfg.model.audio_backbone
        elif self.cfg.data.modality == "video":
            self.backbone_args = self.cfg.model.visual_backbone
        else:
            raise NotImplementedError
        self.model = self.load_model()

        self.ignore_id = -1

        self.beam_search = self.get_beam_search(self.model)
        self.wer = WER()

    
    def load_model(self):
        if self.cfg.data.labels_type == "unigram1000":
            odim = len(UNIGRAM1000_LIST)
        else:
            raise NotImplementedError

        model = E2E(odim, self.backbone_args)

        if self.cfg.model.pretrained_model_path:
            print("Load pretrained model weights")
            ckpt = torch.load(self.cfg.model.pretrained_model_path, map_location=lambda storage, loc: storage)
            if self.cfg.model.transfer_only_encoder:
                ckpt = {(k.replace("encoder.","") if k.startswith("encoder.") else k): v for k, v in ckpt.items()}
                strict = True
                if self.cfg.model.reinit_blocks != -1:
                    strict = False
                    total_blocks = self.backbone_args.elayers
                    names = tuple((f"encoders.{total_blocks-i-1}" for i in range(self.cfg.model.reinit_blocks)))
                    ckpt = {k: v for k, v in ckpt.items() if not k.startswith(names) and not k.startswith("after_norm")}
                model.encoder.load_state_dict(ckpt, strict=strict)
            else:
                model.load_state_dict(ckpt)
            
        return model

    def get_beam_search(self, model):
        if getattr(self.cfg.data, "labels_type", "char") == "unigram1000":
            token_list = UNIGRAM1000_LIST
        else:
            raise NotImplementedError
        odim = len(token_list)
        self.token_list = token_list

        scorers = model.scorers()

        if self.cfg.decode.lm_weight and self.cfg.model.pretrained_lm_path:
            lm = TransformerLM(len(token_list), self.cfg.model.language_model)
            set_requires_grad(lm, False)
            print("Load pretrained language model weights")
            torch_load(self.cfg.model.pretrained_lm_path, lm)
        else:
            lm = None

        scorers["lm"] = lm
        scorers["length_bonus"] = LengthBonus(len(token_list))

        weights = dict(
            decoder=1.0 - self.cfg.decode.ctc_weight,
            ctc=self.cfg.decode.ctc_weight,
            lm=self.cfg.decode.lm_weight,
            length_bonus=self.cfg.decode.penalty,
        )
        beam_search = BatchBeamSearch(
            beam_size=self.cfg.decode.beam_size,
            vocab_size=len(token_list),
            weights=weights,
            scorers=scorers,
            sos=odim - 1,
            eos=odim - 1,
            token_list=token_list,
            pre_beam_score_key=None if self.cfg.decode.ctc_weight == 1.0 else "decoder",
        )

        return beam_search

    def forward(self, model, data, padding_mask, lengths, label):
        return model(data, padding_mask, lengths, label=label)
    
    def training_step(self, data, batch_idx):
        lengths = torch.tensor(data["data_lengths"], device=data["data"].device)
        padding_mask = make_non_pad_mask(lengths).to(lengths.device)
        label = data["label"].squeeze(1)
        _, loss, loss_ctc, loss_att, acc = self.forward(
            self.model, 
            data["data"].squeeze(1), 
            padding_mask.unsqueeze(-2), 
            lengths,
            label, 
        )

        self.log("loss", loss, on_step=True, on_epoch=True, batch_size=len(label))
        self.log("loss_ctc", loss_ctc, on_step=False, on_epoch=True, batch_size=len(label))
        self.log("loss_att", loss_att, on_step=False, on_epoch=True, batch_size=len(label))
        self.log("decoder_acc", acc, on_step=True, on_epoch=True, batch_size=len(label))

        self.log('monitoring_step', self.global_step)  # this is to save the last k checkpoints

        return loss

    def calculate_wer(self, data, padding_mask, labels):
        labels = labels.squeeze(1)
        data = data.squeeze(1)
        padding_mask = padding_mask
        for idx, (vid, label, mask) in enumerate(zip(data, labels, padding_mask)):
            x = vid[mask].unsqueeze(0)
            feat, _ = self.model.encoder(x, None)

            if isinstance(self.beam_search, BatchBeamSearch):
                nbest_hyps = self.beam_search(
                        x=feat.squeeze(0),
                        maxlenratio=self.cfg.decode.maxlenratio,
                        minlenratio=self.cfg.decode.minlenratio
                    )
            else:
                raise NotImplementedError
 
            nbest_hyps = [
                h.asdict() for h in nbest_hyps[: min(len(nbest_hyps), 1)]
            ]
            transcription = add_results_to_json(nbest_hyps, self.token_list)
            transcription = transcription.replace("<eos>", "")

            label = label[label != self.ignore_id]
            groundtruth = ids_to_str(label, self.token_list)

            groundtruth = groundtruth.replace("▁", " ").strip()
            transcription = transcription.replace("▁", " ").strip()

            self.wer.update(transcription, groundtruth)

    def validation_step(self, data, batch_idx):
        lengths = torch.tensor(data["data_lengths"], device=data["data"].device)
        padding_mask = make_non_pad_mask(lengths).to(lengths.device)

        label = data["label"].squeeze(1)

        _, loss, loss_ctc, loss_att, acc = self.forward(
            self.model, 
            data["data"].squeeze(1), 
            padding_mask.unsqueeze(-2), 
            lengths,
            label, 
        )

        self.log("loss_val", loss, batch_size=len(label))
        self.log("loss_ctc_val", loss_ctc, batch_size=len(label))
        self.log("loss_att_val", loss_att, batch_size=len(label))
        self.log("decoder_acc_val", acc, batch_size=len(label))

    def test_step(self, data, batch_idx):
        lengths = torch.tensor(data["data_lengths"], device=data["data"].device)
        padding_mask = make_non_pad_mask(lengths).to(lengths.device)
        self.calculate_wer(data["data"], padding_mask, data["label"])

    def test_epoch_end(self, outputs):
        wer = self.wer.compute()
        print(wer)
        self.log("wer", wer)
        self.wer.reset()
    
    def on_train_epoch_start(self):
        sampler = self.trainer.train_dataloader.loaders.batch_sampler
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(self.current_epoch)
        return super().on_train_epoch_start()

    # potentially want different schedulers for predictors and rest of model
    def configure_optimizers(self):
        opt_cls = AdamW if self.cfg.optimizer.weight_decay > 0. else Adam
        optimizer = opt_cls(
            get_param_groups(
                self.model, 
                self.backbone_args.elayers, 
                self.cfg.optimizer.lr, 
                self.cfg.optimizer.lr_other, 
                self.cfg.optimizer.lr_decay_rate,
                self.cfg.optimizer.ctc_equals_other,
                min_lr=self.cfg.optimizer.min_lr,
            ),
            weight_decay=self.cfg.optimizer.weight_decay,
            betas=self.cfg.optimizer.betas,
        )

        scheduler = WarmupCosineScheduler(
            optimizer, 
            self.cfg.optimizer.warmup_epochs, 
            self.cfg.trainer.max_epochs, 
            len(self.trainer.datamodule.train_dataloader()),
            min_lr=self.cfg.optimizer.min_lr,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1} 
        return [optimizer], [scheduler]      
        