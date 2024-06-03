import torch
import types
import warnings
import traceback
import sys
import os
from easyeditor import BaseEditor, MultimodalTrainer, MultimodalEditor
# from easyeditor.dataset.evaluate_vqa import VQADataset
from easyeditor.dataset.vqa import VQADataset
from easyeditor import MENDMultimodalTrainingHparams, SERACMultimodalTrainingHparams, IKEMultimodalHyperParams, MENDMultimodalHparams, SERACMultimodalHparams

def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
    log = file if hasattr(file, 'write') else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))


def serac_qwen():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    training_hparams = SERACMultimodalTrainingHparams.from_hparams('hparams/TRAINING/SERAC/qwen-vl_final.yaml')
    train_ds = VQADataset('/data0/huanghan/datasets/mmkb/utils/form_dataset/mmkb_edits_train_alter_entity.json', config=training_hparams)
    eval_ds = VQADataset('/data0/huanghan/datasets/mmkb/utils/form_dataset/mmkb_edits_eval_alter_entity.json', config=training_hparams)
    # train_ds = VQADataset('/home/zhonghaitian/MLLM/EasyEdit/data/port_50_birthplace_train_.json', config=training_hparams)
    # eval_ds = VQADataset('/home/zhonghaitian/MLLM/EasyEdit/data/port_50_birthplace_eval_.json', config=training_hparams)
    trainer = MultimodalTrainer(
        config=training_hparams,
        train_set=train_ds,
        val_set=eval_ds
    )

    trainer.run()

def mend_qwen():
    training_hparams = MENDMultimodalTrainingHparams.from_hparams('hparams/TRAINING/MEND/qwen-vl_final.yaml')
    train_ds = VQADataset('/data0/huanghan/datasets/mmkb/utils/form_dataset/mmkb_edits_train_alter_entity.json', config=training_hparams)
    eval_ds = VQADataset('/data0/huanghan/datasets/mmkb/utils/form_dataset/mmkb_edits_eval_alter_entity.json', config=training_hparams)
    # train_ds = VQADataset('/home/zhonghaitian/MLLM/EasyEdit/data/port_50_birthplace_train_.json', config=training_hparams)
    # eval_ds = VQADataset('/home/zhonghaitian/MLLM/EasyEdit/data/port_50_birthplace_eval_.json', config=training_hparams)
    trainer = MultimodalTrainer(
        config=training_hparams,
        train_set=train_ds,
        val_set=eval_ds
    )

    trainer.run()

def serac_owl():
    training_hparams = SERACMultimodalTrainingHparams.from_hparams('hparams/TRAINING/SERAC/owl2_final.yaml')
    # train_ds = VQADataset('/data0/huanghan/datasets/mmkb/utils/form_dataset/mmkb_edits_train_alter_entity.json', config=training_hparams)
    # eval_ds = VQADataset('/data0/huanghan/datasets/mmkb/utils/form_dataset/mmkb_edits_eval_alter_entity.json', config=training_hparams)
    train_ds = VQADataset('/home/v-hazhong/Datasets/VLKEB/huanghan/mmkb_edits_train_alter_entity.json', config=training_hparams)
    # train_ds = VQADataset('/home/v-hazhong/Datasets/VLKEB/mmkb_edits_train_alter_entity_50.json', config=training_hparams)
    eval_ds = VQADataset('/home/v-hazhong/Datasets/VLKEB/eval_Port_qa.json', config=training_hparams)
    # eval_ds = VQADataset('/home/v-hazhong/Datasets/VLKEB/eval_Port_100.json', config=training_hparams)
    trainer = MultimodalTrainer(
        config=training_hparams,
        train_set=train_ds,
        val_set=eval_ds
    )

    trainer.run()

def ft_qwen():
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/qwen-vl.yaml')
    eval_ds = VQADataset('/data0/zhonghaitian/Datasets/mmkb/Portability/eval_Port.json', config=hparams)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.run()

def ft_owl():
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/owl2.yaml')
    eval_ds = VQADataset('/data0/zhonghaitian/Datasets/mmkb/Portability/eval_Port.json', config=hparams)
    # eval_ds = VQADataset('/data0/zhonghaitian/Datasets/mmkb/Portability/port_qa_short.json', config=hparams)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.run()

def ft_qwen_vision():
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/qwen-vl_vision.yaml')
    eval_ds = VQADataset('/data0/zhonghaitian/Datasets/mmkb/Portability/eval_Port.json', config=hparams)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.run()

def ft_owl_vision():
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/owl2_vision.yaml')
    eval_ds = VQADataset('/data0/zhonghaitian/Datasets/mmkb/Portability/eval_Port.json', config=hparams)
    # eval_ds = VQADataset('/home/zhonghaitian/MLLM/EasyEdit/data/port_50_birthplace_eval_.json', config=hparams)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.run()

if __name__ == '__main__':
    # warnings.showwarning = warn_with_traceback
    warnings.filterwarnings("ignore", message="None of the inputs have requires_grad=True. Gradients will be None", category=UserWarning, module='torch.utils.checkpoint')
    # serac_qwen()
    # mend_qwen()
    serac_owl()
    # ft_qwen()
    # ft_owl()
    # ft_qwen_vision()
    # ft_owl_vision()