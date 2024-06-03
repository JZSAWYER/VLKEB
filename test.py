import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
import types
from statistics import mean

from easyeditor import BaseEditor, MultimodalTrainer, MultimodalEditor
from easyeditor import CaptionDataset, VQADataset
from easyeditor import MENDMultimodalTrainingHparams, SERACMultimodalTrainingHparams, IKEMultimodalHyperParams, MENDMultimodalHparams \
    , SERACMultimodalHparams, FTMultimodalHparams
from easyeditor import encode_ike_facts_multimodal
from sentence_transformers import SentenceTransformer
import sys
eval_path = "/home/v-hazhong/Datasets/VLKEB/eval_Port_qa.json"
eval_short_path = "/home/v-hazhong/Datasets/VLKEB/eval_Port_qa_100.json"


def print_result(metrics, save_path=None):
    rewrite_acc = mean([m['post']['rewrite_acc'].item() for m in metrics])
    rephrase_acc = mean([m['post']['rephrase_acc'].item() for m in metrics])
    rephrase_image_acc = mean([m['post']['rephrase_image_acc'].item() for m in metrics])
    locality_acc = mean([m['post']['locality_acc'].item() for m in metrics])
    locality_image_acc = mean([m['post']['locality_image_acc'].item() for m in metrics])
    print(f'rewrite_acc: {rewrite_acc}')
    print(f'rephrase_acc: {rephrase_acc}')
    print(f'rephrase_image_acc: {rephrase_image_acc}')
    print(f'locality_acc: {locality_acc}')
    print(f'locality_image_acc: {locality_image_acc}')

    ### portability
    portability_acc = mean([m['post']['portability_acc'].item() for m in metrics if 'portability_acc' in m['post']])
    print(f'portability_acc: {portability_acc}')

    if save_path is not None:
        with open(save_path, 'w') as f:
            f.write(f'rewrite_acc: {rewrite_acc}\n')
            f.write(f'rephrase_acc: {rephrase_acc}\n')
            f.write(f'rephrase_image_acc: {rephrase_image_acc}\n')
            f.write(f'locality_acc: {locality_acc}\n')
            f.write(f'locality_image_acc: {locality_image_acc}\n')

            #### portability
            f.write(f'portability_acc: {portability_acc}\n')

def test_SERAC_qwen():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    hparams = SERACMultimodalHparams.from_hparams('hparams/SERAC/qwen-vl.yaml')
    # eval_ds = VQADataset(eval_path, config=hparams)
    eval_ds = VQADataset(eval_short_path, config=hparams)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.run()

def test_SERAC_owl():
    hparams = SERACMultimodalHparams.from_hparams('hparams/SERAC/owl.yaml')
    # eval_ds = VQADataset(eval_path, config=hparams)
    # eval_ds = VQADataset(eval_short_path, config=hparams)
    # eval_ds = VQADataset("/home/v-hazhong/Datasets/VLKEB/eval_Port_qa_onehop_edit_twice_100.json", config=hparams)
    eval_ds = VQADataset("/home/v-hazhong/Datasets/VLKEB/eval_Port_qa_onehop_edit_twice.json", config=hparams)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.run()

def test_FT_owl():
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/owl2.yaml')
    # eval_ds = VQADataset(eval_path, config=hparams)
    eval_ds = VQADataset("/home/v-hazhong/Datasets/VLKEB/eval_Port_qa_onehop_edit_twice.json", config=hparams)
    # eval_ds = VQADataset(eval_short_path, config=hparams)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.run()

def test_FT_qwen():
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/qwen-vl.yaml')
    # eval_ds = VQADataset(eval_path, config=hparams)
    eval_ds = VQADataset("/home/v-hazhong/Datasets/VLKEB/eval_Port_qa_onehop_edit_twice.json", config=hparams)
    # eval_ds = VQADataset(eval_short_path, config=hparams)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.run()

def test_FT_qwenvis():
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/qwen-vl_vision.yaml')
    eval_ds = VQADataset(eval_path, config=hparams)
    # eval_ds = VQADataset(eval_short_path, config=hparams)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.run()

def test_FT_owlvis():
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/owl2_vision.yaml')
    eval_ds = VQADataset(eval_path, config=hparams)
    # eval_ds = VQADataset(eval_short_path, config=hparams)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.run()

def test_MEND_owl():
    hparams = MENDMultimodalHparams.from_hparams('hparams/MEND/owl2.yaml')
    # eval_ds = VQADataset(eval_path, config=hparams)
    eval_ds = VQADataset("/home/v-hazhong/Datasets/VLKEB/eval_Port_qa_onehop_edit_twice.json", config=hparams)
    # eval_ds = VQADataset(eval_short_path, config=hparams)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.run()

def test_MEND_qwen():
    hparams = MENDMultimodalHparams.from_hparams('hparams/MEND/qwen-vl.yaml')
    eval_ds = VQADataset(eval_path, config=hparams)
    # eval_ds = VQADataset(eval_short_path, config=hparams)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.run()

def test_owl():
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/owl2.yaml')
    eval_ds = VQADataset(eval_path, config=hparams)
    # eval_ds = VQADataset(eval_short_path, config=hparams)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.test_knowledge(log=True)

def test_qwen():
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/qwen-vl.yaml')
    eval_ds = VQADataset(eval_path, config=hparams)
    # eval_ds = VQADataset(eval_short_path, config=hparams)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.test_knowledge(log=True)


def test_seq_FT_owl():
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/owl2.yaml')
    eval_ds = VQADataset("/home/v-hazhong/Datasets/VLKEB/eval_Port_qa_onehop.json", config=hparams)
    # eval_ds = VQADataset('/home/v-hazhong/Datasets/VLKEB/eval_Port_qa_onehop_100.json', config=hparams)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.test_sequencial(log=True, gap_num=gap_num)

def test_seq_FT_qwen():
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/qwen-vl.yaml')
    eval_ds = VQADataset("/home/v-hazhong/Datasets/VLKEB/eval_Port_qa_onehop.json", config=hparams)
    # eval_ds = VQADataset('/home/v-hazhong/Datasets/VLKEB/eval_Port_qa_onehop_100.json', config=hparams)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.test_sequencial(log=True, gap_num=gap_num)

def test_seq_FTvis_owl():
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/owl2_vision.yaml')
    eval_ds = VQADataset("/home/v-hazhong/Datasets/VLKEB/eval_Port_qa_onehop.json", config=hparams)
    # eval_ds = VQADataset('/home/v-hazhong/Datasets/VLKEB/eval_Port_qa_onehop_100.json', config=hparams)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.test_sequencial(log=True, gap_num=gap_num)

def test_seq_FTvis_qwen():
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/qwen-vl_vision.yaml')
    eval_ds = VQADataset("/home/v-hazhong/Datasets/VLKEB/eval_Port_qa_onehop.json", config=hparams)
    # eval_ds = VQADataset('/home/v-hazhong/Datasets/VLKEB/eval_Port_qa_onehop_100.json', config=hparams)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.test_sequencial(log=True, gap_num=gap_num)
    
def test_seq_MEND_owl():
    hparams = MENDMultimodalHparams.from_hparams('hparams/MEND/owl2.yaml')
    # eval_ds = VQADataset("/home/v-hazhong/Datasets/VLKEB/eval_Port_qa_onehop.json", config=hparams)
    eval_ds = VQADataset('/home/v-hazhong/Datasets/VLKEB/eval_Port_qa_onehop_100.json', config=hparams)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.test_sequencial(log=True, gap_num=gap_num)

def test_seq_SERAC_owl():
    hparams = SERACMultimodalHparams.from_hparams('hparams/SERAC/owl.yaml')
    # eval_ds = VQADataset("/home/v-hazhong/Datasets/VLKEB/eval_Port_qa_onehop.json", config=hparams)
    eval_ds = VQADataset('/home/v-hazhong/Datasets/VLKEB/eval_Port_qa_onehop_100.json', config=hparams)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.test_sequencial(log=True, test_num=20, gap_num=gap_num)


if __name__ == '__main__':
    # test_SERAC_qwen()
    # test_SERAC_owl()
    # test_FT_owl()
    # test_FT_qwen()
    # test_FT_qwenvis()
    # test_FT_owlvis()
    test_MEND_owl()
    # test_MEND_qwen()
    # test_owl()
    # test_qwen()
    

    # for gap_num in [10, 20, 50 ,100]:
        # gap_num = 10
        # test_seq_FT_owl()
        # test_seq_FT_qwen()
        # test_seq_FTvis_owl()
        # test_seq_FTvis_qwen()
        # test_seq_MEND_owl()
        # test_seq_SERAC_owl()